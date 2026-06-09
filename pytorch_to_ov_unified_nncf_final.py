"""
===============================================================================
Gemma-4 E2B PyTorch Native Dismantling and OpenVINO Conversion Script
(Full-Pipeline Custom Precision Quantization Version)
===============================================================================
"""
import os
import torch
import urllib.request
import numpy as np
import openvino as ov
import argparse
import nncf
from pathlib import Path
from transformers import AutoProcessor, AutoModelForMultimodalLM
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recreate", action="store_true", help="Force recreation of converted models")
    parser.add_argument("--precision", type=str, choices=["fp16", "int8", "int4"], default="fp16", help="Select quantization precision for the entire pipeline")
    return parser.parse_args()

MODEL_ID = "google/gemma-4-E2B-it"
AUDIO_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
IMAGE_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg"

# =========================================================================
# Global Interceptor: Protect all modules from HF masking bug when tracing inside OpenVINO
# =========================================================================
import transformers.masking_utils
_orig_sdpa_mask = getattr(transformers.masking_utils, "sdpa_mask", None)
_orig_create_bidir_mask = getattr(transformers.masking_utils, "create_bidirectional_mask", None)

try:
    import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling
    _orig_g4_sdpa = getattr(gemma4_modeling, "sdpa_mask", None)
    _orig_g4_bidir = getattr(gemma4_modeling, "create_bidirectional_mask", None)
except:
    gemma4_modeling = None

def apply_tracing_patches():
    transformers.masking_utils.sdpa_mask = lambda *args, **kwargs: None
    transformers.masking_utils.create_bidirectional_mask = lambda *args, **kwargs: None
    if gemma4_modeling:
        gemma4_modeling.sdpa_mask = lambda *args, **kwargs: None
        gemma4_modeling.create_bidirectional_mask = lambda *args, **kwargs: None

def restore_tracing_patches():
    transformers.masking_utils.sdpa_mask = _orig_sdpa_mask
    transformers.masking_utils.create_bidirectional_mask = _orig_create_bidir_mask
    if gemma4_modeling:
        gemma4_modeling.sdpa_mask = _orig_g4_sdpa
        gemma4_modeling.create_bidirectional_mask = _orig_g4_bidir

class MinimalAudioEncoderWrapper(torch.nn.Module):
    def __init__(self, tower, embed):
        super().__init__()
        self.subsample_conv = tower.subsample_conv_projection
        self.rel_pos = tower.rel_pos_enc
        self.output_proj = tower.output_proj
        self.embed_norm = embed.embedding_pre_projection_norm
        self.embed_proj = embed.embedding_projection
        self.layers = torch.nn.ModuleList([layer for layer in tower.layers])
    def forward(self, input_features):
        hidden_states, _ = self.subsample_conv(input_features, None)
        rel_pos = self.rel_pos(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=None, position_embeddings=rel_pos)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.embed_norm(hidden_states)
        return self.embed_proj(hidden_states)

class MinimalVisionEncoderWrapper(torch.nn.Module):
    def __init__(self, tower, embed):
        super().__init__()
        self.tower = tower
        self.embed_norm = embed.embedding_pre_projection_norm
        self.embed_proj = embed.embedding_projection
    def forward(self, pixel_values, pixel_position_ids):
        tower_out = self.tower(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids)
        hidden_states = tower_out.last_hidden_state if hasattr(tower_out, 'last_hidden_state') else (tower_out[0] if isinstance(tower_out, tuple) else tower_out)
        hidden_states = self.embed_norm(hidden_states)
        return self.embed_proj(hidden_states)

# Stateless Multi-Layer Cache
class MiniCache:
    def __init__(self):
        self.shared_layers = {}
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states

class UltimateDecoderWrapper(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.lm = full_model.model.language_model if hasattr(full_model.model, 'language_model') else full_model.model
        self.num_layers = len(self.lm.layers)
        self.layer_types = getattr(full_model.config.text_config if hasattr(full_model.config, 'text_config') else full_model.config, 'layer_types', None)
        
        self.rope = self.lm.rotary_emb
        self.lm_head = getattr(self.lm, "lm_head", getattr(full_model, "lm_head", getattr(full_model.model, "lm_head", None)))
        self.softcap = getattr(self.lm.config, "final_logit_softcapping", None)
        self.head_dim = getattr(self.lm.config, 'head_dim', 256)

    def forward(self, inputs_embeds, per_layer_inputs, mask_full, mask_sliding, position_ids):
        dummy_x = inputs_embeds[..., :self.head_dim]
        rope_attrs = dir(self.rope)
        full_str = "full_attention" if "full_attention_inv_freq" in rope_attrs else "full"
        sliding_str = "sliding_attention" if "sliding_attention_inv_freq" in rope_attrs else "sliding"

        try: cos_f, sin_f = self.rope(dummy_x, position_ids, layer_type=full_str)
        except: cos_f, sin_f = self.rope(dummy_x, position_ids)
        try: cos_s, sin_s = self.rope(dummy_x, position_ids, layer_type=sliding_str)
        except: cos_s, sin_s = cos_f, sin_f

        hidden_states = inputs_embeds
        if getattr(self.lm, "hidden_size_per_layer_input", 0) and per_layer_inputs is not None and per_layer_inputs.numel() > 1:
            if hasattr(self.lm, "project_per_layer_inputs"):
                per_layer_inputs = self.lm.project_per_layer_inputs(hidden_states, per_layer_inputs)

        causal_mask_dict = {"full_attention": mask_full, "sliding_attention": mask_sliding, "full": mask_full, "sliding": mask_sliding}
        pos_emb_dict = {"full_attention": (cos_f, sin_f), "sliding_attention": (cos_s, sin_s), "full": (cos_f, sin_f), "sliding": (cos_s, sin_s)}
        past_key_values = MiniCache()

        for layer_idx in range(self.num_layers):
            layer = self.lm.layers[layer_idx]
            l_type = self.layer_types[layer_idx] if self.layer_types else "full"

            current_per_layer_input = None
            if per_layer_inputs is not None and per_layer_inputs.numel() > 1:
                if per_layer_inputs.ndim >= 4: current_per_layer_input = per_layer_inputs[:, :, layer_idx, :]
                elif per_layer_inputs.ndim == 3: current_per_layer_input = per_layer_inputs[..., layer_idx, :]

            layer_outputs = layer(
                hidden_states, attention_mask=causal_mask_dict.get(l_type, mask_full), position_ids=position_ids,
                position_embeddings=pos_emb_dict.get(l_type, (cos_f, sin_f)), per_layer_input=current_per_layer_input,
                past_key_values=past_key_values, use_cache=False, output_attentions=False,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        hidden_states = self.lm.norm(hidden_states)

        last_hidden = hidden_states[:, -1:, :]
        logits = self.lm_head(last_hidden)

        if self.softcap is not None:
            logits = logits / self.softcap
            logits = torch.tanh(logits) * self.softcap

        return torch.argmax(logits, dim=-1)

def apply_quantization(ov_model, precision, model_name):
    """
    Common quantization function: converts the specified module to INT8 or INT4 format.
    """
    if precision == "fp16":
        return ov_model

    print(f"      🗜️ Starting NNCF {precision.upper()} quantization compression for ({model_name})...")
    if precision == "int8":
        compressed_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    elif precision == "int4":
        # Keep 80/20 mixed-precision for Decoder to prevent quality degradation, other Encoders are fully compressed (ratio=1.0)
        ratio = 0.6 if model_name == "Decoder" else 1.0
        compressed_model = nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT4_ASYM,
            group_size=128,
            ratio=ratio
        )
    print(f"OK: {model_name} {precision.upper()} compression completed successfully!")
    return compressed_model


def main():
    args = parse_args()
    OV_DIR = Path(f"./gemma4-openvino-{args.precision}")

    print("="*75)
    print(f"Launching Gemma-4 PyTorch Full-Line Decompilation and OpenVINO Conversion Engine")
    print(f"Target precision: {args.precision.upper()}")
    print(f"Output directory: {OV_DIR}")
    print("="*75)

    if not Path("test_audio.wav").exists(): urllib.request.urlretrieve(AUDIO_URL, "test_audio.wav")
    if not Path("test_image.jpeg").exists(): urllib.request.urlretrieve(IMAGE_URL, "test_image.jpeg")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32).eval()

    audio_msgs = [{"role": "user", "content": [{"type": "audio", "audio": "test_audio.wav"}, {"type": "text", "text": "Transcribe."}]}]
    audio_inputs = processor.apply_chat_template(audio_msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)

    image_msgs = [{"role": "user", "content": [{"type": "image", "url": "test_image.jpeg"}, {"type": "text", "text": "Describe."}]}]
    image_inputs = processor.apply_chat_template(image_msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)

    OV_DIR.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    core_model = model.model
    orig_lm = core_model.language_model if hasattr(core_model, 'language_model') else core_model

    #Exporting settings and weights for pure OpenVINO inference...
    import json
    config_dict = {
        "vocab_size": int(orig_lm.config.vocab_size),
        "pad_token_id": int(getattr(orig_lm.config, 'pad_token_id', 0) if getattr(orig_lm.config, 'pad_token_id', None) is not None else 0),
        "audio_token_id": int(getattr(model.config, 'audio_token_id', 0) if getattr(model.config, 'audio_token_id', None) is not None else 0),
        "image_token_id": int(getattr(model.config, 'image_token_id', 0) if getattr(model.config, 'image_token_id', None) is not None else 0),
        "eos_token_id": int(processor.tokenizer.eos_token_id),
        "hidden_size_per_layer_input": int(getattr(orig_lm, 'hidden_size_per_layer_input', 0)),
        "num_hidden_layers": int(orig_lm.config.num_hidden_layers if hasattr(orig_lm.config, 'num_hidden_layers') else 0),
        "head_dim": int(getattr(orig_lm.config, 'head_dim', 256)),
        "hidden_size": int(getattr(orig_lm.config, 'hidden_size', 3072)),
        "sliding_window": int(getattr(orig_lm.config, 'sliding_window', 4096)),
        "final_logit_softcapping": float(orig_lm.config.final_logit_softcapping) if getattr(orig_lm.config, 'final_logit_softcapping', None) is not None else None
    }
    with open(OV_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
    print("Successfully saved config.json configuration file")

    if hasattr(orig_lm, 'embed_tokens_per_layer') and orig_lm.embed_tokens_per_layer is not None:
        print("Exporting Per-Layer Embedding Weights (For Memory-Mapping)...")
        vocab_size = orig_lm.config.vocab_size
        all_ids = torch.arange(vocab_size, dtype=torch.long, device="cpu")
        with torch.no_grad():
            pl_weights = orig_lm.embed_tokens_per_layer(all_ids).half().cpu().numpy()
        np.save(OV_DIR / "per_layer_weights.npy", pl_weights)
        print("Per-Layer Embedding Weights exported successfully!")

    print("\n[Step 2] Launching module-by-module compiler/converter pipeline...")

    if not (OV_DIR / "embed_tokens.xml").exists() or args.force_recreate:
        print("Converting Embed Tokens...")
        try:
            dummy_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
            ov_embeds = ov.convert_model(orig_lm.embed_tokens, example_input=(dummy_ids,))
            ov_embeds = apply_quantization(ov_embeds, args.precision, "Embed Tokens")
            ov.save_model(ov_embeds, OV_DIR / "embed_tokens.xml", compress_to_fp16=True)
            print("Embed Tokens converted and saved successfully!")
        except Exception as e: print(f"Embed Tokens conversion failed: {e}")

    if not (OV_DIR / "audio_encoder.xml").exists() or args.force_recreate:
        print("Converting Audio Encoder...")
        try:
            ov_audio = ov.convert_model(MinimalAudioEncoderWrapper(core_model.audio_tower, core_model.embed_audio).eval(), example_input=(audio_inputs.get("input_features"),))
            ov_audio = apply_quantization(ov_audio, args.precision, "Audio Encoder")
            ov.save_model(ov_audio, OV_DIR / "audio_encoder.xml", compress_to_fp16=True)
            print("Audio Encoder converted and saved successfully!")
        except Exception as e: print(f"Audio Encoder conversion failed: {e}")

    if not (OV_DIR / "vision_encoder.xml").exists() or args.force_recreate:
        print("Converting Vision Encoder...")
        try:
            pix_vals = image_inputs.get("pixel_values")
            pix_pos = image_inputs.get("image_position_ids") if image_inputs.get("image_position_ids") is not None else image_inputs.get("pixel_position_ids")
            if pix_pos is None:
                num_patches = pix_vals.shape[1]
                grid_h = grid_w = int(np.sqrt(num_patches))
                rows = torch.arange(grid_h, device=pix_vals.device).unsqueeze(1).expand(grid_h, grid_w).flatten() % 128
                cols = torch.arange(grid_w, device=pix_vals.device).unsqueeze(0).expand(grid_h, grid_w).flatten() % 128
                pix_pos = torch.stack([rows, cols], dim=-1).unsqueeze(0).to(torch.int64)

            apply_tracing_patches()
            ov_vision = ov.convert_model(MinimalVisionEncoderWrapper(core_model.vision_tower, core_model.embed_vision).eval(), example_input=(pix_vals, pix_pos))
            restore_tracing_patches()

            ov_vision = apply_quantization(ov_vision, args.precision, "Vision Encoder")
            ov.save_model(ov_vision, OV_DIR / "vision_encoder.xml", compress_to_fp16=True)
            print("Vision Encoder converted and saved successfully!")
        except Exception as e:
            restore_tracing_patches()
            print(f"Vision Encoder conversion failed: {e}")

    if not (OV_DIR / "decoder_embeds.xml").exists() or args.force_recreate:
        print("Converting Decoder (Multimodal Inference Core)...")
        try:
            embed_wrapper = UltimateDecoderWrapper(model).eval()
            seq_len = 16
            dummy_input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            dummy_embeds = orig_lm.embed_tokens(dummy_input_ids)
            dummy_per_layer = orig_lm.get_per_layer_inputs(input_ids=dummy_input_ids, inputs_embeds=dummy_embeds) if hasattr(orig_lm, 'get_per_layer_inputs') else torch.zeros(1)

            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            dummy_mask_full = torch.where(causal, 0.0, -65500.0).to(torch.float32).unsqueeze(0).unsqueeze(0)
            dummy_mask_sliding = torch.where(torch.triu(causal, diagonal=-4096+1), 0.0, -65500.0).to(torch.float32).unsqueeze(0).unsqueeze(0)
            dummy_pos = torch.arange(seq_len).unsqueeze(0)

            apply_tracing_patches()
            ov_decoder = ov.convert_model(embed_wrapper, example_input=(dummy_embeds, dummy_per_layer, dummy_mask_full, dummy_mask_sliding, dummy_pos))
            restore_tracing_patches()

            ov_decoder = apply_quantization(ov_decoder, args.precision, "Decoder")
            ov.save_model(ov_decoder, OV_DIR / "decoder_embeds.xml", compress_to_fp16=True)
            print("Decoder converted and saved successfully!")

        except Exception as e:
            restore_tracing_patches()
            print(f"Decoder conversion failed: {e}")

    print(f"\nOpenVINO conversion process finished successfully! Run `python infer_openvino_audio_vision_asr_ast_nncf_final.py --precision {args.precision}` for diagnostic testing & benchmarks.")

if __name__ == "__main__":
    main()