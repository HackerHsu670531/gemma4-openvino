"""
===============================================================================
Gemma-4 E2B Pure OpenVINO High-Performance Multimodal Inference Pipeline
===============================================================================
"""
import time
import torch
import numpy as np
import openvino as ov
import argparse
from pathlib import Path
from transformers import AutoProcessor
import warnings
import urllib.request
import json

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str, choices=["fp16", "int8", "int4"], default="fp16", help="Select model precision directory to load")
    return parser.parse_args()

MODEL_ID = "google/gemma-4-E2B-it"
AUDIO_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
IMAGE_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg"

def get_per_layer_inputs_numpy(input_ids, per_layer_weights, hidden_size_per_layer_input, num_layers):
    if per_layer_weights is None:
        return np.zeros((input_ids.shape[0], input_ids.shape[1], 1), dtype=np.float32)
    # Fast table lookup via Memory Mapped numpy
    pl_res = per_layer_weights[input_ids] # [batch, seq, 8960]
    # pl_res has already been scaled during export in pytorch_to_ov_unified_nncf.py (via forward call).
    # DO NOT scale it by multiplying sqrt(hidden_size_per_layer_input) again here, otherwise it will cause numerical overflow, leading to hallucinations and gibberish.
    pl_res = pl_res.astype(np.float32)
    # Reshape to [batch, seq, num_layers, hidden_size_per_layer_input]
    batch, seq = input_ids.shape
    pl_res = pl_res.reshape(batch, seq, num_layers, hidden_size_per_layer_input)
    return pl_res

def extract_features_pure_ov(inputs, ov_models, config_dict, per_layer_weights):
    input_ids = inputs['input_ids'].clone() if isinstance(inputs['input_ids'], torch.Tensor) else torch.tensor(inputs['input_ids'])
    
    # 1. Look up via OpenVINO Word Embeddings
    ov_embed_res = list(ov_models['embed']([input_ids.cpu().numpy()]).values())[0]
    inputs_embeds = torch.from_numpy(ov_embed_res) # [batch, seq_len, hidden_size]

    # 2. Handle Audio encoder embedding special token placeholder
    if 'input_features' in inputs and 'audio' in ov_models:
        input_features = inputs['input_features'].cpu().numpy() if isinstance(inputs['input_features'], torch.Tensor) else inputs['input_features']
        ov_audio_res = list(ov_models['audio']([input_features]).values())[0]
        
        audio_token_id = config_dict.get("audio_token_id", None)
        if audio_token_id is not None and (input_ids == audio_token_id).any():
            audio_features = torch.from_numpy(ov_audio_res).to(inputs_embeds.device)
            audio_mask = (input_ids == audio_token_id)
            expected_audio_len = audio_mask.sum().item()
            audio_features = audio_features.view(-1, audio_features.shape[-1])[:expected_audio_len]
            inputs_embeds[audio_mask] = audio_features.to(inputs_embeds.dtype)

    # 3. Handle Vision encoder embedding special token placeholder
    if 'pixel_values' in inputs and 'vision' in ov_models:
        pix_vals = inputs['pixel_values'].cpu().numpy() if isinstance(inputs['pixel_values'], torch.Tensor) else inputs['pixel_values']
        pix_pos = inputs.get("image_position_ids") if inputs.get("image_position_ids") is not None else inputs.get("pixel_position_ids")
        if pix_pos is None:
            num_patches = pix_vals.shape[1]
            grid_h = grid_w = int(np.sqrt(num_patches))
            rows = torch.arange(grid_h).unsqueeze(1).expand(grid_h, grid_w).flatten() % 128
            cols = torch.arange(grid_w).unsqueeze(0).expand(grid_h, grid_w).flatten() % 128
            pix_pos = torch.stack([rows, cols], dim=-1).unsqueeze(0).to(torch.int64)
        
        pix_pos_np = pix_pos.cpu().numpy() if isinstance(pix_pos, torch.Tensor) else pix_pos
        ov_vision_res = list(ov_models['vision']([pix_vals, pix_pos_np]).values())[0]
        
        image_token_id = config_dict.get("image_token_id", None)
        if image_token_id is not None and (input_ids == image_token_id).any():
            ov_image_features = torch.from_numpy(ov_vision_res).to(inputs_embeds.device)
            image_mask = (input_ids == image_token_id)
            expected_image_len = image_mask.sum().item()
            image_features = ov_image_features.view(-1, ov_image_features.shape[-1])[:expected_image_len]
            inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)

    # 4. Prepare LLM Input IDs (replace special tokens with pad_token_id)
    llm_input_ids = input_ids.clone()
    pad_id = config_dict.get("pad_token_id", 0)
    audio_token_id = config_dict.get("audio_token_id", None)
    if audio_token_id is not None:
        llm_input_ids[llm_input_ids == audio_token_id] = pad_id
    image_token_id = config_dict.get("image_token_id", None)
    if image_token_id is not None:
        llm_input_ids[llm_input_ids == image_token_id] = pad_id

    # 5. Compute Per-Layer Inputs using Memory Mapping
    per_layer = None
    if per_layer_weights is not None:
        per_layer_np = get_per_layer_inputs_numpy(
            llm_input_ids.cpu().numpy(),
            per_layer_weights,
            config_dict["hidden_size_per_layer_input"],
            config_dict["num_hidden_layers"]
        )
        per_layer = torch.from_numpy(per_layer_np)

    return inputs_embeds, per_layer, llm_input_ids

def generate_loop_pure_ov(processor, embeds, per_layer, llm_input_ids, ov_decoder, config_dict, per_layer_weights, ov_models, max_new_tokens=40):
    cur_embeds = embeds.clone()
    cur_per_layer = per_layer.clone() if per_layer is not None else torch.zeros(1)
    cur_input_ids = llm_input_ids.clone()
    cur_pos_ids = torch.arange(cur_embeds.shape[1], device=cur_embeds.device).unsqueeze(0)

    window = config_dict.get("sliding_window", 4096)
    eos_token_id = config_dict.get("eos_token_id")

    generated_tokens = []

    # ⏱️ Record performance
    start_time = time.perf_counter()
    ttft = 0.0

    for step in range(max_new_tokens):
        L_old = cur_embeds.shape[1]
        causal = torch.tril(torch.ones((L_old, L_old), dtype=torch.bool, device=cur_embeds.device))
        cur_mask_f = torch.where(causal, 0.0, -65500.0).to(torch.float32).unsqueeze(0).unsqueeze(0)
        cur_mask_s = torch.where(torch.triu(causal, diagonal=-window+1), 0.0, -65500.0).to(torch.float32).unsqueeze(0).unsqueeze(0)

        # Execute "pure OpenVINO" high-performance decoding inference with integrated internal RoPE and LM Head
        ov_inputs_list = [
            cur_embeds.cpu().numpy(),
            cur_per_layer.cpu().numpy(),
            cur_mask_f.cpu().numpy(),
            cur_mask_s.cpu().numpy(),
            cur_pos_ids.cpu().numpy()
        ]
        
        ov_res = list(ov_decoder(ov_inputs_list).values())[0] # decoder outputs the argmax of the optimal token directly with shape [batch, 1]
        next_token = int(ov_res[0, 0])

        generated_tokens.append(next_token)

        # Record Time to First Token (TTFT)
        if step == 0:
            ttft = time.perf_counter() - start_time

        if next_token == eos_token_id:
            break

        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=cur_embeds.device)
        cur_input_ids = torch.cat([cur_input_ids, next_token_tensor], dim=1)
        
        # Get next token's embedding via OpenVINO
        next_embed_res = list(ov_models['embed']([next_token_tensor.cpu().numpy()]).values())[0]
        new_token_embeds = torch.from_numpy(next_embed_res).to(cur_embeds.device)
            
        cur_embeds = torch.cat([cur_embeds, new_token_embeds], dim=1)
        cur_pos_ids = torch.cat([cur_pos_ids, cur_pos_ids[:, -1:] + 1], dim=1)

        # Get next token's Per-Layer Inputs
        if per_layer_weights is not None:
            next_per_layer_res = get_per_layer_inputs_numpy(
                next_token_tensor.cpu().numpy(),
                per_layer_weights,
                config_dict["hidden_size_per_layer_input"],
                config_dict["num_hidden_layers"]
            )
            new_per_layer = torch.from_numpy(next_per_layer_res).to(cur_embeds.device)
            cur_per_layer = torch.cat([cur_per_layer, new_per_layer], dim=1)

    gen_time = time.perf_counter() - start_time
    final_text = processor.decode(generated_tokens, skip_special_tokens=True).replace("\n", " ").strip()
    return final_text, len(generated_tokens), ttft, gen_time

def run_pure_ov_inference(test_title, inputs, processor, ov_models, ov_decoder, config_dict, per_layer_weights, max_new_tokens=40):
    print(f"\n\n{'='*80}")
    print(f"Starting test suite: {test_title}")
    print("="*80)

    # Feature extraction progress
    t0 = time.perf_counter()
    embeds_ov, per_layer_ov, llm_ids_ov = extract_features_pure_ov(inputs, ov_models, config_dict, per_layer_weights)
    ext_time = time.perf_counter() - t0

    # Execute OpenVINO generation loop
    text_ov, n_tokens, ttft, gen_time = generate_loop_pure_ov(
        processor, embeds_ov, per_layer_ov, llm_ids_ov, ov_decoder, 
        config_dict, per_layer_weights, ov_models, max_new_tokens=max_new_tokens
    )
    tps = n_tokens / gen_time if gen_time > 0 else 0

    print(f"\nOpenVINO Output: '{text_ov}'")
    print(f"Performance Statistics (Pure OpenVINO Pipeline):")
    print(f"Preprocessing feature extraction: {ext_time:.4f}s")
    print(f"Time to First Token (TTFT): {ttft:.4f}s")
    print(f"Total generation time: {gen_time:.4f}s")
    print(f"Generated Tokens count: {n_tokens}")
    print(f"Average inference speed: {tps:.2f} tokens/s")
    print("="*80)

def main():
    args = parse_args()
    OV_DIR = Path(f"./gemma4-openvino-{args.precision}")

    print("="*80)
    print(f"Launching Gemma-4 Pure OpenVINO High-Performance Multimodal Inference Pipeline")
    print(f"Target precision: {args.precision.upper()}")
    print(f"Model directory: {OV_DIR}")
    print("="*80)

    config_path = OV_DIR / "config.json"
    if not config_path.exists():
        print(f"Cannot find configuration file {config_path}! Please run conversion first: python pytorch_to_ov_unified_nncf.py --precision {args.precision}")
        return

    # 1. Read configuration file to load model parameters, no PyTorch weights are loaded
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    print("Successfully loaded config.json")

    print("\nInitializing Preprocessor / Tokenizer...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    core = ov.Core()
    ov_models = {}

    print("\nLoading OpenVINO modules...")
    if (OV_DIR / "audio_encoder.xml").exists():
        print("Loaded Audio Encoder (OpenVINO)")
        ov_models['audio'] = core.compile_model(OV_DIR / "audio_encoder.xml", "CPU")
    if (OV_DIR / "embed_tokens.xml").exists():
        print("Loaded Embed Tokens (OpenVINO)")
        ov_models['embed'] = core.compile_model(OV_DIR / "embed_tokens.xml", "CPU")
    if (OV_DIR / "vision_encoder.xml").exists():
        print("Loaded Vision Encoder (OpenVINO)")
        ov_models['vision'] = core.compile_model(OV_DIR / "vision_encoder.xml", "CPU")

    print("Loaded Decoder Core (OpenVINO)")
    ov_decoder = core.compile_model(OV_DIR / "decoder_embeds.xml", "CPU")

    # 2. Load the huge Per-Layer Embedding weight matrix via Memory Mapping (mmap) to save 2.4 GB of RAM
    per_layer_weights_path = OV_DIR / "per_layer_weights.npy"
    if per_layer_weights_path.exists():
        print("[Mmap Memory Optimization] Loading Memory-Mapped Per-Layer Embedding weights...")
        per_layer_weights = np.load(per_layer_weights_path, mmap_mode="r")
    else:
        print("Cannot find per_layer_weights.npy file, PLE feature will not be used")
        per_layer_weights = None

    if not Path("test_audio.wav").exists(): urllib.request.urlretrieve(AUDIO_URL, "test_audio.wav")
    if not Path("test_image.jpeg").exists(): urllib.request.urlretrieve(IMAGE_URL, "test_image.jpeg")

    print("\nPreparing multimodal inputs...")
    
    asr_msgs = [{"role": "user", "content": [{"type": "audio", "audio": "test_audio.wav"}, {"type": "text", "text": "Transcribe."}]}]
    asr_inputs = processor.apply_chat_template(asr_msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)

    ast_msgs = [{"role": "user", "content": [{"type": "audio", "audio": "test_audio.wav"}, {"type": "text", "text": "Please translate this audio to Traditional Chinese:"}]}]
    ast_inputs = processor.apply_chat_template(ast_msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)

    image_msgs = [{"role": "user", "content": [{"type": "image", "url": "test_image.jpeg"}, {"type": "text", "text": "Describe."}]}]
    image_inputs = processor.apply_chat_template(image_msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)

    run_pure_ov_inference("Audio Recognition Test (ASR - Transcribe)", asr_inputs, processor, ov_models, ov_decoder, config_dict, per_layer_weights, max_new_tokens=40)
    run_pure_ov_inference("Audio Translation Test (AST - Translate to Chinese)", ast_inputs, processor, ov_models, ov_decoder, config_dict, per_layer_weights, max_new_tokens=60)
    run_pure_ov_inference("Image Inference Test (Image Describe)", image_inputs, processor, ov_models, ov_decoder, config_dict, per_layer_weights, max_new_tokens=60)

if __name__ == "__main__":
    main()
