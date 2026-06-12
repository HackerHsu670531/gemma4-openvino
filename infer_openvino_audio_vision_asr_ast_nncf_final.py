"""
===============================================================================
Gemma-4 E2B Pure OpenVINO High-Performance NPU Static Shape Multimodal Inference Pipeline
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
    parser.add_argument("--device", type=str, default="NPU", choices=["CPU", "GPU", "NPU"], help="Target compile device for static shape models")
    return parser.parse_args()

MODEL_ID = "google/gemma-4-E2B-it"
AUDIO_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
IMAGE_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg"

def get_per_layer_inputs_numpy(input_ids, per_layer_weights, hidden_size_per_layer_input, num_layers):
    if per_layer_weights is None:
        return np.zeros((input_ids.shape[0], input_ids.shape[1], 1), dtype=np.float32)
    # Fast table lookup via Memory Mapped numpy
    pl_res = per_layer_weights[input_ids] # [batch, seq, 8960]
    pl_res = pl_res.astype(np.float32)
    # Reshape to [batch, seq, num_layers, hidden_size_per_layer_input]
    batch, seq = input_ids.shape
    pl_res = pl_res.reshape(batch, seq, num_layers, hidden_size_per_layer_input)
    return pl_res

def extract_features_pure_ov(inputs, core, ov_models, config_dict, per_layer_weights, device="NPU"):
    input_ids = inputs['input_ids'].clone() if isinstance(inputs['input_ids'], torch.Tensor) else torch.tensor(inputs['input_ids'])
    input_ids_np = input_ids.cpu().numpy()
    
    # 1. Embed
    if 'embed' in ov_models:
        embed_model = core.read_model(ov_models['embed'])
        if device == "NPU":
            print(f"      [NPU Dynamic To Static] Modifying Embed model to static shape {input_ids_np.shape}...")
            embed_model.reshape({embed_model.inputs[0]: tuple(input_ids_np.shape)})
        try:
            compiled_embed = core.compile_model(embed_model, device)
        except Exception as e:
            print(f"      ⚠️ [NPU Compile Fallback] Failed to compile Embed on {device} ({e}). Falling back to CPU...")
            compiled_embed = core.compile_model(embed_model, "CPU")
        ov_embed_res = list(compiled_embed([input_ids_np]).values())[0]
    else:
        raise FileNotFoundError("Missing embed_tokens.xml model which is required.")
        
    inputs_embeds = torch.from_numpy(ov_embed_res)

    # 2. Audio
    if 'input_features' in inputs and 'audio' in ov_models:
        input_features = inputs['input_features'].cpu().numpy() if isinstance(inputs['input_features'], torch.Tensor) else inputs['input_features']
        audio_model = core.read_model(ov_models['audio'])
        
        if device == "NPU" or device == "GPU":
            print(f"      [NPU and GPU Dynamic To Static] Modifying Audio Encoder to static shape {input_features.shape}...")
            audio_model.reshape({audio_model.inputs[0]: tuple(input_features.shape)})
            
        try:
            compiled_audio = core.compile_model(audio_model, device)
        except Exception as e:
            print(f"      ⚠️ [NPU and GPU Compile Fallback] Failed to compile Audio Encoder on {device} ({e}). Falling back to CPU...")
            compiled_audio = core.compile_model(audio_model, "CPU")
        
        ov_audio_res = list(compiled_audio([input_features]).values())[0]
        
        audio_token_id = config_dict.get("audio_token_id", None)
        if audio_token_id is not None and (input_ids == audio_token_id).any():
            audio_features = torch.from_numpy(ov_audio_res).to(inputs_embeds.device)
            audio_mask = (input_ids == audio_token_id)
            expected_audio_len = audio_mask.sum().item()
            audio_features = audio_features.view(-1, audio_features.shape[-1])[:expected_audio_len]
            inputs_embeds[audio_mask] = audio_features.to(inputs_embeds.dtype)

    # 3. Vision
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
        vision_model = core.read_model(ov_models['vision'])
        
        if device == "NPU" or device == "GPU":
            print(f"      [NPU and GPU Dynamic To Static] Modifying Vision Encoder to static shapes {pix_vals.shape} and {pix_pos_np.shape}...")
            vision_model.reshape({
                vision_model.inputs[0]: tuple(pix_vals.shape),
                vision_model.inputs[1]: tuple(pix_pos_np.shape)
            })
            
        try:
            compiled_vision = core.compile_model(vision_model, device)
        except Exception as e:
            print(f"      ⚠️ [NPU and GPU Compile Fallback] Failed to compile Vision Encoder on {device} ({e}). Falling back to CPU...")
            compiled_vision = core.compile_model(vision_model, "CPU")
        
        ov_vision_res = list(compiled_vision([pix_vals, pix_pos_np]).values())[0]
        
        image_token_id = config_dict.get("image_token_id", None)
        if image_token_id is not None and (input_ids == image_token_id).any():
            ov_image_features = torch.from_numpy(ov_vision_res).to(inputs_embeds.device)
            image_mask = (input_ids == image_token_id)
            expected_image_len = image_mask.sum().item()
            image_features = ov_image_features.view(-1, ov_image_features.shape[-1])[:expected_image_len]
            inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)

    # 4. Prepare LLM Input IDs
    llm_input_ids = input_ids.clone()
    pad_id = config_dict.get("pad_token_id", 0)
    for t_id in [config_dict.get("audio_token_id"), config_dict.get("image_token_id")]:
        if t_id is not None:
            llm_input_ids[llm_input_ids == t_id] = pad_id

    # 5. Compute Per-Layer Inputs
    per_layer = None
    if per_layer_weights is not None:
        per_layer_np = get_per_layer_inputs_numpy(llm_input_ids.cpu().numpy(), per_layer_weights, config_dict["hidden_size_per_layer_input"], config_dict["num_hidden_layers"])
        per_layer = torch.from_numpy(per_layer_np)

    return inputs_embeds, per_layer, llm_input_ids

def generate_loop_pure_ov(processor, embeds, per_layer, llm_input_ids, core, ov_decoder, MAX_LEN, ov_models, config_dict, per_layer_weights, device="NPU", max_new_tokens=40):
    cur_embeds = embeds.clone()
    cur_per_layer = per_layer.clone() if per_layer is not None else torch.zeros(1)
    cur_input_ids = llm_input_ids.clone()
    cur_pos_ids = torch.arange(cur_embeds.shape[1], device=cur_embeds.device).unsqueeze(0)

    window = config_dict.get("sliding_window", 4096)
    eos_token_id = config_dict.get("eos_token_id")
    generated_tokens = []
    start_time = time.perf_counter()
    ttft = 0.0

    embed_model_step = core.read_model(ov_models['embed'])
    if device == "NPU":
        embed_model_step.reshape({embed_model_step.inputs[0]: (1, 1)})
        
    try:
        compiled_embed_step = core.compile_model(embed_model_step, device)
    except Exception as e:
        print(f"      ⚠️ [NPU Compile Fallback] Failed to compile Step Embed on {device} ({e}). Falling back to CPU...")
        compiled_embed_step = core.compile_model(embed_model_step, "CPU")

    pad_id = config_dict.get("pad_token_id", 0)
    pad_embed_single = list(compiled_embed_step([[[pad_id]]]).values())[0]

    for step in range(max_new_tokens):
        L_old = cur_embeds.shape[1]
        is_static = (device == "NPU")
        
        # 動態形狀 (CPU/GPU) 不需要 Pad，直接用 L_old
        pad_len = (MAX_LEN - L_old) if is_static else 0
        seq_dim = MAX_LEN if is_static else L_old
        
        causal = torch.tril(torch.ones((seq_dim, seq_dim), dtype=torch.bool, device=cur_embeds.device))
        cur_mask_f = torch.where(causal, 0.0, -65500.0).to(torch.float32)
        cur_mask_s = torch.where(torch.triu(causal, diagonal=-window+1), 0.0, -65500.0).to(torch.float32)
        
        if pad_len > 0:
            cur_mask_f[:, :pad_len] = -65500.0
            cur_mask_s[:, :pad_len] = -65500.0
            
        cur_mask_f_np = cur_mask_f.unsqueeze(0).unsqueeze(0).cpu().numpy()
        cur_mask_s_np = cur_mask_s.unsqueeze(0).unsqueeze(0).cpu().numpy()

        cur_embeds_np = cur_embeds.cpu().numpy()
        if pad_len > 0:
            pad_embeds = np.repeat(pad_embed_single, pad_len, axis=1)
            cur_embeds_np_padded = np.concatenate([pad_embeds, cur_embeds_np], axis=1)
        else:
            cur_embeds_np_padded = cur_embeds_np

        if per_layer_weights is not None:
            cur_per_layer_np = cur_per_layer.cpu().numpy()
            if pad_len > 0:
                pad_per_layer = np.zeros((1, pad_len, config_dict["num_hidden_layers"], config_dict["hidden_size_per_layer_input"]), dtype=np.float32)
                cur_per_layer_np_padded = np.concatenate([pad_per_layer, cur_per_layer_np], axis=1)
            else:
                cur_per_layer_np_padded = cur_per_layer_np
        else:
            cur_per_layer_np_padded = np.zeros((1,), dtype=np.float32)

        cur_pos_ids_np = cur_pos_ids.cpu().numpy()
        if pad_len > 0:
            pad_pos_ids = np.zeros((1, pad_len), dtype=np.int64)
            cur_pos_ids_np_padded = np.concatenate([pad_pos_ids, cur_pos_ids_np], axis=1)
        else:
            cur_pos_ids_np_padded = cur_pos_ids_np

        ov_inputs_list = [
            cur_embeds_np_padded,
            cur_per_layer_np_padded,
            cur_mask_f_np,
            cur_mask_s_np,
            cur_pos_ids_np_padded
        ]
        
        ov_res = list(ov_decoder(ov_inputs_list).values())[0]
        next_token = int(ov_res[0, 0])
        generated_tokens.append(next_token)

        if step == 0:
            ttft = time.perf_counter() - start_time

        if next_token == eos_token_id:
            break

        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=cur_embeds.device)
        cur_input_ids = torch.cat([cur_input_ids, next_token_tensor], dim=1)
        
        next_embed_res = list(compiled_embed_step([next_token_tensor.cpu().numpy()]).values())[0]
        new_token_embeds = torch.from_numpy(next_embed_res).to(cur_embeds.device)
            
        cur_embeds = torch.cat([cur_embeds, new_token_embeds], dim=1)
        cur_pos_ids = torch.cat([cur_pos_ids, cur_pos_ids[:, -1:] + 1], dim=1)

        if per_layer_weights is not None:
            next_per_layer_res = get_per_layer_inputs_numpy(
                next_token_tensor.cpu().numpy(),
                per_layer_weights, config_dict["hidden_size_per_layer_input"], config_dict["num_hidden_layers"]
            )
            new_per_layer = torch.from_numpy(next_per_layer_res).to(cur_embeds.device)
            cur_per_layer = torch.cat([cur_per_layer, new_per_layer], dim=1)

    gen_time = time.perf_counter() - start_time
    final_text = processor.decode(generated_tokens, skip_special_tokens=True).replace("\n", " ").strip()
    return final_text, len(generated_tokens), ttft, gen_time

def run_pure_ov_inference(test_title, inputs, processor, core, ov_models, config_dict, per_layer_weights, device="NPU", max_new_tokens=40):
    print(f"\n\n{'='*80}")
    print(f"Starting test suite: {test_title}")
    print("="*80)

    t0 = time.perf_counter()
    embeds_ov, per_layer_ov, llm_ids_ov = extract_features_pure_ov(inputs, core, ov_models, config_dict, per_layer_weights, device=device)
    ext_time = time.perf_counter() - t0

    prompt_len = embeds_ov.shape[1]
    MAX_LEN = prompt_len + max_new_tokens

    decoder_model = core.read_model(ov_models['decoder'])

    if device == "NPU":
        print(f"      [NPU Dynamic To Static] Setting static MAX_LEN = {MAX_LEN} for decoder compilation...")
        embed_shape = (1, MAX_LEN, config_dict["hidden_size"])
        if per_layer_weights is not None:
            per_layer_shape = (1, MAX_LEN, config_dict["num_hidden_layers"], config_dict["hidden_size_per_layer_input"])
        else:
            per_layer_shape = (1,)
            
        mask_full_shape = (1, 1, MAX_LEN, MAX_LEN)
        mask_sliding_shape = (1, 1, MAX_LEN, MAX_LEN)
        pos_ids_shape = (1, MAX_LEN)

        reshape_dict = {
            decoder_model.inputs[0]: embed_shape,
            decoder_model.inputs[1]: per_layer_shape,
            decoder_model.inputs[2]: mask_full_shape,
            decoder_model.inputs[3]: mask_sliding_shape,
            decoder_model.inputs[4]: pos_ids_shape
        }
        decoder_model.reshape(reshape_dict)
    else:
        print(f"      [CPU/GPU] Using Native Dynamic Shapes for compilation...")
    
    t_comp_start = time.perf_counter()
    print(f"      Compiling Decoder Core to {device}...")
    try:
        ov_decoder = core.compile_model(decoder_model, device)
    except Exception as e:
        print(f"      ⚠️ [NPU Compile Fallback] Failed to compile Decoder Core on {device} ({e}). Falling back to CPU...")
        ov_decoder = core.compile_model(decoder_model, "CPU")
    print(f"      Decoder compiled successfully in {time.perf_counter() - t_comp_start:.4f}s!")

    text_ov, n_tokens, ttft, gen_time = generate_loop_pure_ov(
        processor, embeds_ov, per_layer_ov, llm_ids_ov, core, ov_decoder, MAX_LEN, ov_models,
        config_dict, per_layer_weights, device=device, max_new_tokens=max_new_tokens
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
    print(f"Launching Gemma-4 Pure OpenVINO High-Performance Multimodal NPU Inference Pipeline")
    print(f"Target precision: {args.precision.upper()}")
    print(f"Target compilation device: {args.device}")
    print(f"Model directory: {OV_DIR}")
    print("="*80)

    config_path = OV_DIR / "config.json"
    if not config_path.exists():
        print(f"Cannot find configuration file {config_path}! Please run conversion first: python pytorch_to_ov_unified_nncf.py --precision {args.precision}")
        return

    # 1. Read configuration file to load model parameters
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    print("Successfully loaded config.json")

    print("\nInitializing Preprocessor / Tokenizer...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 2. Setup the raw XML file paths to compile with static shapes later
    ov_models = {}
    if (OV_DIR / "audio_encoder.xml").exists():
        ov_models['audio'] = OV_DIR / "audio_encoder.xml"
    if (OV_DIR / "embed_tokens.xml").exists():
        ov_models['embed'] = OV_DIR / "embed_tokens.xml"
    if (OV_DIR / "vision_encoder.xml").exists():
        ov_models['vision'] = OV_DIR / "vision_encoder.xml"
    if (OV_DIR / "decoder_embeds.xml").exists():
        ov_models['decoder'] = OV_DIR / "decoder_embeds.xml"

    core = ov.Core()

    # Apply NPU / compiler specific cache config if applicable
    if args.device == "NPU":
        core.set_property("NPU", {"CACHE_DIR": "./log/npu_cache"})
        print("NPU cache enabled at ./log/npu_cache")

    # 3. Load Per-Layer Embedding weight matrix via Memory Mapping (mmap)
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

    ast_msgs = [{"role": "user", "content": [{"type": "audio", "audio": "test_audio.wav"}, {"type": "text", "text": "請將這段語音翻譯成繁體中文："}]}]
    ast_inputs = processor.apply_chat_template(ast_msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)

    image_msgs = [{"role": "user", "content": [{"type": "image", "url": "test_image.jpeg"}, {"type": "text", "text": "Describe."}]}]
    image_inputs = processor.apply_chat_template(image_msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)

    run_pure_ov_inference("Audio Recognition Test (ASR - Transcribe)", asr_inputs, processor, core, ov_models, config_dict, per_layer_weights, device=args.device, max_new_tokens=40)
    run_pure_ov_inference("Audio Translation Test (AST - Translate to Chinese)", ast_inputs, processor, core, ov_models, config_dict, per_layer_weights, device=args.device, max_new_tokens=60)
    run_pure_ov_inference("Image Inference Test (Image Describe)", image_inputs, processor, core, ov_models, config_dict, per_layer_weights, device=args.device, max_new_tokens=60)

if __name__ == "__main__":
    main()
