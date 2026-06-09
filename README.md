# Gemma-4-E2B-it Pure OpenVINO High-Performance Multimodal Inference Pipeline

This repository contains a high-performance, native OpenVINO-based multimodal inference pipeline for the **Gemma-4 E2B** model (`google/gemma-4-E2B-it`). It supports **speech-to-text (ASR)**, **audio translation (AST)**, and **image describe (Vision-Language)** tasks using modular, fully quantized, and memory-optimized architectures.

---

## 🚀 Key Features

* **Pure OpenVINO Pipeline:** Fully optimized C++ & Python runtime execution without requiring active PyTorch weight holding during inference. Saves significant memory!
* **Memory Optimization (Map-Mode / PLE):** Leverage memory-mapped numpy loading (`mmap_mode="r"`) for massive Per-Layer Embedding weight matrices (~2.4 GB memory savings).
* **NNCF Compression Suite:** High-performance quantization using OpenVINO Neural Network Compression Framework (NNCF) to select standard precisions, including:
  * **FP16**
  * **INT8**
  * **INT4** (with custom sliding ratio protection on deep decoding cores to maintain model coherence).
* **Optimized Decoders:** Deeply integrated causal masks, customized sliding window support, and high-performance argmax projection directly within the OpenVINO-designed decoder.

---

## 📂 Repository Structure

```tree
N:\Hacker\Gemm4_E2B\
├── pytorch_to_ov_unified_nncf_final.py      # Module decomposition, weight translation, and NNCF quantization
├── infer_openvino_audio_vision_asr_ast_nncf_final.py  # High-performance, pure-OpenVINO execution pipeline
├── requirements.txt                         # Full snapshot of Python dependencies
└── README.md                                # This reference guide
```

---

## 🛠️ Step-by-Step Guide

### 1. Installation

Set up your virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Model Export & Quantization (`Step 1`)

Execute the native PyTorch decomposition to export Embed, Audio, Vision, and Decoder weights to OpenVINO XML models. You can specify the desired precision using `--precision`:

```bash
# Export using INT8 precision (Recommended for balanced performance & latency)
python pytorch_to_ov_unified_nncf_final.py --precision int8 --force-recreate

# Export using high-precision FP16 mode
python pytorch_to_ov_unified_nncf_final.py --precision fp16 --force-recreate

# Export using highly compact INT4 mode
python pytorch_to_ov_unified_nncf_final.py --precision int4 --force-recreate
```

*Note: The script outputs the unified schema to `./gemma4-openvino-<precision>/` along with its mapped config file.*

### 3. Inference Run & Diagnostics (`Step 2`)

Launch the pure OpenVINO inference pipeline on any test configuration:

```bash
python infer_openvino_audio_vision_asr_ast_nncf_final.py --precision int8
```

---

## 📈 Benchmark Performance Diagnostics

Below is a benchmark snapshot captured from a real system diagnostic running Gemma-4 on an Intel core platform (compiled with OpenVINO version `2026.3.0.dev20260525` running under INT8 precision):

| Modality / Performance Metric | Preprocessing feature extraction | Time to First Token (TTFT) | Total Generation Time | Generated Tokens | Average Speeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🎵 Audio Recognition (ASR)** | `6.21s` | `74.81s` | `120.30s` | 30 tokens | `0.25 tokens/s` |
| **🌍 Audio Translation (AST)** | `0.27s` | `1.25s` | `40.92s` | 30 tokens | `0.73 tokens/s` |
| **🖼️ Image Inference (Vision)** | `4.31s` | `1.23s` | `85.64s` | 60 tokens | `0.70 tokens/s` |

---

## 👩‍💻 Author & Contributions

* **Author:** Hacker Hsu ([HackerHsu670531](https://github.com/HackerHsu670531)) - Deep Learning Software Engineer
* **Enterprise / Hardware Org:** IOTG/VMC/VPU NPU Execution Architectures
