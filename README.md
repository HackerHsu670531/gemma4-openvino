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
python -m venv openvino_gemma4_env
openvino_gemma4_env\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
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

Launch the pure OpenVINO inference pipeline on any test configuration. You can specify the target device (NPU, CPU, or GPU):

```bash
python infer_openvino_audio_vision_asr_ast_nncf_final.py --precision int4 --device NPU
```

---

## 📈 Benchmark Performance Diagnostics

Below is a benchmark snapshot captured from a real system diagnostic running Gemma-4 on an Intel core platform (compiled with OpenVINO version `2026.3.0.dev20260525` running under INT4 precision):

**[Device: NPU]**
| Modality / Performance Metric | Preprocessing feature extraction | Time to First Token (TTFT) | Total Generation Time | Generated Tokens | Average Speeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🎵 Audio Recognition (ASR)** | `146.22s` | `25.78s` | `48.64s` | 31 tokens | `0.64 tokens/s` |
| **🌍 Audio Translation (AST)** | `30.68s` | `16.26s` | `88.71s` | 60 tokens | `0.68 tokens/s` |
| **🖼️ Image Inference (Vision)** | `259.71s` | `16.16s` | `66.30s` | 60 tokens | `0.90 tokens/s` |

**[Device: CPU]**
| Modality / Performance Metric | Preprocessing feature extraction | Time to First Token (TTFT) | Total Generation Time | Generated Tokens | Average Speeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🎵 Audio Recognition (ASR)** | `6.37s` | `52.92s` | `96.74s` | 31 tokens | `0.32 tokens/s` |
| **🌍 Audio Translation (AST)** | `6.39s` | `2.81s` | `101.23s` | 60 tokens | `0.59 tokens/s` |
| **🖼️ Image Inference (Vision)** | `5.60s` | `2.83s` | `95.95s` | 60 tokens | `0.63 tokens/s` |

**[Device: GPU]**
| Modality / Performance Metric | Preprocessing feature extraction | Time to First Token (TTFT) | Total Generation Time | Generated Tokens | Average Speeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🎵 Audio Recognition (ASR)** | `15.99s` | `5.02s` | `17.19s` | 31 tokens | `1.80 tokens/s` |
| **🌍 Audio Translation (AST)** | `6.13s` | `0.71s` | `26.46s` | 60 tokens | `2.27 tokens/s` |
| **🖼️ Image Inference (Vision)** | `14.41s` | `5.00s` | `29.45s` | 60 tokens | `2.04 tokens/s` |

---

## 👩‍💻 Author & Contributions

* **Author:** Hacker Hsu ([HackerHsu670531](https://github.com/HackerHsu670531)) - Deep Learning Software Engineer
* **Enterprise / Hardware Org:** IOTG/VMC/VPU NPU Execution Architectures
```