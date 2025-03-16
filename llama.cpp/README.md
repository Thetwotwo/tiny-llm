# llama.cpp

## 1.简介

Tiny LLM 92M 模型已支持 llama.cpp C++ 推理框架，建议在 linux 环境下测试，windows效果不好；

### 1.1 llama.cpp

llama.cpp 是一个C++库，用于简化LLM推理的设置。它使得在本地机器上运行Qwen成为可能。该库是一个纯C/C++实现，不依赖任何外部库，并且针对x86架构提供了AVX、AVX2和AVX512加速支持。此外，它还提供了2、3、4、5、6以及8位量化功能，以加快推理速度并减少内存占用。对于大于总VRAM容量的大规模模型，该库还支持CPU+GPU混合推理模式进行部分加速。本质上，llama.cpp的用途在于运行GGUF（由GPT生成的统一格式）模型。

### 1.2 gguf

GGUF是指一系列经过特定优化，能够在不同硬件上高效运行的大模型格式。这些模型格式包括但不限于原始格式、exl2、finetuned模型（如axolotl、unsloth等）。每种格式都有其特定的应用场景和优化目标，例如加速模型推理、减少模型大小、提高模型准确性等。


## 2.使用

### 2.1 准备

编译纯CPU版本：

```shell
cmake -B build_cpu
cmake --build build_cpu --config Release -j 24
```

编译CUDA版本：
```shell
cmake -B build_cuda -DLLAMA_CUDA=ON
cmake --build build_cpu --config Release -j 24
```

### 2.2 模型转化

先需要按照如下所示的方式为fp16模型创建一个GGUF文件：

```shell
python convert-hf-to-gguf.py {path}/tiny_llm_sft_92m --outfile {path}/tinyllm-92m-fp16.gguf
```

其中，第一个参数指代的是预训练模型所在的路径或者HF模型的名称，第二个参数则指的是想要生成的GGUF文件的路径；在运行命令之前，需要先创建这个目录。

下面需要根据实际需求将其量化至低比特位。以下是一个将模型量化至4位的具体示例：

```shell
./llama-quantize models/tinyllm/tinyllm-92m-fp16.gguf  models/tiny_llm_92m/tinyllm-92m-q4_0.gguf q4_0
```

到现在为止，已经完成了将模型量化为4比特，并将其放入GGUF文件中。这里的 q4_0 表示4比特量化。现在，这个量化后的模型可以直接通过llama.cpp运行。

### 2.3 推理

使用如下命令可以运行模型

```shell
./llama-cli -m ./models/tinyllm/tinyllm-92m-fp16.gguf -p "<|system|>\n你是人工智能个人助手。\n<|user|>\n请介绍一下北京，你好。\n<|assistant|>\n" -n 128 --repeat-penalty 1.2 --top-p 0.8 --top-k 0
```

`-n` 指的是要生成的最大token数量。这里还有其他超参数供你选择，并且你可以运行

执行如下命令，了解更多功能：
```shell
./llama-cli -h
```




# GGUF介绍

# **GGUF（GGML Unified Format）介绍**
## **1. 什么是 GGUF？**
**GGUF（GGML Unified Format）** 是 GGML（一个轻量级高效的张量计算库）引入的一种 **模型文件格式**，用于存储 **优化后的 LLM（大语言模型）权重**，以便高效推理。  
它是 GGML 继 **GGML（.ggml）和 GGJT（.ggjt）格式** 后的最新模型存储格式。

---

## **2. GGUF 的特点**
### **✅ 统一格式（Unified Format）**
- 取代 GGML 早期的 `.ggml` 和 `.ggjt` 格式，提供更灵活和兼容的存储方式。

### **✅ 支持多种 LLM（大语言模型）**
- 适用于 **LLaMA、Mistral、GPT-2/3、Gemma** 等 Transformer 模型。
- 兼容不同模型结构（标准 Transformer、Mixture of Experts 等）。

### **✅ 高效存储**
- 采用 **二进制格式**，支持 **FP16、Q4、Q5、Q8 量化**，减少模型大小，提高推理速度。

### **✅ 包含元数据（Metadata）**
- 存储 **超参数、架构信息、量化方式、词表等**，方便模型加载和解析。换句话说，多种信息打包到为一个同意的文件。

### **✅ 跨平台支持**
- 兼容 **CPU、GPU（CUDA/Metal）、ARM** 等硬件，适用于移动端和边缘计算。

---

## **3. GGUF 文件结构**
一个 GGUF 文件包含：
- **Header（文件头）**：存储格式版本、元数据等信息。
- **Tensor Data（模型权重）**：以二进制格式存储的张量数据，支持量化。
- **Metadata（额外信息）**：存储超参数、模型配置等。

---

## **4. GGUF vs. 其他格式**
| 格式  | 主要用途  | 兼容性 | 量化支持 | 适用场景 |
|-------|---------|--------|---------|---------|
| **GGUF** | LLM 推理 | GGML / llama.cpp | ✅ 是 | 轻量推理 |
| GGML (.ggml) | 早期 GGML 格式 | 旧版本 GGML | ✅ 是 | 过时 |
| GGJT (.ggjt) | 过渡格式 | 旧版 GGML | ✅ 是 | 过时 |
| PyTorch (.pt) | 训练 & 推理 | PyTorch | ❌ 否 | 研究 & 训练 |
| ONNX (.onnx) | 兼容多框架 | ONNX 兼容框架 | ❌ 否 | 训练 & 推理 |
| TensorFlow (.pb) | TensorFlow 推理 | TensorFlow | ❌ 否 | 训练 & 推理 |

---


