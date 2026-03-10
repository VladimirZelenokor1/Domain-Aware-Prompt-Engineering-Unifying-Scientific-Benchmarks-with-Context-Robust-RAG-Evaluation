# Hardware and Software Environment

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| CPU | AMD Ryzen 7 5700X 8-Core Processor, 3401 MHz (16 logical processors) |
| RAM | 16 GB DDR4 |
| Motherboard | Gigabyte B550M K |
| OS (host) | Windows 11 Pro, Build 26200 |
| OS (work) | Ubuntu 22.04 via WSL2 |

## Software Stack

| Package | Version | Role |
|---------|---------|------|
| Python | 3.10 | Runtime |
| torch | 2.10.0+cu128 | Deep learning framework |
| CUDA | 12.8 | GPU acceleration |
| vLLM | 0.17.0 | Batched LLM inference |
| transformers | 4.57.6 | Model loading, tokenization |
| sentence-transformers | 5.2.3 | BGE embeddings, reranker |
| faiss-gpu | GPU count: 1 | Dense retrieval index |
| pyserini | installed | BM25 sparse retrieval |
| scikit-learn | 1.5.2 | Inter-rater reliability metrics |
| sacrebleu | 2.4.3 | BLEU-4 computation |
| rouge_score | installed | ROUGE-L computation |
| tiktoken | 0.12.0 | Tokenizer for chunking (cl100k_base) |
| datasets | installed | Hugging Face dataset loading |
| numpy | 1.26.4 | Numerical operations |

## Quantization

All models run in 4-bit quantization (GPTQ or AWQ) to fit within 8 GB VRAM.

| Model | Parameters | Est. VRAM (4-bit) |
|-------|-----------|-------------------|
| Mistral-Nemo-Instruct-2407 | 12B | ~6 GB |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~3.5 GB |
| Qwen2.5-7B-Instruct | 7.6B | ~3.5 GB |
| SciPhi-Mistral-7B-32k | 7B | ~3.5 GB |
| Gemma-2-9B-IT | 9B | ~4.5 GB |
| Llama-3.2-3B-Instruct | 3B | ~1.5 GB |

## Global Configuration

- Random seed: 42
- Max new tokens: 1024
- SC samples: N=5, temperature=0.7