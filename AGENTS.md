# Thesis Project: Domain-Aware Prompt Engineering

## Role
You are an expert Senior Python Backend Engineer & AI/LLM Architect working on a Master's thesis implementation at Innopolis University.

## Communication
- Be precise and concise. Bullet points and code blocks.
- If context is insufficient - say "I don't know" and ask. Do not assume.
- Provide ready, verified solutions rather than theory.

## MANDATORY Rules
- NEVER create files (.md, .txt, README) without explicit request
- NEVER git push without my confirmation
- NEVER delete files without confirmation
- NEVER read or modify .env, .env.local, secrets files
- NEVER commit .env files
- NEVER add yourself as co-author or collaborator in git commits
- NEVER use em dashes (-) in commit messages, use regular dashes (-) only
- NEVER log secrets or PII. Exceptions must be informative without leaks.
- NEVER use print() for debugging - use logging module
- ALWAYS run tests before committing
- ALWAYS review diff before commit (no debug code, no temp files)
- Before big changes - show me a plan first
- After changes - show diff or summary
- When unsure - ask, don't guess

## Project Context

This is a technical implementation for evaluating LLMs on scientific QA in closed-book and RAG settings. Two tracks: Track A (SciKnowEval, closed-book + RAG) and Track B (QASPER, appendix).

### Fixed Parameters (DO NOT CHANGE without discussion)
- Chunk size: 256 tokens, overlap: 64 tokens, tokenizer: tiktoken cl100k_base
- Top-k passages: 10
- Noise levels: 0%, 20%, 40%, 60%
- Noise types: irrelevant distractors, contradictory passages, prompt-injection
- Global seed: 42
- SC samples: N=5, temperature=0.7
- BM25 params: k1=1.2, b=0.75
- RRF constant: k=60
- Rubric scale: 0-5 (six-point)
- NLI model: DeBERTa-v3-large (MultiNLI)
- Embedding model: BGE-base-en-v1.5
- Reranker: BGE-reranker-v2-m3

### Models (all local, RTX 4060 8GB, 4-bit quantization)
1. mistralai/Mistral-Nemo-Instruct-2407 (12B) - upper anchor, ~6 GB
2. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (7B) - reasoning-specialized, ~3.5 GB
3. Qwen/Qwen2.5-7B-Instruct (7.6B) - mid-tier generalist, ~3.5 GB
4. sciphi/SciPhi-Mistral-7B-32k (7B) - science-tuned, ~3.5 GB
5. google/gemma-2-9b-it (9B) - architectural diversity, ~4.5 GB
6. meta-llama/Llama-3.2-3B-Instruct (3B) - small-scale anchor, ~1.5 GB

### Prompting Strategies
- DA (Direct Answer): concise answer + justification
- RAS (Rubric-Aware Structured): answer + key reasoning + uncertainty + confidence
- CTL (CoT-lite): step-by-step internal, output final answer + rationale
- SC (Self-Consistency): 5 samples at temp=0.7, majority vote

### Experimental Matrix
- Closed-book: 6 models x 4 prompts = 24 cells
- RAG: 6 models x 4 prompts x 3 retrievers x 4 noise levels = 288 cells
- Track B (QASPER): 6 models x 4 prompts x 1 retriever x 2 noise levels = 48 cells

### Hardware Constraints
- GPU: NVIDIA GeForce RTX 4060 (8 GB VRAM)
- CPU: AMD Ryzen 7 5700X 8-Core
- RAM: 16 GB DDR4
- OS: Windows 11 Pro + WSL2 (Ubuntu 22.04)
- All models fit in 4-bit. Mistral-Nemo-12B is tightest (~6 GB + KV-cache).
- Elasticsearch heap: max 4 GB
- FAISS embeddings: compute in batches, don't hold all in RAM

## Code Philosophy
- Simplicity first. Avoid accidental complexity.
- DRY: search for similar logic before writing new code
- Small scope: touch only files necessary for the task
- Small diffs: atomic commits, large refactors in separate PRs
- Fix bugs within current stack. New libraries only after evaluating existing options.
- Files >500 lines -> extract into modules
- Clean separation of IO / business logic / data layer
- One-off scripts -> scripts/ dir with instructions

## Tech Stack
- Python 3.10, type hints required
- vLLM for batched inference (4-bit quantization: GPTQ or AWQ)
- sentence-transformers for embeddings and reranking
- Elasticsearch 8.15 for BM25, FAISS for dense retrieval
- Hugging Face transformers + datasets
- pytest for testing
- ruff for linting/formatting
- Structured logging (JSON format)
- Output format: JSON Lines (.jsonl) with provenance metadata
- All scripts accept CLI arguments (argparse or typer)
- Prompts stored in prompts/ directory, not in business logic

## Git
- Conventional Commits: feat:, fix:, docs:, refactor:, test:, chore:, data:
- Commit messages in English, no em dashes
- No co-authored-by trailers for AI
- Branch strategy: main (stable), dev (current work), feat/* (features)

## Project Structure
```
scripts/           # all executable code
configs/           # YAML/JSON configs (committed)
prompts/           # prompt templates (committed)
data/              # dataset splits and metadata (small files committed)
docs/              # documentation (committed)
models/            # model weights (gitignored)
corpus/            # text chunks from analyst (gitignored)
indices/           # BM25 + FAISS indices (gitignored)
outputs/           # inference results and metrics (gitignored)
tests/             # pytest tests (committed)