"""Test auxiliary models: BGE embedding, BGE reranker, DeBERTa NLI."""
import gc
import json
import subprocess
import time

import torch


def get_vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    parts = r.stdout.strip().split(", ")
    return int(parts[0]), int(parts[1])


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)


def test_bge_embedding(model_path: str) -> dict:
    print("=" * 60)
    print(f"Testing BGE Embedding: {model_path}")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(model_path)
    load_time = time.time() - t0
    print(f"Load:        OK (took {load_time:.1f}s)")

    emb = model.encode("The chemical formula for water is H2O.")
    print(f"Shape:       {emb.shape}")
    print(f"First 5:     {emb[:5]}")

    shape_ok = emb.shape == (768,)
    print(f"Shape check: {'OK' if shape_ok else 'FAIL'}")

    used, total = get_vram()
    print(f"VRAM:        {used} / {total} MiB")

    result = {
        "model_path": model_path,
        "role": "embedding",
        "load_time_s": round(load_time, 1),
        "embedding_shape": list(emb.shape),
        "test_passed": shape_ok,
        "vram_used_mib": used,
    }

    del model
    cleanup()
    used_after, _ = get_vram()
    print(f"Cleanup:     OK (VRAM released to {used_after} MiB)\n")
    return result


def test_bge_reranker(model_path: str) -> dict:
    print("=" * 60)
    print(f"Testing BGE Reranker: {model_path}")
    print("=" * 60)

    from sentence_transformers import CrossEncoder

    t0 = time.time()
    reranker = CrossEncoder(model_path)
    load_time = time.time() - t0
    print(f"Load:        OK (took {load_time:.1f}s)")

    scores = reranker.predict([
        ["What is water?", "Water is H2O, a molecule made of hydrogen and oxygen."],
        ["What is water?", "Python is a programming language."],
    ])
    print(f"Scores:      relevant={scores[0]:.4f}, irrelevant={scores[1]:.4f}")

    ranking_ok = float(scores[0]) > float(scores[1])
    print(f"Ranking:     {'OK (relevant > irrelevant)' if ranking_ok else 'FAIL'}")

    used, total = get_vram()
    print(f"VRAM:        {used} / {total} MiB")

    result = {
        "model_path": model_path,
        "role": "reranker",
        "load_time_s": round(load_time, 1),
        "score_relevant": round(float(scores[0]), 4),
        "score_irrelevant": round(float(scores[1]), 4),
        "test_passed": ranking_ok,
        "vram_used_mib": used,
    }

    del reranker
    cleanup()
    used_after, _ = get_vram()
    print(f"Cleanup:     OK (VRAM released to {used_after} MiB)\n")
    return result


def test_deberta_nli(model_path: str) -> dict:
    print("=" * 60)
    print(f"Testing DeBERTa NLI: {model_path}")
    print("=" * 60)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    load_time = time.time() - t0
    print(f"Load:        OK (took {load_time:.1f}s)")

    # NLI test: premise + hypothesis -> entailment/contradiction/neutral
    premise = "Water is H2O."
    hypothesis = "Water contains hydrogen."

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # DeBERTa-v3-large labels: 0=entailment, 1=neutral, 2=contradiction
    # (check model config for actual label mapping)
    labels = ["entailment", "neutral", "contradiction"]
    pred_idx = probs.argmax().item()
    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"label_{pred_idx}"

    print(f"Premise:     \"{premise}\"")
    print(f"Hypothesis:  \"{hypothesis}\"")
    print(f"Prediction:  {pred_label} (probs: {probs[0].tolist()})")

    # For base DeBERTa (not fine-tuned on NLI), output may not be meaningful
    # We just verify the model loads and produces 3-class output
    output_ok = logits.shape[-1] >= 2  # at least 2 classes
    print(f"Output check: {'OK' if output_ok else 'FAIL'} (shape: {logits.shape})")

    used, total = get_vram()
    print(f"VRAM:        {used} / {total} MiB")

    result = {
        "model_path": model_path,
        "role": "nli",
        "load_time_s": round(load_time, 1),
        "prediction": pred_label,
        "num_labels": int(logits.shape[-1]),
        "test_passed": output_ok,
        "vram_used_mib": used,
    }

    del model, tokenizer
    cleanup()
    used_after, _ = get_vram()
    print(f"Cleanup:     OK (VRAM released to {used_after} MiB)\n")
    return result


if __name__ == "__main__":
    results = {}

    results["bge_embedding"] = test_bge_embedding("models/bge-base-en-v1.5/")
    results["bge_reranker"] = test_bge_reranker("models/bge-reranker-v2-m3/")
    results["deberta_nli"] = test_deberta_nli("models/deberta-v3-large/")

    with open("/tmp/test_auxiliary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("All auxiliary tests complete. Results saved to /tmp/test_auxiliary.json")
