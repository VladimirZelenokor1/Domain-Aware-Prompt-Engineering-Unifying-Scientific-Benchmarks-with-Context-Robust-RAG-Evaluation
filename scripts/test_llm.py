"""Test a single LLM with vLLM and report results."""
import argparse
import gc
import json
import subprocess
import time

import torch
from vllm import LLM, SamplingParams


def get_vram_used():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    parts = result.stdout.strip().split(", ")
    return int(parts[0]), int(parts[1])


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)


def test_model(model_path, gpu_util=0.85, max_len=2048, enforce_eager=False):
    print("=" * 60)
    print(f"Loading: {model_path}")
    print(f"  gpu_util={gpu_util}, max_len={max_len}, eager={enforce_eager}")
    print("=" * 60)

    t0 = time.time()
    llm = LLM(
        model=model_path,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=gpu_util,
        max_model_len=max_len,
        enforce_eager=enforce_eager,
    )
    load_time = time.time() - t0
    print(f"Load:        OK (took {load_time:.1f}s)")

    # Basic generation test
    params = SamplingParams(temperature=0.0, max_tokens=128)
    prompt = "What is the chemical formula for water? Answer in one sentence."
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text.strip()
    print(f"Generation:  \"{text[:200]}\"")

    h2o_found = "H2O" in text or "H2O" in text.upper()
    status = "OK" if h2o_found else "WARNING - H2O not found"
    print(f"H2O check:   {status}")

    used, total = get_vram_used()
    print(f"VRAM:        {used} / {total} MiB")

    result = {
        "model_path": model_path,
        "load_time_s": round(load_time, 1),
        "test_output": text[:200],
        "h2o_check": h2o_found,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "gpu_memory_utilization": gpu_util,
    }

    del llm
    cleanup()

    used_after, _ = get_vram_used()
    print(f"Cleanup:     OK (VRAM released to {used_after} MiB)")
    result["vram_after_cleanup_mib"] = used_after

    return result


def test_long_context(model_path, gpu_util=0.85):
    """Additional long-context test for Mistral-Nemo."""
    print("\n--- Long-context test (10 passages, ~2000 tokens) ---")

    passage = (
        "Water is a chemical substance with the formula H2O. It is a transparent, "
        "tasteless, odourless, and nearly colourless chemical substance. It is the "
        "main constituent of Earth's hydrosphere and the fluids of all known living "
        "organisms. It is vital for all known forms of life, even though it provides "
        "neither food, energy, nor organic micronutrients. Its chemical formula, H2O, "
        "indicates that each of its molecules contains one oxygen and two hydrogen atoms, "
        "connected by covalent bonds. The hydrogen atoms are attached to the oxygen atom "
        "at an angle of 104.45 degrees. Water is also a good solvent and is often referred "
        "to as the universal solvent. Substances that dissolve in water are called hydrophilic. "
        "Water molecules form hydrogen bonds with each other and are strongly polar. "
    )

    passages = []
    for i in range(10):
        passages.append(f"Passage {i+1}: {passage}")

    context = "\n\n".join(passages)
    prompt = (
        f"Based on the following passages, answer the question.\n\n"
        f"{context}\n\n"
        f"Question: What angle do the hydrogen atoms form with the oxygen atom in water?\n"
        f"Answer in one sentence."
    )

    llm = LLM(
        model=model_path,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=gpu_util,
        max_model_len=4096,
    )

    params = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text.strip()
    print(f"Long-ctx output: \"{text[:200]}\"")

    used, total = get_vram_used()
    print(f"Long-ctx VRAM:   {used} / {total} MiB")

    result = {
        "long_ctx_output": text[:200],
        "long_ctx_vram_mib": used,
    }

    del llm
    cleanup()
    used_after, _ = get_vram_used()
    print(f"Cleanup:         OK (VRAM released to {used_after} MiB)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    result = test_model(args.model_path, args.gpu_util, args.max_len, args.enforce_eager)

    if args.long_context:
        lc_result = test_long_context(args.model_path, args.gpu_util)
        result.update(lc_result)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print("\nDONE")
