"""Test Gemma-2-9B-AWQ with awq_marlin quantization."""
import gc
import json
import subprocess
import time

import torch
from vllm import LLM, SamplingParams


def get_vram() -> tuple[int, int]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    parts = r.stdout.strip().split(", ")
    return int(parts[0]), int(parts[1])


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gemma-2-9B-AWQ with awq_marlin")
    print("=" * 60)

    t0 = time.time()
    llm = LLM(
        model="models/gemma-2-9b-awq/",
        quantization="awq_marlin",
        gpu_memory_utilization=0.85,
        max_model_len=512,
        enforce_eager=True,
    )
    load_time = time.time() - t0
    print(f"Load: OK ({load_time:.1f}s)")

    params = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = llm.generate(
        ["What is the chemical formula for water? Answer in one sentence."],
        params,
    )
    text = outputs[0].outputs[0].text.strip()
    print(f'Generation: "{text[:200]}"')

    h2o = "H2O" in text.upper() or "H2O" in text
    status = "OK" if h2o else "WARNING"
    print(f"H2O check: {status}")

    used, total = get_vram()
    print(f"VRAM: {used} / {total} MiB")

    result = {
        "model_path": "models/gemma-2-9b-awq/",
        "backend": "vllm",
        "quantization": "awq_marlin",
        "load_time_s": round(load_time, 1),
        "test_output": text[:200],
        "h2o_check": h2o,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 512,
        "enforce_eager": True,
    }

    with open("/tmp/test_gemma.json", "w") as f:
        json.dump(result, f, indent=2)

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    used_after, _ = get_vram()
    print(f"Cleanup: VRAM released to {used_after} MiB")
    print("DONE")
