"""Fallback test: load AWQ model via transformers when vLLM OOMs."""
import argparse
import gc
import json
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    parts = r.stdout.strip().split(", ")
    return int(parts[0]), int(parts[1])


def test(model_path: str, output_json: str | None = None) -> dict:
    print("=" * 60)
    print(f"Loading (transformers): {model_path}")
    print("=" * 60)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    load_time = time.time() - t0
    print(f"Load:        OK (took {load_time:.1f}s)")

    used, total = get_vram()
    print(f"VRAM (load): {used} / {total} MiB")

    prompt = "What is the chemical formula for water? Answer in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=1.0, do_sample=False)

    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"Generation:  \"{text[:200]}\"")

    h2o = "H2O" in text.upper() or "H2O" in text or "\u2082" in text
    print(f"H2O check:   {'OK' if h2o else 'WARNING'}")

    used_gen, _ = get_vram()
    print(f"VRAM (gen):  {used_gen} / {total} MiB")

    result = {
        "model_path": model_path,
        "backend": "transformers",
        "load_time_s": round(load_time, 1),
        "test_output": text[:200],
        "h2o_check": h2o,
        "vram_used_mib": used_gen,
        "vram_total_mib": total,
    }

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    used_after, _ = get_vram()
    print(f"Cleanup:     OK (VRAM released to {used_after} MiB)")
    result["vram_after_cleanup_mib"] = used_after

    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_json}")

    print("DONE")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--output-json", default=None)
    test(parser.parse_args().model_path, parser.parse_args().output_json)
