"""Step B1: Split SciKnowEval into dev / main_test / holdout."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJ = "/mnt/d/Projects/Domain-Aware-Prompt-Engineering-Unifying-Scientific-Benchmarks-with-Context-Robust-RAG-Evaluation"
RAW_PATH = os.path.join(PROJ, "data", "sciknoweval", "raw", "sciknoweval.parquet")
OUT_DIR = os.path.join(PROJ, "data", "sciknoweval")
SEED = 42

# Confirmed sizes
DEV_SIZE = 2839
HOLDOUT_SIZE = 200
TOTAL = 28392
MAIN_SIZE = TOTAL - DEV_SIZE - HOLDOUT_SIZE  # 25353

df = pd.read_parquet(RAW_PATH)
assert len(df) == TOTAL, f"Expected {TOTAL}, got {len(df)}"

# Extract level and create combined stratification column
df["level"] = df["details"].apply(lambda x: x.get("level", "") if isinstance(x, dict) else "")
df["domain_level"] = df["domain"] + "_" + df["level"]

print(f"Total records: {len(df)}")
print(f"Strata: {df['domain_level'].nunique()}")

# --- Pass 1: separate (dev+holdout) from main_test ---
dev_holdout_size = DEV_SIZE + HOLDOUT_SIZE  # 3039
main_test, dev_holdout = train_test_split(
    df,
    test_size=dev_holdout_size,
    random_state=SEED,
    stratify=df["domain_level"],
)

# --- Pass 2: separate dev from holdout ---
dev, holdout = train_test_split(
    dev_holdout,
    test_size=HOLDOUT_SIZE,
    random_state=SEED,
    stratify=dev_holdout["domain_level"],
)

# --- Verification ---
print("\n===== VERIFICATION =====")

checks = [
    ("len(dev) == DEV_SIZE", len(dev) == DEV_SIZE),
    ("len(holdout) == HOLDOUT_SIZE", len(holdout) == HOLDOUT_SIZE),
    ("len(main_test) == MAIN_SIZE", len(main_test) == MAIN_SIZE),
    (
        "no overlap dev & holdout",
        len(set(dev.index) & set(holdout.index)) == 0,
    ),
    (
        "no overlap dev & main_test",
        len(set(dev.index) & set(main_test.index)) == 0,
    ),
    (
        "no overlap holdout & main_test",
        len(set(holdout.index) & set(main_test.index)) == 0,
    ),
    (
        "sum == total",
        len(dev) + len(holdout) + len(main_test) == TOTAL,
    ),
]

all_pass = True
for desc, result in checks:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  {'✓' if result else '✗'} {status}: {desc}")

if not all_pass:
    raise RuntimeError("Verification failed!")

# --- Save to JSON ---
# Drop helper columns before saving
drop_cols = ["level", "domain_level"]


def to_native(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    return obj


def save_split(split_df: pd.DataFrame, name: str) -> None:
    """Save a split DataFrame as JSON."""
    out = split_df.drop(columns=drop_cols)
    records = [to_native(row) for row in out.to_dict(orient="records")]
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {path} ({len(records)} records)")


save_split(dev, "dev")
save_split(main_test, "main_test")
save_split(holdout, "holdout")

# --- Distribution tables ---
print("\n===== DISTRIBUTION TABLES =====")

for name, split_df in [("dev", dev), ("main_test", main_test), ("holdout", holdout), ("TOTAL", df)]:
    ct = pd.crosstab(split_df["domain"], split_df["level"], margins=True)
    print(f"\n--- {name} ({len(split_df)} records) ---")
    print(ct.to_string())
