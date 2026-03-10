"""Step A1: Download and audit SciKnowEval dataset."""
import os
import pandas as pd
from datasets import load_dataset

PROJ = "/mnt/d/Projects/Domain-Aware-Prompt-Engineering-Unifying-Scientific-Benchmarks-with-Context-Robust-RAG-Evaluation"
RAW_DIR = os.path.join(PROJ, "data", "sciknoweval", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

print("=" * 60)
print("A1: DOWNLOADING SciKnowEval")
print("=" * 60)

ds = load_dataset("hicai-zju/SciKnowEval", split="test")
print("Dataset ID: hicai-zju/SciKnowEval")
print("Subsets/configs: 1 (default)")
print("Split downloaded: test")
print(f"Total records: {len(ds)}")

# Save raw
raw_path = os.path.join(RAW_DIR, "sciknoweval.parquet")
ds.to_parquet(raw_path)
print("Saved raw to: data/sciknoweval/raw/sciknoweval.parquet")

# Convert to DataFrame
df = ds.to_pandas()

# Extract nested fields
df["level"] = df["details"].apply(lambda x: x.get("level", "") if isinstance(x, dict) else "")
df["task_field"] = df["details"].apply(lambda x: x.get("task", "") if isinstance(x, dict) else "")
df["subtask"] = df["details"].apply(lambda x: x.get("subtask", "") if isinstance(x, dict) else "")
df["source_field"] = df["details"].apply(lambda x: x.get("source", "") if isinstance(x, dict) else "")
df["prompt_default"] = df["prompt"].apply(lambda x: x.get("default", "") if isinstance(x, dict) else str(x))

print()
print("=" * 60)
print("COLUMN ANALYSIS")
print("=" * 60)

columns_info = [
    ("question", df["question"]),
    ("answer", df["answer"]),
    ("type", df["type"]),
    ("domain", df["domain"]),
    ("answerKey", df["answerKey"]),
    ("prompt.default", df["prompt_default"]),
    ("details.level", df["level"]),
    ("details.task", df["task_field"]),
    ("details.subtask", df["subtask"]),
    ("details.source", df["source_field"]),
]

for name, col in columns_info:
    nulls = col.isna().sum() + (col == "").sum()
    uniq = col.nunique()
    non_empty = col[col != ""].dropna()
    examples = non_empty.head(3).tolist()
    print(f"\n  {name}:")
    print(f"    dtype={col.dtype}, unique={uniq}, nulls_or_empty={nulls}")
    print(f"    examples: {examples}")

print()
print("=" * 60)
print("STRATIFICATION AXIS: domain (value_counts)")
print("=" * 60)
print(df["domain"].value_counts().sort_values(ascending=False).to_string())

print()
print("=" * 60)
print("STRATIFICATION AXIS: details.level (value_counts)")
print("=" * 60)
print(df["level"].value_counts().sort_values(ascending=False).to_string())

print()
print("=" * 60)
print("TYPE distribution")
print("=" * 60)
print(df["type"].value_counts().sort_values(ascending=False).to_string())

print()
print("=" * 60)
print("TEXT LENGTH DISTRIBUTION (question field)")
print("=" * 60)
lengths = df["question"].str.len()
print(f"  min:    {lengths.min()}")
print(f"  max:    {lengths.max()}")
print(f"  mean:   {lengths.mean():.1f}")
print(f"  median: {lengths.median():.1f}")
print(f"  std:    {lengths.std():.1f}")

print()
print("=" * 60)
print("5 SAMPLE ROWS FROM DIFFERENT STRATIFICATION GROUPS")
print("=" * 60)

seen = set()
samples = []
for domain in ["Biology", "Chemistry", "Material", "Physics"]:
    for lvl in ["L1", "L3", "L5", "L2", "L4"]:
        key = f"{domain}_{lvl}"
        if key not in seen:
            subset = df[(df["domain"] == domain) & (df["level"] == lvl)]
            if len(subset) > 0:
                samples.append(subset.iloc[0])
                seen.add(key)
                break
    if len(samples) >= 5:
        break

# Add 5th sample if needed
if len(samples) < 5:
    for lvl in ["L5", "L4", "L3"]:
        subset = df[(df["domain"] == "Physics") & (df["level"] == lvl)]
        if len(subset) > 0:
            samples.append(subset.iloc[0])
            break

for i, row in enumerate(samples):
    print(f"\n--- Sample {i+1}: Domain={row['domain']}, Level={row['level']} ---")
    q = row["question"]
    if len(q) > 300:
        q = q[:300] + "..."
    print(f"  question: {q}")
    a = str(row["answer"]) if row["answer"] else "[empty]"
    if len(a) > 200:
        a = a[:200] + "..."
    print(f"  answer: {a}")
    print(f"  type: {row['type']}")
    ak = row["answerKey"] if row["answerKey"] else "[empty]"
    print(f"  answerKey: {ak}")
    print(f"  task: {row['task_field']}")
    print(f"  subtask: {row['subtask']}")
    print(f"  source: {row['source_field']}")
    # Access choices from original dataset to avoid pandas conversion issues
    idx = row.name
    orig_row = ds[idx]
    choices = orig_row["choices"]
    if choices and choices.get("text"):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        print("  choices:")
        for lbl, txt in zip(labels, texts):
            if len(txt) > 150:
                txt = txt[:150] + "..."
            print(f"    {lbl}: {txt}")
    else:
        print("  choices: [none]")

print()
print("=" * 60)
print("CROSS-TABULATION: domain x level")
print("=" * 60)
ct = pd.crosstab(df["domain"], df["level"], margins=True)
print(ct.to_string())

print()
print("=" * 60)
print("CROSS-TAB: domain x type")
print("=" * 60)
ct2 = pd.crosstab(df["domain"], df["type"], margins=True)
print(ct2.to_string())
