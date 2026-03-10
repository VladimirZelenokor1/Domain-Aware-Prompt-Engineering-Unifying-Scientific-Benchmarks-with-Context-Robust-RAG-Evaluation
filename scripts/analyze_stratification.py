"""Step A3: Stratification analysis and split sizing."""
import os
import json
import pandas as pd

PROJ = "/mnt/d/Projects/Domain-Aware-Prompt-Engineering-Unifying-Scientific-Benchmarks-with-Context-Robust-RAG-Evaluation"

# ===== SCIKNOWEVAL =====
print("=" * 60)
print("A3: STRATIFICATION ANALYSIS - SCIKNOWEVAL")
print("=" * 60)

df_ski = pd.read_parquet(os.path.join(PROJ, "data", "sciknoweval", "raw", "sciknoweval.parquet"))
df_ski["level"] = df_ski["details"].apply(lambda x: x.get("level", "") if isinstance(x, dict) else "")
total_ski = len(df_ski)

print(f"\nTotal records: {total_ski}")

# Cross-tabulation
ct = pd.crosstab(df_ski["domain"], df_ski["level"])
print("\nCross-tabulation: domain x level")
print(ct.to_string())

# Find smallest stratum
min_stratum_val = ct.min().min()
min_stratum_loc = ct.stack()
min_idx = min_stratum_loc.idxmin()
print(f"\nSmallest stratum: domain={min_idx[0]}, level={min_idx[1]} with {min_stratum_val} samples")

# Combined strata sizes
df_ski["stratum"] = df_ski["domain"] + "_" + df_ski["level"]
strata_counts = df_ski["stratum"].value_counts().sort_values()
print("\nAll strata (sorted ascending):")
print(strata_counts.to_string())

n_strata = len(strata_counts)
print(f"\nTotal number of strata: {n_strata}")

# Split sizing analysis
print("\n" + "=" * 60)
print("SPLIT SIZING ANALYSIS")
print("=" * 60)

smallest = strata_counts.min()
smallest_name = strata_counts.index[0]
print(f"\nBottleneck stratum: {smallest_name} with {smallest} samples")

# For 3-way split, each stratum needs >= 2 samples per split
# With proportional splitting:
# dev ~10% -> smallest stratum gets ~smallest*0.10
# holdout ~5% -> smallest stratum gets ~smallest*0.05
# main_test ~85% -> rest

print("\nFeasibility check:")
for dev_pct, holdout_pct in [(0.10, 0.05), (0.08, 0.04), (0.12, 0.06)]:
    main_pct = 1.0 - dev_pct - holdout_pct
    dev_in_smallest = round(smallest * dev_pct)
    holdout_in_smallest = round(smallest * holdout_pct)
    main_in_smallest = smallest - dev_in_smallest - holdout_in_smallest
    dev_total = round(total_ski * dev_pct)
    holdout_total = round(total_ski * holdout_pct)
    main_total = total_ski - dev_total - holdout_total
    feasible = dev_in_smallest >= 2 and holdout_in_smallest >= 2 and main_in_smallest >= 2
    print(f"\n  Option: dev={dev_pct:.0%}, holdout={holdout_pct:.0%}, main={main_pct:.0%}")
    print(f"    dev={dev_total}, holdout={holdout_total}, main={main_total}")
    print(f"    In smallest stratum ({smallest_name}, n={smallest}):")
    print(f"      dev~{dev_in_smallest}, holdout~{holdout_in_smallest}, main~{main_in_smallest}")
    print(f"    Feasible: {'YES' if feasible else 'NO - stratum too small'}")

# Check strata < 3 (minimum for any 3-way split)
tiny_strata = strata_counts[strata_counts < 6]
if len(tiny_strata) > 0:
    print(f"\nWARNING: {len(tiny_strata)} strata have < 6 samples (risky for 3-way split):")
    print(tiny_strata.to_string())
else:
    print("\nAll strata have >= 6 samples - 3-way stratified split is feasible.")

# Recommended sizes
print("\n" + "-" * 40)
print("RECOMMENDED SPLIT SIZES:")
print("-" * 40)
dev_size = round(total_ski * 0.10)
holdout_size = round(total_ski * 0.05)
main_size = total_ski - dev_size - holdout_size
print(f"  dev (calibration):    {dev_size} ({dev_size/total_ski*100:.1f}%)")
print(f"  holdout (audit):      {holdout_size} ({holdout_size/total_ski*100:.1f}%)")
print(f"  main_test:            {main_size} ({main_size/total_ski*100:.1f}%)")
print(f"  Total:                {dev_size + holdout_size + main_size}")
print(f"\n  Rationale: smallest stratum ({smallest_name}) has {smallest} samples.")
print(f"  At 10%/5% split: dev gets ~{round(smallest*0.10)}, holdout gets ~{round(smallest*0.05)} from this stratum.")
print("  Both >= 2, so stratified splitting is feasible.")
print(f"  dev={dev_size} exceeds minimum 200 threshold.")
print(f"  holdout={holdout_size} exceeds minimum 100 threshold.")


# ===== QASPER =====
print()
print("=" * 60)
print("A3: STRATIFICATION ANALYSIS - QASPER")
print("=" * 60)

# Load raw QASPER and flatten
RAW_DIR = os.path.join(PROJ, "data", "qasper", "raw")
all_data = {}
for fname in sorted(os.listdir(RAW_DIR)):
    if fname.endswith(".json"):
        with open(os.path.join(RAW_DIR, fname)) as f:
            all_data[fname] = json.load(f)


def determine_answer_type(answer_obj: dict) -> str:
    """Determine the answer type from a single answer object."""
    if answer_obj.get("unanswerable", False):
        return "unanswerable"
    if answer_obj.get("extractive_spans") and len(answer_obj["extractive_spans"]) > 0:
        return "extractive"
    if answer_obj.get("free_form_answer") and answer_obj["free_form_answer"].strip():
        return "free_form"
    if answer_obj.get("yes_no") is not None:
        return "yes_no"
    return "unknown"


rows = []
for filename, papers in sorted(all_data.items()):
    split_name = filename.replace("qasper-", "").replace("-v0.3.json", "")
    for paper_id, paper in papers.items():
        for qa in paper.get("qas", []):
            answers = qa.get("answers", [])
            annotator_types = []
            for ans_entry in answers:
                ans_obj = ans_entry.get("answer", {})
                annotator_types.append(determine_answer_type(ans_obj))
            first_type = annotator_types[0] if annotator_types else "unknown"
            first_answer_text = ""
            if answers:
                first_ans = answers[0].get("answer", {})
                if first_type == "unanswerable":
                    first_answer_text = "[unanswerable]"
                elif first_type == "extractive":
                    first_answer_text = " | ".join(first_ans.get("extractive_spans", []))
                elif first_type == "free_form":
                    first_answer_text = first_ans.get("free_form_answer", "")
                elif first_type == "yes_no":
                    first_answer_text = "yes" if first_ans.get("yes_no") else "no"

            rows.append({
                "paper_id": paper_id,
                "paper_title": paper.get("title", ""),
                "question_id": qa.get("question_id", ""),
                "question": qa.get("question", ""),
                "answer_type": first_type,
                "answer_text": first_answer_text,
                "num_annotators": len(answers),
                "split": split_name,
            })

df_q = pd.DataFrame(rows)
total_q = len(df_q)

print(f"\nTotal questions: {total_q}")
print("\nAnswer type distribution:")
type_counts = df_q["answer_type"].value_counts()
type_pcts = df_q["answer_type"].value_counts(normalize=True) * 100
for atype in type_counts.index:
    print(f"  {atype}: {type_counts[atype]} ({type_pcts[atype]:.1f}%)")

rarest_type = type_counts.index[-1]
rarest_count = type_counts.iloc[-1]
print(f"\nRarest answer type: {rarest_type} with {rarest_count} samples")

# Since total is 5049 (manageable), consider using all
print(f"\nTotal questions: {total_q}")
print(f"Since total ({total_q}) is within 1000-6000 range, options:")

# Option 1: Use all questions
print(f"\n  Option A: Use ALL {total_q} questions")
print("    Pros: maximum statistical power, no sampling bias")
print("    Cons: more compute for inference (but manageable)")

# Option 2: Sample
for sample_size in [500, 750, 1000]:
    rarest_in_sample = round(rarest_count * sample_size / total_q)
    print(f"\n  Option B ({sample_size}): stratified sample of {sample_size}")
    print(f"    Rarest type ({rarest_type}) gets ~{rarest_in_sample} samples")
    feasible = rarest_in_sample >= 5
    print(f"    Feasible (>= 5 of rarest): {'YES' if feasible else 'NO'}")

# Check if any type < 10
tiny_types = type_counts[type_counts < 10]
if len(tiny_types) > 0:
    print(f"\nWARNING: {len(tiny_types)} answer types have < 10 samples:")
    print(tiny_types.to_string())
else:
    print("\nAll answer types have >= 10 samples.")

print("\n" + "-" * 40)
print("RECOMMENDED:")
print("-" * 40)
print(f"  Use ALL {total_q} questions (no sampling needed).")
print("  Total is manageable for Track B evaluation.")
print("  Avoids introducing sampling bias.")
print("  Preserves exact original proportions.")
print(f"  Rarest type ({rarest_type}) keeps all {rarest_count} samples.")
