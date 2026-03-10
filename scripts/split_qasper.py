"""Step B2: Stratified sample of QASPER questions."""
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

PROJ = "/mnt/d/Projects/Domain-Aware-Prompt-Engineering-Unifying-Scientific-Benchmarks-with-Context-Robust-RAG-Evaluation"
RAW_DIR = os.path.join(PROJ, "data", "qasper", "raw")
OUT_DIR = os.path.join(PROJ, "data", "qasper")
SEED = 42
SAMPLE_SIZE = 500
TOTAL_EXPECTED = 5049


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


# Load and flatten
all_data = {}
for fname in sorted(os.listdir(RAW_DIR)):
    if fname.endswith(".json"):
        with open(os.path.join(RAW_DIR, fname)) as f:
            all_data[fname] = json.load(f)

rows = []
for filename, papers in sorted(all_data.items()):
    split_name = filename.replace("qasper-", "").replace("-v0.3.json", "")
    for paper_id, paper in papers.items():
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        full_text = paper.get("full_text", [])
        for qa in paper.get("qas", []):
            question = qa.get("question", "")
            question_id = qa.get("question_id", "")
            answers = qa.get("answers", [])

            annotator_types = []
            for ans_entry in answers:
                ans_obj = ans_entry.get("answer", {})
                annotator_types.append(determine_answer_type(ans_obj))

            first_type = annotator_types[0] if annotator_types else "unknown"

            # First annotator's answer text
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

            # Collect evidence from first annotator
            evidence = []
            if answers:
                evidence = answers[0].get("answer", {}).get("evidence", [])

            rows.append({
                "paper_id": paper_id,
                "paper_title": title,
                "question_id": question_id,
                "question": question,
                "answer_type": first_type,
                "answer_text": first_answer_text,
                "evidence": evidence,
                "num_annotators": len(answers),
                "split": split_name,
            })

df = pd.DataFrame(rows)
print(f"Total flattened questions: {len(df)}")
assert len(df) == TOTAL_EXPECTED, f"Expected {TOTAL_EXPECTED}, got {len(df)}"

# --- Stratified sample ---
_, sample = train_test_split(
    df,
    test_size=SAMPLE_SIZE,
    random_state=SEED,
    stratify=df["answer_type"],
)

# --- Verification ---
print("\n===== VERIFICATION =====")
print(f"  {'✓' if len(sample) == SAMPLE_SIZE else '✗'} "
      f"{'PASS' if len(sample) == SAMPLE_SIZE else 'FAIL'}: "
      f"len(sample) == {SAMPLE_SIZE} (got {len(sample)})")

print("\n===== ANSWER TYPE DISTRIBUTION =====")
full_pcts = df["answer_type"].value_counts(normalize=True) * 100
sample_pcts = sample["answer_type"].value_counts(normalize=True) * 100
sample_counts = sample["answer_type"].value_counts()

print(f"  {'Type':<15} {'Full %':>8} {'Sample %':>10} {'Sample N':>10} {'Delta':>8}")
print(f"  {'-'*55}")
max_delta = 0.0
for atype in full_pcts.index:
    fp = full_pcts[atype]
    sp = sample_pcts.get(atype, 0.0)
    sc = sample_counts.get(atype, 0)
    delta = abs(fp - sp)
    max_delta = max(max_delta, delta)
    print(f"  {atype:<15} {fp:>7.1f}% {sp:>9.1f}% {sc:>10} {delta:>7.1f}pp")

within_threshold = max_delta <= 2.0
print(f"\n  {'✓' if within_threshold else '✗'} "
      f"{'PASS' if within_threshold else 'FAIL'}: "
      f"max delta = {max_delta:.1f}pp (threshold: 2.0pp)")

# --- Save ---
records = sample.to_dict(orient="records")
out_path = os.path.join(OUT_DIR, "sample.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"\n  Saved {out_path} ({len(records)} records)")
