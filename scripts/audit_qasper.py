"""Step A2: Download and audit QASPER dataset."""
import os
import json
import tarfile
import urllib.request
import tempfile
import pandas as pd

PROJ = "/mnt/d/Projects/Domain-Aware-Prompt-Engineering-Unifying-Scientific-Benchmarks-with-Context-Robust-RAG-Evaluation"
RAW_DIR = os.path.join(PROJ, "data", "qasper", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

print("=" * 60)
print("A2: DOWNLOADING QASPER")
print("=" * 60)

url_td = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
url_test = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

with tempfile.TemporaryDirectory() as tmpdir:
    print("Downloading train+dev tarball...")
    td_path = os.path.join(tmpdir, "train-dev.tgz")
    urllib.request.urlretrieve(url_td, td_path)
    print(f"  Downloaded: {os.path.getsize(td_path)} bytes")

    print("Downloading test tarball...")
    test_path = os.path.join(tmpdir, "test.tgz")
    urllib.request.urlretrieve(url_test, test_path)
    print(f"  Downloaded: {os.path.getsize(test_path)} bytes")

    print("Extracting...")
    with tarfile.open(td_path, "r:gz") as tar:
        tar.extractall(tmpdir)
    with tarfile.open(test_path, "r:gz") as tar:
        tar.extractall(tmpdir)

    # Find and save JSON files
    all_data = {}
    for root, dirs, files in os.walk(tmpdir):
        for f in sorted(files):
            if f.endswith(".json") and "qasper" in f.lower():
                full = os.path.join(root, f)
                dest = os.path.join(RAW_DIR, f)
                with open(full) as inf:
                    data = json.load(inf)
                with open(dest, "w", encoding="utf-8") as outf:
                    json.dump(data, outf, ensure_ascii=False, indent=2)
                all_data[f] = data
                print(f"  Saved: {f} - {len(data)} papers")

print()
print("=" * 60)
print("NESTED STRUCTURE DESCRIPTION")
print("=" * 60)

# Describe structure from first paper
sample_file = "qasper-train-v0.3.json"
if sample_file not in all_data:
    sample_file = list(all_data.keys())[0]

first_paper_id = list(all_data[sample_file].keys())[0]
first_paper = all_data[sample_file][first_paper_id]

print("Top-level keys per paper:")
for key in first_paper.keys():
    val = first_paper[key]
    if isinstance(val, list):
        print(f"  {key}: list of {len(val)} items")
        if len(val) > 0 and isinstance(val[0], dict):
            print(f"    item keys: {list(val[0].keys())}")
            # Check for nested answers
            if "answers" in val[0]:
                ans = val[0]["answers"]
                if len(ans) > 0:
                    print(f"    answers[0] keys: {list(ans[0].keys())}")
                    if "answer" in ans[0]:
                        print(f"    answer fields: {list(ans[0]['answer'].keys())}")
    elif isinstance(val, str):
        preview = val[:100] + "..." if len(val) > 100 else val
        print(f"  {key}: str - {preview}")
    elif isinstance(val, dict):
        print(f"  {key}: dict with keys {list(val.keys())[:5]}")
    else:
        print(f"  {key}: {type(val).__name__}")


def determine_answer_type(answer_obj):
    """Determine the answer type from a single answer object.

    Logic:
    - If unanswerable is True -> 'unanswerable'
    - If extractive_spans is non-empty -> 'extractive'
    - If free_form_answer is non-empty string -> 'free_form'
    - If yes_no is not None (True or False) -> 'yes_no'
    - Otherwise -> 'unknown'
    """
    if answer_obj.get("unanswerable", False):
        return "unanswerable"
    if answer_obj.get("extractive_spans") and len(answer_obj["extractive_spans"]) > 0:
        return "extractive"
    if answer_obj.get("free_form_answer") and answer_obj["free_form_answer"].strip():
        return "free_form"
    if answer_obj.get("yes_no") is not None:
        return "yes_no"
    return "unknown"


# Flatten all splits into question-level rows
print()
print("=" * 60)
print("FLATTENING TO QUESTION-LEVEL")
print("=" * 60)

rows = []
split_counts = {}
mixed_type_count = 0
total_questions = 0

for filename, papers in sorted(all_data.items()):
    split_name = filename.replace("qasper-", "").replace("-v0.3.json", "")
    split_q_count = 0

    for paper_id, paper in papers.items():
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        qas = paper.get("qas", [])

        for qa in qas:
            total_questions += 1
            split_q_count += 1
            question = qa.get("question", "")
            question_id = qa.get("question_id", "")
            answers = qa.get("answers", [])

            # Determine answer types from ALL annotators
            annotator_types = []
            for ans_entry in answers:
                ans_obj = ans_entry.get("answer", {})
                atype = determine_answer_type(ans_obj)
                annotator_types.append(atype)

            # Check for mixed types
            unique_types = set(annotator_types)
            is_mixed = len(unique_types) > 1
            if is_mixed:
                mixed_type_count += 1

            # Use FIRST annotator's answer type (document this choice)
            first_type = annotator_types[0] if annotator_types else "unknown"

            # Get first annotator's answer text
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
                "paper_title": title,
                "question_id": question_id,
                "question": question,
                "answer_type": first_type,
                "answer_text": first_answer_text,
                "num_annotators": len(answers),
                "is_mixed_type": is_mixed,
                "all_annotator_types": annotator_types,
                "split": split_name,
            })

    split_counts[split_name] = split_q_count

df = pd.DataFrame(rows)

print("\nAnswer type determination method:")
print("  Used FIRST annotator's answer for each question.")
print("  Logic per annotator answer object:")
print("    1. unanswerable=True -> 'unanswerable'")
print("    2. extractive_spans non-empty -> 'extractive'")
print("    3. free_form_answer non-empty -> 'free_form'")
print("    4. yes_no is not None -> 'yes_no'")
print("    5. else -> 'unknown'")

# Count papers and questions
total_papers = sum(len(papers) for papers in all_data.values())
print(f"\n1. Number of papers: {total_papers}")
print(f"   Number of questions (flattened): {len(df)}")
print("   Questions per split:")
for split, count in sorted(split_counts.items()):
    print(f"     {split}: {count}")

print("\n4. Distribution of answer types:")
type_counts = df["answer_type"].value_counts()
type_pcts = df["answer_type"].value_counts(normalize=True) * 100
for atype in type_counts.index:
    print(f"   {atype}: {type_counts[atype]} ({type_pcts[atype]:.1f}%)")

print(f"\n6. Questions with MIXED answer types across annotators: {mixed_type_count}")
print(f"   ({mixed_type_count / len(df) * 100:.1f}% of all questions)")

print("\n7. Distribution of questions per paper:")
qpp = df.groupby("paper_id").size()
print(f"   min:    {qpp.min()}")
print(f"   max:    {qpp.max()}")
print(f"   mean:   {qpp.mean():.1f}")
print(f"   median: {qpp.median():.1f}")

print()
print("=" * 60)
print("3 EXAMPLE QUESTION-ANSWER PAIRS")
print("=" * 60)

# Get examples from different types
shown = set()
for _, row in df.iterrows():
    atype = row["answer_type"]
    if atype not in shown and atype != "unknown":
        shown.add(atype)
        print(f"\n--- Type: {atype} ---")
        print(f"  Paper: {row['paper_title'][:100]}")
        print(f"  Question: {row['question'][:200]}")
        ans = row["answer_text"]
        if len(ans) > 200:
            ans = ans[:200] + "..."
        print(f"  Answer: {ans}")
        print(f"  Annotators: {row['num_annotators']}, Mixed: {row['is_mixed_type']}")
    if len(shown) >= 3:
        break

# If we still need more examples
if len(shown) < 3:
    for atype in ["unanswerable", "yes_no", "extractive", "free_form"]:
        if atype not in shown:
            subset = df[df["answer_type"] == atype]
            if len(subset) > 0:
                row = subset.iloc[0]
                shown.add(atype)
                print(f"\n--- Type: {atype} ---")
                print(f"  Paper: {row['paper_title'][:100]}")
                print(f"  Question: {row['question'][:200]}")
                ans = row["answer_text"]
                if len(ans) > 200:
                    ans = ans[:200] + "..."
                print(f"  Answer: {ans}")
        if len(shown) >= 3:
            break
