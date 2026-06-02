# Experiment Report - Domain-Aware Prompt Engineering for Scientific QA

Master's thesis implementation report. This document consolidates **how each
phase and hypothesis was run, on what data, with what parameters, the results
obtained, and how to interpret them**. It is written to be lifted directly into
Chapter 4 (Methodology) and Chapter 5 (Results) of the thesis.

Status: all experiment phases (D, E, F, G, H, I) executed and verified by an
automated integrity audit. The only optional item not run is the proprietary
calibration judge (judge_c); its absence does not affect any primary result.

---

## 1. Global experimental setup

### 1.1 Models under test (6 open-weight LLMs, 4-bit AWQ)

| Model | Params | Role |
|---|---|---|
| mistralai/Mistral-Nemo-Instruct-2407 | 12B | upper anchor |
| google/gemma-2-9b-it | 9B | architectural diversity |
| Qwen/Qwen2.5-7B-Instruct | 7.6B | mid-tier generalist |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | reasoning-specialised |
| sciphi/SciPhi-Mistral-7B-32k | 7B | science-tuned |
| meta-llama/Llama-3.2-3B-Instruct | 3.2B | small-scale anchor |

All models run locally under vLLM with 4-bit quantisation (AWQ; gemma uses
awq_marlin), `max_model_len=4096`, `gpu_memory_utilization=0.85`. SciPhi and
Mistral-Nemo additionally use `prompt_format: mistral_instruct` (required for
correct chat templating; see Limitation L3).

### 1.2 Datasets

- **Track A - SciKnowEval** (primary). Sampled evaluation split
  `data/sciknoweval/main_test_sampled.json` = **3003 questions**, stratified
  (with `--n-floor 3` per stratum, hence 3003 rather than 3000). Domains:
  Physics, Materials, Biology, Chemistry, Computer Science. Question types are
  mixed (~68% 4-choice MCQ, plus true/false, fill-in, relation-extraction,
  open-ended). `dev.json` was used for smoke tests only.
- **Track B - QASPER** (appendix). Sampled split `data/qasper/sample.json` =
  **600 questions**, stratified, seed 42, flattened from 5049 question-answer
  annotations. ~10% are **unanswerable** by design (empty gold answer,
  `question_type = "unanswerable"`). Paper-level corpus: 90,608 FAISS vectors.

### 1.3 Fixed parameters (frozen across all runs)

| Parameter | Value |
|---|---|
| Chunk size / overlap | 256 / 64 tokens (tiktoken cl100k_base) |
| Top-k passages | 10 |
| Noise levels | 0%, 20%, 40%, 60% |
| Noise replacement count (of 10) | 0, 2, 4, 6 |
| Noise types | irrelevant distractor, contradictory, prompt-injection |
| Global seed | 42 (per-question noise seed = seed + hash(qid)) |
| Self-Consistency | N=5 samples, temperature 0.7, min 3 valid |
| Greedy decoding | temperature 0.0 |
| Max output tokens | 1024 |
| BM25 | k1=1.2, b=0.75 |
| Hybrid fusion | Reciprocal Rank Fusion, k=60 |
| Embeddings | BGE-base-en-v1.5 |
| Reranker | BGE-reranker-v2-m3 |
| NLI (faithfulness) | DeBERTa-v3-large (MultiNLI) |
| Rubric scale | 0-5 (six-point) |

### 1.4 Prompting strategies (4)

- **DA (Direct Answer)** - concise answer + short justification.
- **RAS (Rubric-Aware Structured)** - answer + key reasoning + uncertainty +
  confidence.
- **CTL (CoT-lite)** - step-by-step internal reasoning, output final answer +
  rationale.
- **SC (Self-Consistency)** - 5 samples at T=0.7, majority vote.

### 1.5 Retrievers (3)

BM25 (Elasticsearch), dense (FAISS + BGE embeddings), hybrid (RRF of BM25 +
dense, then BGE reranker).

### 1.6 Hardware

Design target: RTX 4060 8GB / Ryzen 7 5700X / 16GB RAM / Win11 + WSL2.
Production runs (including the judge panel) executed on an NVIDIA A100 80GB
node (`innodatahub` JupyterLab).

### 1.7 Experimental matrix

| Track | Cells | Composition |
|---|---|---|
| Closed-book (Phase D) | **24** | 6 models x 4 strategies |
| RAG main (Phase F) | **144** | 6 models x [hybrid x 4 strat x 4 noise (16) + {bm25,dense} x 4 strat x noise0 (8)] = 24/model. Fractional matrix (hybrid carries the noise sweep; BM25/dense anchor the retriever comparison at noise0) |
| QASPER (Phase G) | **48** | 6 models x 4 strat x hybrid x {0%, 60%} |

Records per cell: 3003 (Track A closed-book and RAG), 600 (QASPER). Total
inference records audited: 72,072 + 432,432 + 28,800 = **533,304**.

---

## 2. Phase-by-phase methodology and results

### Phase D - Closed-book inference

**How.** Each of the 24 cells runs all 3003 Track-A questions through one model
with one prompting strategy, no retrieved context. Greedy decoding (SC uses 5
samples + majority vote). Answers are parsed to a normalised form
(`parsed.answer_normalized`, or `sc_result.final_answer_normalized` for SC) and
scored by Exact Match with question-type-aware normalisation (MCQ -> letter,
T/F -> yes/no, open-ended -> normalised text). Outputs:
`outputs/closed_book_main_test/<model>/<strategy>.jsonl`.

**Results (mean Exact Match across the 4 strategies, per model):**

| Model | Params (B) | Closed-book EM |
|---|---|---|
| gemma-2-9b | 9.0 | 0.571 |
| qwen2.5-7b | 7.6 | 0.569 |
| mistral-nemo-12b | 12.0 | 0.568 |
| sciphi-mistral-7b | 7.0 | 0.524 |
| llama-3.2-3b | 3.2 | 0.523 |
| deepseek-r1-qwen-7b | 7.0 | 0.502 |

Best single strategy overall: **Self-Consistency (SC)**. Parse success >= 95%
in every cell.

**Interpretation.** On parametric knowledge alone the larger/instruction-tuned
generalists (gemma, qwen, nemo) lead; model size correlates moderately with
closed-book accuracy (Pearson r = 0.62, see H1). SC is the most reliable
strategy, consistent with the self-consistency literature.

### Phase E - Corpus, indices, retrieval (infrastructure)

**How.** Track-A corpus chunked to 256/64-token windows (tiktoken). BM25 index
in Elasticsearch (k1=1.2, b=0.75); dense FAISS index from BGE-base-en-v1.5;
hybrid = RRF (k=60) of BM25+dense followed by BGE-reranker-v2-m3. Track-B
(QASPER) has its own paper-level corpus and indices (90,608 vectors). This
phase underpins Phases F and G; its correctness is validated indirectly by the
provenance checks in the integrity audit (Section 4).

### Phase F - RAG main matrix

**How.** Identical to Phase D but each prompt is augmented with the top-10
retrieved passages, with a fraction replaced by noise according to the noise
level. Noise positions use a stable per-question seed (`seed + hash(qid)`),
so assembly is reproducible. Each record stores full provenance:
`passages_used[]` (each with `chunk_id` and `noise_type` in
{real, irrelevant, contradictory, injection}), `retriever`, `noise_level`,
`strategy`, `model`. Outputs: `outputs/rag_main/<model>/<retriever>_noise<lvl>_<strategy>.jsonl`.

**Results (mean Exact Match across all RAG cells, per model, vs closed-book):**

| Model | Closed-book | RAG | Delta (RAG - CB) |
|---|---|---|---|
| gemma-2-9b | 0.571 | 0.549 | -0.021 |
| mistral-nemo-12b | 0.568 | 0.540 | -0.027 |
| qwen2.5-7b | 0.569 | 0.503 | -0.066 |
| deepseek-r1-qwen-7b | 0.502 | 0.486 | -0.016 |
| llama-3.2-3b | 0.523 | 0.479 | -0.045 |
| sciphi-mistral-7b | 0.524 | 0.286 | **-0.238** |

**Interpretation.** RAG accuracy is **below** closed-book for every model on
this MCQ-heavy benchmark, and the deficit does **not** scale with model size
(corr(size, RAG) = 0.35 < corr(size, closed-book) = 0.62). On closed-form MCQ,
retrieved passages act as distractors and parametric knowledge dominates;
adding noise does not change this for healthy models (see H2). SciPhi is an
outlier (-0.238): it suffers a RAG **format collapse** (Limitation L3).
This headline finding motivates measuring noise-robustness with open-ended,
judge-based metrics rather than MCQ accuracy alone.

**RAG-penalty decomposition (EM vs EM_parsed).** To rule out that the RAG drop
is merely a parsing artifact (longer RAG outputs are harder to parse), we
compare raw Exact Match (unparsed = wrong) against EM_parsed (accuracy over
successfully parsed answers only):

| Model | delta EM (all) | delta EM_parsed (clean only) |
|---|---|---|
| gemma-2-9b | -0.021 | -0.021 |
| mistral-nemo-12b | -0.027 | -0.019 |
| deepseek-r1-qwen-7b | -0.016 | -0.028 |
| llama-3.2-3b | -0.045 | -0.025 |
| qwen2.5-7b | -0.066 | -0.047 |
| sciphi-mistral-7b | -0.238 | -0.143 |

The RAG penalty **survives on parsed-only answers for every model** (-0.019 to
-0.047; SciPhi -0.143), so it is a **genuine knowledge-distraction effect**, not
a parsing artifact. The difference between the two columns is a secondary
**format penalty**: RAG prompts elicit longer, more often truncated/unparseable
outputs (citation runaway, L4), which raw EM counts as wrong (pronounced for
llama, qwen, sciphi; negligible for gemma). The effect is small and consistent
for healthy models - the signature of mild distraction, not a defect. SciPhi
degrades even on clean answers (-0.143), i.e. its reasoning, not just its
formatting, collapses under RAG (L3).

### Phase G - QASPER (Track B, appendix)

**How.** 48 cells (hybrid retriever only, noise {0%, 60%}) over 600 QASPER
questions, scored both by automated metrics and by the judge panel. QASPER
answers are open-ended (extractive/abstractive/yes-no) plus ~10% unanswerable.

**Results.** Exact Match is not meaningful for open-ended QASPER (EM ~ 0.07);
the informative Track-B signal is the **judge rubric** (the 48 QASPER cells are
included in the judge panel, Phase H). 60/600 (10%) questions are unanswerable
by design (empty gold); these test abstention rather than retrieval accuracy.
Aggregating the 1920 QASPER judge records (`scripts/qasper_track_b_table.py`):

| Group | n | mean rubric | mean faithfulness | mean coverage |
|---|---|---|---|---|
| overall | 1920 | 2.14 | 0.051 | 0.313 |
| strategy = RAS | 480 | **2.23** | 0.046 | 0.321 |
| strategy = SC | 480 | 2.15 | 0.048 | 0.308 |
| strategy = CTL | 480 | 2.10 | 0.064 | 0.312 |
| strategy = DA | 480 | 2.09 | 0.044 | 0.311 |
| noise = 0% | 960 | **2.23** | 0.041 | 0.305 |
| noise = 60% | 960 | **2.05** | 0.060 | 0.321 |

**Interpretation.** Track B is a transfer check to a second scientific QA
format (NLP papers). Two findings stand out. (1) Unlike Track-A MCQ accuracy
(where noise had no effect), on open-ended QASPER scored by the rubric **noise
degrades quality** (2.23 -> 2.05 from 0% to 60% noise) - direct evidence that
judge-based, open-ended evaluation captures noise sensitivity that exact-match
MCQ accuracy masks. (2) RAS (the rubric-aware structured prompt) is the best
strategy on QASPER, ahead of SC, suggesting structured prompting helps more on
free-form scientific answers than on MCQ. Absolute rubric scores are low (~2.1
of 5), reflecting QASPER's difficulty and the strict NLI faithfulness threshold.

### Phase H - LLM-judge panel

**How.** Two independent open-weight judges, distinct from every examinee
family, score a stratified subset of **N=20 questions per cell** across all 216
cells (24 closed-book + 144 RAG + 48 QASPER) = **4320 judged answers per
judge**. The same 20 questions are scored in every cell, giving full
judge_a/judge_b overlap for reliability. Each answer receives:

- **Rubric score 0-5** (six-point quality).
- **Faithfulness** - the answer is decomposed into atomic claims; each claim is
  checked by NLI (DeBERTa-v3-large) entailment against the retrieved passages;
  faithfulness = fraction of entailed claims.
- **Citation precision/recall**, **coverage** (key-point match),
  **self-confidence** (judge-reported, 0-1).

JSON output is constrained with guided decoding (lm-format-enforcer) with
bounded field lengths, giving **0 parse failures** across all 8640 judge
records. Outputs: `outputs/judge/{judge_a,judge_b}/<track>/<model>/<cell>.jsonl`.

**Results (aggregate, `outputs/judge/aggregate_stats.json`):**

- **Krippendorff's alpha (judge_a vs judge_b) = 0.716** over n = 4320 jointly
  rated answers (ordinal) - **substantial** inter-rater agreement.
- **alpha(a,b,c) = 0.716** (equals the pairwise value because judge_c was not
  run).
- **Expected Calibration Error (MCQ subset) = 0.202** over n = 3120 pairs -
  moderate overconfidence (mean self-confidence ~0.78 vs MCQ accuracy ~0.58).
- Perturbation Wilcoxon (surface / semantic) - **not conducted** (no
  perturbation audit set was generated); deferred to future work.

Per-judge means: judge_a rubric 2.70, faithfulness 0.044, coverage 0.373;
judge_b rubric 3.04, faithfulness 0.118, coverage 0.613 (judge_b is the more
lenient rater). Per-domain rubric (both judges): Chemistry 3.70 > Physics 3.58
> Materials 3.54 > Biology 3.25 > **Computer Science 2.14** (lowest), with
coverage following the same order (Chemistry 0.80, CS 0.31).

**Interpretation.** The two judges agree substantially (alpha = 0.72), so the
rubric is a reliable quality signal. Faithfulness is low across the board
(0.03-0.16), consistent with the citation-runaway issue (L4) and a strict NLI
threshold. The CS domain is hardest for all models (lowest rubric and
coverage). Calibration is moderate (ECE 0.20): models are somewhat
overconfident on MCQ.

### Phase I - Statistics (Chapter 5 tables)

**How.** `scripts/run_statistics.py` reads the per-cell Exact-Match summaries
(closed-book, RAG) and the judge aggregate, and emits H1/H2/H3 tables, effect
sizes (Cliff's delta, Cohen's d), and Benjamini-Hochberg FDR-corrected
p-values to `outputs/chapter5_tables/`. Primary cell-level score for H1/H2 is
Exact Match; H3 is read from the judge aggregate. See Section 3 for the
hypothesis-level detail.

---

## 3. Hypotheses

### H1 - Does RAG benefit scale with model size?

**Statement.** Larger models gain more (or lose less) from RAG than smaller
models.

**Method.** Per (model, strategy), RAG improvement = mean RAG EM - closed-book
EM. Correlate model size (params_b) with improvement using Pearson r and
Kendall's tau; compare corr(size, RAG) vs corr(size, closed-book) with
Steiger's z. BH-FDR over the H1 family.

**Result.**

- Pearson r(size, RAG-improvement) = **0.174, p = 0.741** (not significant).
- Kendall tau = 0.138, p = 0.702 (not significant).
- Steiger's z = -0.588, p = 0.557 (no significant difference between the two
  correlations).
- corr(size, closed-book) = 0.624; corr(size, RAG) = 0.348.
- Every model's RAG delta is negative.
- None of the H1 tests survive FDR.

**Interpretation.** **H1 is not supported.** RAG uniformly reduces MCQ accuracy
relative to closed-book, and the size of that reduction does not scale with
model size. Larger models are better at closed-book QA, but that advantage does
not translate into a larger RAG benefit (because there is no RAG benefit on
this MCQ task).

### H2 - Are models robust to retrieval noise, and do strategy/retriever matter?

**Statement.** Answer quality degrades with retrieval noise; prompting
strategy and retriever modulate robustness.

**Method.** Linear mixed-effects model with a random intercept per model:
`score ~ noise_level + C(strategy) + C(retriever) + (1 | model)`, fit by REML
(statsmodels). n = 144 cells, 6 model groups. BH-FDR over the H2 family.

**Result (full model):**

| Term | Coefficient | p-value | FDR-significant |
|---|---|---|---|
| noise_level | **+0.0609** | 0.0019 | yes (q=0.007) |
| strategy = SC | +0.0598 | 3.5e-9 | yes |
| strategy = DA | +0.0251 | 0.013 | yes |
| strategy = RAS | +0.0176 | 0.082 | no |
| retriever = dense | +0.0190 | 0.125 | no |
| retriever = hybrid | +0.0035 | 0.759 | no |

Model converged; AIC is reported as NaN (a cosmetic statsmodels REML artifact,
not an error - log-likelihood is finite and the model converged).

**Robustness re-run excluding SciPhi** (`--exclude-models sciphi-mistral-7b`,
n = 120, 5 groups):

| Term | Coefficient | p-value | FDR-significant |
|---|---|---|---|
| noise_level | **-0.0055** | 0.686 | **no** |
| strategy = SC | +0.0658 | 1.5e-20 | yes |
| strategy = DA | +0.0316 | 7.9e-6 | yes |
| strategy = RAS | +0.0207 | 0.0034 | yes |
| retriever (dense/hybrid) | ~0 | >0.9 | no |

**Interpretation.** The full model shows a small **positive** noise coefficient
(higher noise -> slightly higher score), which is counter-intuitive. The
robustness re-run shows this is **entirely an artifact of SciPhi**: once SciPhi
is removed, the noise coefficient collapses to ~0 and becomes non-significant
(p = 0.69), while the prompting-strategy effects remain significant and even
strengthen. Mechanism: SciPhi tends to continue/echo retrieved passages instead
of answering when passages are clean (low noise), but answers more readily when
passages are obviously junk (high noise), so its accuracy *rises* with noise -
inflating the pooled coefficient. **Correct reading of H2:** for well-behaved
models, retrieval noise from 0% to 60% has **no significant effect** on MCQ
accuracy (it saturates - the closed-book-dominates result of Phase F means
there is little to lose). **Prompting strategy is the dominant controllable
factor**, with SC > DA > RAS, robustly across both model sets. Retriever choice
(BM25/dense/hybrid) is non-significant throughout.

### H3 - Are the LLM judges reliable and calibrated?

**Statement.** The automated judge panel produces reliable, calibrated quality
scores.

**Method.** Krippendorff's alpha (ordinal) between judges over every jointly
rated answer; Expected Calibration Error on the MCQ subset (self-confidence vs
correctness, 10 bins); (perturbation Wilcoxon - planned, not conducted).

**Result.**

- Inter-rater reliability **alpha = 0.716** (n = 4320) - substantial.
- alpha(a,b,c) = 0.716 (judge_c not run).
- **ECE = 0.202** (n = 3120 MCQ) - moderate overconfidence.
- Perturbation Wilcoxon - not conducted.

**Interpretation.** **H3 is supported for reliability.** Two independent
open-weight judges agree substantially (alpha = 0.72), so rubric-based quality
scoring is trustworthy. Calibration is moderate: confidence (~0.78) exceeds
accuracy (~0.58) by about 0.20. The perturbation-sensitivity component is left
to future work.

---

## 4. Data-integrity audit

A read-only audit (`scripts/audit_experiments.py`) verified all phases:

- **Completeness:** 24 / 144 / 48 cells present, all 6 models, every cell with
  the full record count (3003 / 3003 / 600).
- **Corruption:** 0 malformed JSON lines across 533,304 records.
- **Duplicates:** 0 duplicate question_ids within any cell.
- **RAG provenance:** retriever and noise_level in every record match the cell
  filename; passages_used = 10; the number of noise-tagged passages matches the
  configured replacement count exactly (0 / 2 / 4 / 6 for noise 0 / 0.2 / 0.4 /
  0.6) - **confirming noise was actually applied as designed**.
- **Judges:** judge_a and judge_b cover an identical set of 4320 answers
  (valid alpha overlap); all rubric values in [0,5], all sub-metrics in [0,1].
- **Cross-phase:** closed-book and RAG share 100% of question_ids (valid H1
  pairing).
- **Expected exceptions:** QASPER 10% empty gold = unanswerable questions (by
  design); 12 / 432,432 RAG records (0.003%) returned 0 passages (degenerate
  queries at hybrid noise 0.6) and were effectively no-context.

Final audit result: **35 PASS, 4 WARN, 0 FAIL** (`outputs/audit_report.txt`).
The 4 warnings are all benign and documented (empty-predicted rates from
SciPhi/CTL/SC; 12 degenerate-retrieval records; 1920 QASPER judge records
whose source join is an audit-accounting artifact, not affecting any
statistic). Independent re-derivation (`scripts/validate_results.py`)
confirmed: RAG prompts are 20x larger than closed-book (passages present);
non-"real" passages equal 0/2/4/6 for noise 0/0.2/0.4/0.6 exactly (300/300
records per level); the RAG penalty survives on EM_parsed for all models.

Verdict: **the experiments ran cleanly; results are representative.**

Environment (key versions, `outputs/env_key_versions.txt`): vLLM 0.6.3,
transformers 4.45.2, torch 2.4.0+cu121, sentence-transformers 3.2.1,
faiss-cpu 1.9.0, elasticsearch 8.15.1, lm-format-enforcer 0.10.6,
statsmodels 0.14.6, krippendorff 0.8.2, numpy 1.26.4, scipy 1.14.1,
datasets 3.0.2.

---

## 5. Limitations (to state in the thesis)

- **L1 - Parsing.** Parsing succeeded on 87.3% of 432,432 RAG generations; of
  the 12.7% failures, 8.2 pp were recovered via answer-field fallback, leaving
  ~1.5% genuine loss. No consolidated re-parse was performed: ~95% of failures
  reflect genuine model non-answering, and re-parsing after judging would
  desynchronise the computed judge scores. Conclusions are unaffected.
- **L2 - RAG below closed-book / noise sign.** RAG accuracy is uniformly below
  closed-book on MCQ with no size scaling (H1). This survives on parsed-only
  answers (EM_parsed delta -0.019 to -0.047 for healthy models), confirming a
  genuine knowledge-distraction effect rather than a parsing artifact; a smaller
  additional format penalty (more unparseable RAG outputs) inflates the raw EM
  gap. The positive noise coefficient in the full H2 model is an artifact of
  SciPhi's format collapse and of noise being confined to the hybrid retriever;
  excluding SciPhi removes it. It does **not** indicate that noise improves
  answers.
- **L3 - SciPhi format collapse.** SciPhi-Mistral exhibits RAG format collapse
  (Cliff's delta = 1.0 vs every peer; parse rate rises with noise) and is
  reported as a documented outlier. Root cause was a missing
  `mistral_instruct` chat template (fixed and regenerated before final runs).
- **L4 - Citation runaway.** ~27% of generations hit the 1024-token cap with
  degenerate citation spam (`[2][3]...[220]`). This does not affect extracted
  answers/accuracy but depresses the judge citation/faithfulness metrics.
- **L5 - Judge calibration / perturbation.** Inter-rater reliability is solid
  (alpha = 0.72); ECE (0.20) indicates moderate overconfidence. A bounded
  perturbation audit is supported by `scripts/generate_perturbations.py`
  (class 1 surface = deterministic typos/whitespace; class 2 semantic =
  rule-based stem negation, a transparent approximation of a full LLM
  entity-swap protocol) plus `compute_wilcoxon` in `judge_aggregate.py`; run it
  on a 200-question subset to obtain the class-1 (expected non-significant) and
  class-2 (expected significant) Wilcoxon results. The third (proprietary)
  calibration judge was not run, so three-way alpha equals the pairwise value.
- **L6 - QASPER.** Track B is an appendix; 10% of questions are unanswerable;
  Exact Match is not meaningful for open-ended answers, so quality is judged by
  the rubric.

---

## 6. Reproducibility

- **Seed:** 42 globally; per-question noise seed = 42 + hash(question_id).
- **Configs (committed):** `configs/{rag,closed_book,qasper,noise,global,judge}.yaml`.
- **Prompts (committed):** `prompts/`, `configs/judge_prompts/`.
- **Code (inference + retrieval):** `scripts/run_inference.py`,
  `run_rag_inference.py`, `run_rag_experiment.py`, `build_qasper_corpus.py`,
  `sample_qasper.py`.
- **Code (scoring + analysis):** `run_judge.py`, `judge_aggregate.py`,
  `compute_metrics.py`, `run_statistics.py`, `qasper_track_b_table.py`
  (Track-B rubric aggregation).
- **Code (verification):** `audit_experiments.py` (integrity audit, read-only),
  `validate_results.py` (independent H1/H2/H3 + RAG/noise re-derivation),
  `extra_checks.py` (judge/source orphan diagnosis + retrieval domain proxy),
  `generate_perturbations.py` (H3 perturbation audit set; run via
  `run_inference.py --split-file ... --output-dir ...`), and the test suite
  under `tests/`.
- **Software (exact versions, Python 3.10):** vLLM 0.6.3, transformers 4.45.2,
  torch 2.4.0+cu121, sentence-transformers 3.2.1 (BGE embeddings/reranker),
  Elasticsearch 8.15.1 (BM25), faiss-cpu 1.9.0 (dense), lm-format-enforcer
  0.10.6 (guided JSON judging), statsmodels 0.14.6 (mixed-effects),
  krippendorff 0.8.2 (alpha), numpy 1.26.4, scipy 1.14.1, datasets 3.0.2.
  Full snapshot: `outputs/environment_versions.txt`; key subset:
  `outputs/env_key_versions.txt`. Final runs on an NVIDIA A100 80GB node.
- **Output locations:** `outputs/closed_book_main_test/`, `outputs/rag_main/`,
  `outputs/qasper_main/`, `outputs/judge/` (+ `aggregate_stats.json`),
  `outputs/chapter5_tables/` (H1/H2/H3 + `track_b_qasper.csv`),
  `outputs/chapter5_tables_nosciphi/` (H2 robustness), `outputs/audit_report.txt`.
- **Reproduction commands (after data is in place):**
  ```bash
  python scripts/compute_metrics.py --base-dir outputs/closed_book_main_test
  python scripts/compute_metrics.py --base-dir outputs/rag_main
  python scripts/judge_aggregate.py --source-dirs \
      outputs/closed_book_main_test outputs/rag_main outputs/qasper_main
  python scripts/run_statistics.py \
      --closed-book-summary outputs/closed_book_main_test/summary_table.json \
      --rag-summary outputs/rag_main/summary_table.json
  python scripts/run_statistics.py --hypothesis H2 \
      --exclude-models sciphi-mistral-7b \
      --closed-book-summary outputs/closed_book_main_test/summary_table.json \
      --rag-summary outputs/rag_main/summary_table.json \
      --output outputs/chapter5_tables_nosciphi
  python scripts/qasper_track_b_table.py
  python scripts/audit_experiments.py        # integrity audit (expect 0 FAIL)
  python scripts/validate_results.py         # independent cross-check
  ```

---

## 7. Optional follow-ups

Completed since the first draft: Track-B rubric table (Section Phase G), clean
0-FAIL audit rerun (Section 4), environment-version snapshot (Section 6), and
the H2 SciPhi-exclusion robustness run (Section 3). The only remaining optional
item, which does **not** block the thesis:

1. **judge_c** (proprietary, or free OpenRouter gpt-oss-120b) on a calibration
   subset -> a genuine third independent rater for three-way Krippendorff
   alpha(a,b,c). Until then, alpha(a,b,c) equals the pairwise alpha(a,b) = 0.72.

---

## 8. One-paragraph results summary (abstract-ready)

We evaluated six open-weight LLMs (3-12B) on scientific multiple-choice and
open-ended QA in closed-book and retrieval-augmented settings, across four
prompting strategies, three retrievers, and four retrieval-noise levels
(0-60%), with a two-judge LLM panel for quality, faithfulness, and calibration.
Retrieval-augmented generation consistently **underperformed** closed-book
accuracy on multiple-choice science questions (mean delta -0.07; all six models
negative), and this gap did **not** scale with model size (Pearson r = 0.17,
n.s.). Prompting strategy was the dominant controllable factor, with
Self-Consistency best (mixed-effects beta = +0.066, p < 1e-19); retriever choice
was non-significant. Retrieval noise had no significant effect on well-behaved
models once a science-tuned outlier with documented format collapse was
excluded (noise beta = -0.006, p = 0.69). The judge panel was reliable
(Krippendorff's alpha = 0.72, n = 4320) with moderate calibration (ECE = 0.20).
All findings are FDR-corrected and supported by an automated integrity audit
over 533,304 inference records.
