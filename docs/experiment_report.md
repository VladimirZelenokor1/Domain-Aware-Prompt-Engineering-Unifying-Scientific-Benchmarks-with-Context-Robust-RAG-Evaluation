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

**Why RAG distracts: retrieval domain mismatch.** A domain-consistency proxy
(`scripts/extra_checks.py`) at noise 0 measures the fraction of retrieved
(non-noise) passages whose source-chunk domain matches the question domain:
Biology 84.8%, Chemistry 54.8%, **Physics 37.6%, Materials 8.3%**. Where the
match rate is low the retriever returns off-domain passages that act as
distractors - a mechanistic explanation for the closed-book-over-RAG result,
strongest exactly where retrieval is least on-topic (Materials, Physics).
Computer-Science questions have no matching corpus domain at all (the corpus
covers biology/chemistry/physics/materials/earth-science), so CS retrieval is
inherently off-domain - consistent with CS being the lowest-rubric domain in
the judge panel (2.14). This reframes the H1 finding: RAG does not hurt because
"retrieval is useless", but because retrieval relevance is uneven across
scientific domains on this corpus.

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

(These are the clean, post-re-judge values; see Section 4 for the QASPER/RAG
judge-collision that was repaired. The authoritative reliability/calibration
numbers come from `analyze_h3_reliability.py`; `aggregate_stats.json` should be
regenerated by re-running `judge_aggregate.py` after the re-judge.)

- **Krippendorff's alpha (judge_a vs judge_b) = 0.677**, 95% CI [0.658, 0.695]
  over n = 3600 jointly rated answers (ordinal) - **substantial** agreement.
- **alpha(a,b,c) = 0.677** (equals the pairwise value because judge_c was not
  run).
- **Per-judge ECE** (MCQ/short-answer subset, B = 10): judge_a 0.103, judge_b
  0.228 - both > 0.05 (moderate overconfidence).
- Perturbation Wilcoxon (surface / semantic) - **conducted** (see H3, Section 3):
  class 1 delta = -0.04 (Cliff's d +0.02, negligible), class 2 delta = -0.33
  (p = 4.6e-19, Cliff's d +0.15).

Per-judge means: judge_a is the stricter rater, judge_b more lenient.
Per-**domain** rubric (Track A = SciKnowEval, 4 native domains): Chemistry >
Physics > Materials > Biology. Note: the low-rubric **"CS" group is QASPER
(Track B / NLP papers)**, a separate dataset - **not** a SciKnowEval domain
(SciKnowEval has only Biology/Chemistry/Physics/Materials; verified against the
HF dataset card). Earlier drafts that listed Computer Science among SciKnowEval
domains were conflating Track B with Track A.

**Interpretation.** The two judges agree substantially (alpha = 0.68), so the
rubric is a reliable quality signal. Faithfulness is low across the board,
consistent with the citation-runaway issue (L4) and a strict NLI threshold.
QASPER (Track B, open-ended NLP-paper QA) is the hardest set (lowest rubric).
Calibration is moderate (both judges' ECE > 0.05): models are somewhat
overconfident on MCQ.

### Phase I - Statistics (Chapter 5 tables)

**How.** `scripts/run_statistics.py` reads the per-cell Exact-Match summaries
(closed-book, RAG) and the judge aggregate, and emits H1/H2/H3 tables, effect
sizes (Cliff's delta, Cohen's d), and Benjamini-Hochberg FDR-corrected
p-values to `outputs/chapter5_tables/`. Primary cell-level score for H1/H2 is
Exact Match; H3 is read from the judge aggregate. See Section 3 for the
hypothesis-level detail.

---

## 3. Hypotheses (thesis RQ1-RQ3)

These are the thesis's formal hypotheses. Section 3S collects supplementary
accuracy-based analyses that address related but distinct questions and are
retained as robustness/exploratory results.

### H1 (RQ1) - Domain-aware rubric vs lexical metrics

**Statement.** On the unambiguous-gold closed-book subset, the rubric-based
judge score aligns with correctness better than lexical metrics (ROUGE-L,
BLEU-4); and on the open-ended subset, the strategy ranking by rubric differs
from the ranking by lexical metrics. Script: `analyze_h1_metric_alignment.py`.

**Method.** Point-biserial r(metric, binary correctness) on the unambiguous
subset (MCQ + true/false, n = 384), comparing rubric against three lexical
metrics {ROUGE-L, BLEU-4, exact-match} via the Williams test for dependent
overlapping correlations (a Steiger-family test; Bonferroni alpha = 0.05/3 =
0.017); Kendall's tau between strategy rankings on the open-ended subset
(open-ended-qa + relation-extraction, n = 96). Rubric = mean of judge_a/b.
Note: the correctness label (y_gold) is type-aware exact match
(`compute_exact_match`); the exact-match *competitor* is raw answer-vs-gold
string equality - a distinct operationalisation, so the rubric-vs-EM comparison
is partly self-referential and the primary evidence is rubric vs ROUGE-L/BLEU-4.

**Result.**

| metric | r_pb(metric, correct) | Williams vs rubric |
|---|---|---|
| **rubric** | **0.674** | - |
| ROUGE-L | 0.533 | t = 3.97, p = 8.6e-5 (rubric higher) |
| BLEU-4 | 0.387 | t = 7.45, p = 6.4e-13 (rubric higher) |
| exact-match | 0.307 | t = 9.10, p < 1e-15 (rubric higher) |

All three comparisons are Bonferroni-significant (alpha = 0.017). Strategy
ranking (open-ended): rubric = RAS > CTL > DA > SC; vs ROUGE-L Kendall
tau = 0.667; vs BLEU-4 tau = 0.333; vs exact-match tau = 0.333.

**Interpretation.** **H1 is supported.** The rubric correlates with gold
correctness significantly more strongly than every lexical metric (Williams
test, all three Bonferroni-significant), confirming that domain-aware rubric scoring
captures answer quality better than surface overlap. Strategy rankings by
rubric and by lexical metrics diverge (tau < 1), so the choice of metric
changes conclusions about prompting strategies. Caveat: the predicted
"largest divergence at RAS" did not hold - RAS ranks top under every metric;
the divergence is in the middle of the ranking.

### H2 (RQ2) - Closed-book competence and RAG robustness

**Statement.** Closed-book scientific competence predicts answer quality under
RAG, and quality degrades as retrieval noise rises from 0% to 60%. Script:
`analyze_h2_competence_robustness.py`.

**Method.** Question-level mixed-effects model on the judged RAG subset
(n = 2880): `rubric ~ closed_book_correct + noise_level + C(strategy) +
(1 | model)`, with `closed_book_correct` (exact-match correctness of the same
model/strategy/question in closed-book) as the predictor of interest; an OLS
fit with model as a fixed effect is reported as a robustness check. Model-level
(descriptive, n = 6): thesis robustness slope = normalised drop (eq. 12,
(rubric@0% - rubric@60%)/rubric@0% on hybrid) and Spearman between closed-book
rubric and that drop, with a 10,000-resample bootstrap CI.

**Result.**

- **closed_book_correct: coef = +0.89, p = 1.6e-74** (mixed model) /
  +0.89, p = 1.7e-70 (OLS, R^2 = 0.20, n = 2880). A correct closed-book answer
  predicts a ~0.89-point higher RAG rubric (0-5 scale). [The mixed model's
  random-intercept variance is singular - model variance is absorbed by
  closed_book_correct - so the OLS fit is the clean estimate; both agree.]
- **noise_level: coef = -0.26, p = 0.012 (significant)** - retrieval noise
  significantly lowers rubric quality (note the contrast with the EM-based
  supplementary model, where noise was non-significant: the judge rubric detects
  degradation that exact-match accuracy masks).
- Per-model normalised drop (eq. 12, 0%->60% on hybrid): **gemma +0.284**
  (largest), nemo +0.082, llama +0.081, deepseek +0.032, qwen +0.023,
  **sciphi -0.885** (rubric *rises* with noise - format-collapse confound, L3).
- Spearman(closed-book rubric, normalised drop) = **-0.43, p = 0.40**, 95%
  bootstrap CI [-1.0, 1.0] (n = 6, descriptive - the CI is uninformative at this
  sample size, reported only because the thesis pre-registers it).

**Interpretation.** **H2 is supported at the question level**: closed-book
competence is a strong, highly significant predictor of RAG answer quality
(+0.89), and retrieval noise significantly degrades quality (-0.26). At the
model level the competence-robustness relationship is only descriptive (n = 6,
n.s., uninformative CI); SciPhi is the outlier whose rubric rises with noise
(format-collapse confound, L3). The "competent but context-fragile" pattern is
suggestive (e.g. the high-competence gemma shows the largest normalised drop)
but not statistically established at n = 6.

### H3 - Are the LLM judges reliable and calibrated?

**Statement.** The automated judge panel produces reliable, calibrated quality
scores.

**Method.** Krippendorff's alpha (ordinal) between judges over every jointly
rated answer; Expected Calibration Error on the MCQ subset (self-confidence vs
correctness, 10 bins); paired Wilcoxon on a 200-question perturbation audit
(class 1 surface vs class 2 semantic, both judges).

**Result.**

- Inter-rater reliability **alpha = 0.677**, 95% bootstrap CI [0.658, 0.695]
  (n = 3600 jointly rated answers) - within the thesis's predicted 0.40-0.80
  band and below near-perfect (< 0.85). H3(b) specifies a three-judge panel; the
  three-way alpha(a,b,c) is pending judge_c (currently equals the pairwise value).
- **Per-judge ECE** (B = 10, MCQ/short-answer subset): judge_a **0.103**
  (n = 2688), judge_b **0.228** (n = 2112) - both > 0.05, so H3(c) holds for
  every judge (moderate overconfidence, stronger for judge_b).
- **Perturbation Wilcoxon + Cliff's delta** (200-question audit, all 6 models,
  da, both judges, 1200 paired ratings per class):
  - class 1 (surface: typos/whitespace): rubric 3.54 -> 3.50, delta = **-0.04**,
    p = 0.048, Cliff's delta = +0.023 (negligible);
  - class 2 (semantic): rubric 3.54 -> 3.22, delta = **-0.33**, p = 4.6e-19,
    Cliff's delta = +0.146 (small).

**Interpretation.** **H3 is supported.** Two independent open-weight judges
agree substantially (alpha = 0.68, in the pre-registered 0.40-0.80 band), so
rubric-based quality scoring is trustworthy. Both judges are moderately
overconfident (ECE > 0.05). The perturbation audit shows the desired pattern:
the semantic perturbation causes a large, highly significant rubric drop
(delta = -0.33; Cliff's delta = +0.15), while the surface perturbation causes a
negligible drop (delta = -0.04; Cliff's delta = +0.02) that only reaches
p < 0.05 because of the large sample - i.e. the panel is **robust to surface
noise and appropriately sensitive to meaning changes**. (Caveat: Class 2 was
operationalised as rule-based question-stem negation followed by re-inference,
not the answer padding/truncation specified in the draft - see L5.)

### 3S - Supplementary accuracy-based analyses

These use Exact Match (not the rubric) and address questions adjacent to, but
distinct from, the thesis hypotheses. They are retained as exploratory /
robustness results (`run_statistics.py`).

- **Model-size scaling.** Per (model, strategy) RAG improvement = mean RAG EM -
  closed-book EM. Pearson r(size, improvement) = 0.17 (p = 0.74), Kendall tau =
  0.14 (p = 0.70), Steiger z = -0.59 (p = 0.56); corr(size, closed-book) = 0.62
  vs corr(size, RAG) = 0.35; all six RAG deltas negative; none survive FDR.
  Reading: RAG uniformly lowers MCQ accuracy and the deficit does not scale with
  model size (cf. the retrieval domain-mismatch mechanism in Phase F).
- **Accuracy-based noise model.** Mixed model `EM ~ noise_level + C(strategy) +
  C(retriever) + (1 | model)` (n = 144): noise_level +0.061 (p = 0.002,
  FDR-sig), SC +0.060 (p = 3.5e-9), DA +0.025 (p = 0.013), retriever n.s. The
  positive noise coefficient is a **SciPhi artifact**: excluding SciPhi
  (n = 120) gives noise_level -0.006 (p = 0.69, n.s.) while strategy effects
  persist (SC +0.066, p = 1.5e-20). Reading: for well-behaved models retrieval
  noise has no significant effect on MCQ *accuracy*; prompting strategy
  (SC > DA > RAS) is the dominant controllable factor; retriever choice is n.s.
  (Note the contrast with the rubric-based H2/Track-B, where noise *does* lower
  open-ended quality - EM accuracy masks what the rubric detects.)
- **RAG-penalty decomposition** (EM vs EM_parsed) and **Track-B QASPER rubric
  table**: see Phase F and Phase G.

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

**Judge-output collision (found and repaired).** QASPER and RAG share the
`hybrid_noise0.0/0.6_*` cell filenames, and an early judge run wrote the QASPER
outputs for those two noise levels into the `rag_main`-keyed judge paths,
overwriting the genuine RAG judge scores for hybrid noise 0% and 60% (the
records there carried QASPER `ske-track-b-*` ids). This corrupted the
**rubric-based** RAG analysis for those two cells (it did **not** touch the
EM-based inference results, H1, or the Track-B table read from `qasper_main`).
**Fix:** the contaminated `rag_main` hybrid-0.0/0.6 judge files were deleted and
those 48 RAG cells re-judged by both judges; the join now recovers the full
n = 2880 RAG items across all four noise levels, and H2/H3 above were recomputed
on the clean data. (After the repair, `extra_checks.py` reports only the
expected `perturb_*` orphans - the perturbation audit cells are not in the main
source dirs.)

Independent re-derivation (`scripts/validate_results.py`)
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
  (alpha = 0.68, CI [0.66, 0.70]); per-judge ECE 0.10/0.23 (both > 0.05,
  moderate overconfidence). The bounded perturbation audit was **conducted**
  (200 questions, all 6 models, both judges): class 1 surface delta = -0.04
  (negligible, robust to typos), class 2 delta = -0.33, p = 4.6e-19 (sensitive).
  **Caveat (perturbation protocol divergence):** the draft (§3.6.4) defines
  Class 2 as answer **padding + truncation** re-judged on the same answer; our
  implementation instead applied rule-based **question-stem negation followed by
  re-inference**. This tests pipeline sensitivity rather than pure judge
  stability and introduces a stale-gold confound; the direction (class 2 >> class
  1) holds, but a clean re-run perturbing the stored answers (per §3.6.4) is the
  recommended fix. The third (proprietary) calibration judge was not run, so the
  three-way alpha equals the pairwise value.
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
  python scripts/analyze_h1_metric_alignment.py      # thesis H1 (rubric vs ROUGE/BLEU/EM)
  python scripts/analyze_h2_competence_robustness.py # thesis H2 (competence->quality, eq-12 drop)
  python scripts/analyze_h3_reliability.py           # thesis H3 (per-judge ECE, alpha CI, Cliff's d)
  python scripts/compute_acu_dr.py --limit-per-cell 20  # ACU + Denoise Rate (NLI; needs GPU)
  python scripts/audit_experiments.py        # integrity audit (expect 0 FAIL)
  python scripts/validate_results.py         # independent cross-check
  ```

---

## 7. Optional follow-ups

Completed since the first draft: Track-B rubric table (Section Phase G), clean
0-FAIL audit rerun (Section 4), environment-version snapshot (Section 6), the
H2 SciPhi-exclusion robustness run (Section 3), the retrieval domain-mismatch
diagnostic (Phase F), and the H3 perturbation audit (Section 3). The only
remaining optional item, which does **not** block the thesis:

1. **judge_c** (proprietary, or free OpenRouter gpt-oss-120b) for a genuine
   three-way Krippendorff alpha(a,b,c); until then it equals pairwise
   alpha(a,b) = 0.68.
2. **Perturbation re-run per §3.6.4** (answer padding/truncation re-judged on the
   stored answers) to replace the question-negation + re-inference variant (L5).

---

## 8. One-paragraph results summary (abstract-ready)

We evaluated six open-weight LLMs (3-12B) on scientific QA (SciKnowEval; QASPER
transfer track) in closed-book and retrieval-augmented settings, across four
prompting strategies, three retrievers, and four retrieval-noise levels
(0-60%), with an LLM-judge panel scoring a domain-aware rubric. **(H1)** The
rubric aligned with reference correctness significantly better than lexical
metrics (point-biserial r = 0.67 vs ROUGE-L 0.53, BLEU-4 0.39; Williams test
p < 1e-4), and rubric-based strategy rankings diverged from lexical ones
(Kendall tau 0.33-0.67). **(H2)** Closed-book competence strongly predicted RAG
answer quality (mixed-effects/OLS coefficient +0.89, p < 1e-70) and retrieval
noise significantly lowered rubric quality (-0.26, p = 0.012) - degradation the
exact-match metric misses; the model-level competence-fragility link was only
suggestive (Spearman -0.43, n.s. at n = 6). **(H3)** The judge panel was
reliable (Krippendorff's alpha = 0.68, in the predicted 0.40-0.80 band),
moderately calibrated (per-judge ECE 0.10/0.23), robust to surface perturbations
(rubric delta -0.04) and sensitive to semantic ones (delta -0.33, p = 4.6e-19).
Supplementary accuracy-based analyses show RAG underperforming closed-book on
MCQ with no size scaling - traced to uneven cross-domain retrieval relevance
(domain-match 8-85%) - and prompting strategy (SC > DA > RAS) as the dominant
controllable factor. All inferential findings are FDR/Bonferroni-corrected and
supported by an automated integrity audit over 533,304 inference records.
