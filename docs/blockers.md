# Blockers - Missing Scripts Specification

Каждый скрипт ниже **блокирует одну или несколько фаз** на сервере (см. `docs/server_runbook.md`).
Документ - чёткое ТЗ: что писать, какой интерфейс, какой выход, какие тесты, какие зависимости.
Реализовывать локально на CPU/dev-set, тестировать на 50-100 вопросов, пушить в `origin/dev`,
на сервере перед фазой - `git pull`.

Порядок приоритета: **F-блокеры → H-блокеры → G-блокеры → I-блокеры**.

---

## 1. `scripts/run_rag_inference.py` (Phase F блокер, P0)

**Цель**: запустить одну ячейку RAG-матрицы (model × strategy × retriever × noise_level) на split-вопросах.

**Шаблон**: см. `scripts/run_inference.py` (closed-book версия) - 90% логики переиспользуется. Отличия только в построении prompt'а: вместо question-only → question + retrieved_passages + (optional) noise_passages.

### CLI

```bash
python scripts/run_rag_inference.py \
  --config configs/rag.yaml \
  --model qwen2.5-7b \
  --strategy ras \
  --retriever hybrid \
  --noise-level 0.4 \
  --split data/sciknoweval/main_test_sampled.json \
  --output outputs/rag_main/qwen2.5-7b/hybrid_noise0.4_ras.jsonl
```

### Интерфейс с существующим кодом

```python
from retriever import Retriever            # scripts/retriever.py - класс Retriever
from noise_assembler import NoiseAssembler  # scripts/noise_assembler.py
from prompt_builder import build_rag_prompt # scripts/prompt_builder.py - проверить, есть ли rag-вариант
from response_parser import parse_response  # scripts/response_parser.py
from run_inference import (
    load_dataset, count_existing_records, get_output_path,
    release_engine, save_record,
)
```

### Логика по шагам (псевдокод)

```python
def run_rag_cell(model_cfg, strategy, retriever_mode, noise_level, split, output_path, top_k=10):
    retriever = Retriever(
        bm25_index='indices/bm25',
        faiss_index='indices/faiss/index.faiss',
        faiss_id_map='indices/faiss/id_map.json',
        embedding_model='models/bge-base-en-v1.5',
        chunks_file='corpus/all_chunks.jsonl',
    )
    assembler = NoiseAssembler.from_config('configs/noise.yaml', level=noise_level)
    llm = LLM(**model_cfg)  # vLLM init

    for question in load_dataset(split, resume_from=count_existing_records(output_path)):
        # 1. retrieve clean passages
        if retriever_mode == 'bm25':
            clean = retriever.retrieve_bm25(question['question'], top_k=top_k)
        elif retriever_mode == 'dense':
            clean = retriever.retrieve_dense(question['question'], top_k=top_k)
        elif retriever_mode == 'hybrid':
            clean = retriever.retrieve_hybrid(question['question'], top_k=top_k)

        # 2. assemble passages with noise replacement (uses per-question seed for reproducibility)
        passages = assembler.assemble(
            question_id=question['details']['id'],  # or whatever the unique key is
            clean_passages=clean,
        )

        # 3. build prompt
        prompt = build_rag_prompt(question, passages, strategy=strategy)

        # 4. inference (greedy or SC=5)
        if strategy == 'sc':
            samples = llm.generate(prompt, sampling_params_sc)  # 5 samples
            answer = majority_vote([parse_response(s.outputs[0].text) for s in samples])
        else:
            out = llm.generate(prompt, sampling_params_greedy)
            answer = parse_response(out[0].outputs[0].text)

        # 5. save (resumable, atomic write)
        save_record(output_path, {
            'qid': question['details']['id'],
            'question': question['question'],
            'gold': question['answer'],
            'predicted': answer,
            'passages_used': [p['chunk_id'] for p in passages],
            'retriever': retriever_mode,
            'noise_level': noise_level,
            'strategy': strategy,
            'model': model_cfg['name'],
        })
    release_engine(llm)
```

### Что обязательно

- **Resumable**: при повторном запуске пропускать уже обработанные `qid`. Используй `count_existing_records` из `run_inference.py`.
- **Atomic write**: каждый record - одна строка `.jsonl`, запись через append + flush. Не накапливать в памяти.
- **Per-question reproducibility**: noise positions определяются seed=(global_seed + hash(qid)). Уже реализовано в `noise_assembler.py`, не переписывать.
- **Provenance**: в каждый record писать `passages_used` (chunk_id'ы) - это нужно для citation precision/recall в Phase H.
- **Phase 6 gate**: при старте проверить `outputs/noise_review/contradictory_review_stats.json` - если `acceptance_rate < 0.80`, exit с ошибкой. `scripts/rag_gate.py` уже умеет это делать, импортировать и вызвать.

### Тест: `tests/test_run_rag_inference.py`

- На dev-set (50 вопросов), 1 модель (llama-3.2-3b), 1 retriever (bm25), 1 noise (0.0), 1 strategy (da).
- Запуск ~30 сек на CPU при mock-LLM или ~3 мин на GPU.
- Проверить: 50 records в output, все с `passages_used` непустыми, predicted/gold непустые.

---

## 2. `scripts/run_rag_experiment.py` (Phase F блокер, P0)

**Цель**: orchestrator поверх `run_rag_inference.py` - проходит по матрице ячеек последовательно.

**Шаблон**: см. `scripts/run_experiment.py` (closed-book orchestrator). Отличие только в построении матрицы (4D вместо 2D).

### CLI

```bash
# Полная матрица 6 × 4 × 3 × 4 = 288
python scripts/run_rag_experiment.py --config configs/rag.yaml --matrix full

# Дробная (рекомендуемая по бюджету): Hybrid × 4noise + BM25/Dense × noise=0
python scripts/run_rag_experiment.py --config configs/rag.yaml --matrix fractional

# Subset (для отладки)
python scripts/run_rag_experiment.py --config configs/rag.yaml \
  --models qwen2.5-7b --strategies da --retrievers hybrid --noise-levels 0.0
```

### Логика построения матрицы

```python
def build_matrix(config: dict, mode: str = 'full') -> list[tuple[str, str, str, float]]:
    """Returns ordered list of (model, strategy, retriever, noise_level) tuples."""
    models = list(config['models'].keys())
    strategies = config['strategies']
    retrievers = config['retrievers']
    noise_levels = config['noise']['levels']

    if mode == 'full':
        return [(m, s, r, n) for m in models for s in strategies for r in retrievers for n in noise_levels]
    elif mode == 'fractional':
        # Hybrid × all 4 noise levels
        hybrid_cells = [(m, s, 'hybrid', n) for m in models for s in strategies for n in noise_levels]
        # BM25, Dense × noise=0.0 only (anchor for retriever comparison)
        anchor_cells = [(m, s, r, 0.0) for m in models for s in strategies for r in ['bm25', 'dense']]
        return hybrid_cells + anchor_cells
    else:
        raise ValueError(f'Unknown matrix mode: {mode}')
```

**Outer loop = model** (минимизирует число LLM-загрузок), inner = retriever × noise × strategy.

### Что обязательно

- **Resumable на уровне ячейки**: проверить `os.path.exists(output_path)` И `count_existing_records(output_path) >= len(split)`. Если да - skip.
- **Освобождать LLM между моделями**: `release_engine(llm); torch.cuda.empty_cache()`. Без этого OOM.
- **Логировать прогресс**: каждые 5 ячеек писать `[12/144] qwen2.5-7b × ras × hybrid × noise=0.4 done in 23min`.

### Тест: `tests/test_run_rag_experiment.py`

- Subset 1×1×1×1 (одна ячейка).
- Проверить корректность построения матрицы (full=288, fractional=144).
- Проверить resume: если ячейка уже на диске - skip.

---

## 3. `scripts/run_judge.py` (Phase H блокер, P1)

**Цель**: оценить outputs через rubric (0-5 шестибалльная шкала + faithfulness + citation precision/recall + coverage + self_confidence).

### CLI

```bash
# Open-weight judge на всех outputs
python scripts/run_judge.py \
  --judge judge_a \
  --target outputs/closed_book_main outputs/rag_main outputs/qasper_main

# API judge (judge_c) на bounded subset
python scripts/run_judge.py \
  --judge judge_c \
  --calibration-subset 1000 \
  --cost-cap-usd 100
```

### Логика

```python
def run_judge(judge_id: str, target_dirs: list[Path], config_path: str = 'configs/judge.yaml'):
    cfg = yaml.safe_load(open(config_path))
    judge_cfg = next(j for j in cfg['main_set_judges'] if j['id'] == judge_id) \
                if judge_id != 'judge_c' else cfg['calibration_judge']

    if judge_id in ('judge_a', 'judge_b'):
        llm = LLM(model=f"models/{judge_cfg['model']}", ...)  # vLLM
    else:
        llm = APIClient(...)  # Anthropic / OpenAI

    nli_model = CrossEncoder('models/nli-deberta-v3-large')  # для faithfulness

    for output_jsonl in iter_output_files(target_dirs):
        for record in read_jsonl(output_jsonl):
            scores = score_record(llm, nli_model, record, cfg['scoring'])
            # scores = {'rubric': 4, 'faithfulness': 0.92, 'citation_precision': 0.7,
            #           'citation_recall': 0.6, 'coverage': 0.85, 'self_confidence': 0.8}
            save_judge_output(record['qid'], judge_id, scores, output_jsonl)
```

### Файлы для создания

1. **`scripts/run_judge.py`** - сам скрипт.
2. **`configs/judge_prompts/rubric_template.txt`** - prompt для rubric scoring (6-point):
   - Input: question, gold answer, model answer, [optional retrieved passages].
   - Output: JSON `{"rubric": 0-5, "rationale": "..."}`.
3. **`configs/judge_prompts/faithfulness_template.txt`** - prompt для extracting atomic claims (для NLI).
4. **`configs/judge_prompts/citation_template.txt`** - prompt для citation precision/recall.

### Faithfulness через NLI

```python
# 1. Decompose model answer into atomic claims (LLM-задача).
claims = decompose_claims(llm, model_answer)
# 2. For each claim, run NLI(passage_i, claim) over retrieved passages.
# 3. faithfulness = fraction of claims with entailment >= 0.5 against ANY passage.
faithfulness = sum(
    any(nli_model.predict([(p['text'], c)])[0]['entailment'] >= 0.5 for p in passages)
    for c in claims
) / max(len(claims), 1)
```

### Citation precision/recall

```python
# precision = (cited passages that ARE relevant) / (all cited passages)
# recall = (cited passages that ARE relevant) / (all gold-relevant passages)
# В RAG outputs `passages_used` - это все retrieved (top-k=10), без явных citations.
# Поэтому precision/recall = NLI(answer, passage)-based:
#   relevant_passage = NLI(passage, answer).entailment >= 0.5
```

### Тест: `tests/test_run_judge.py`

- Mock LLM, mock NLI - проверить контракт scores (все ключи присутствуют, диапазоны [0, 5] и [0, 1]).
- Real judge_a на 5 outputs из dev (закрытая книга, известные правильные/неправильные ответы) - проверить что rubric=5 для правильных, rubric<=2 для бредовых.

---

## 4. `scripts/judge_aggregate.py` (Phase H блокер, P1)

**Цель**: вычислить inter-rater reliability и calibration metrics.

### Что считать

- **Krippendorff's α** для пары (judge_a, judge_b) на полном наборе rubric scores.
- **Krippendorff's α** для тройки (a, b, c) на calibration subset.
- **ECE (Expected Calibration Error)** на MCQ subset - confidence bins, 10 bins.
- **Wilcoxon signed-rank** для class 1 perturbations (typo/whitespace) - rubric до vs после возмущения, ожидание: разница незначима.
- **Wilcoxon** для class 2 perturbations (entity swap, negation) - ожидание: значимая разница.

### Реализация

```python
import krippendorff
import scipy.stats as stats

# Krippendorff's alpha
alpha_ab = krippendorff.alpha(reliability_data=ratings_ab_matrix, level_of_measurement='ordinal')

# ECE
def ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctness[mask].mean()
            ece += (mask.sum() / len(confidences)) * abs(avg_conf - avg_acc)
    return ece
```

### Output

`outputs/judge/aggregate_stats.json` с полями `krippendorff_ab`, `krippendorff_abc`, `ece_mcq`, `wilcoxon_class1`, `wilcoxon_class2`.

---

## 5. `scripts/build_qasper_corpus.py` (Phase G блокер, P2)

**Цель**: построить QASPER-specific corpus (paper-level, ~1585 статей) + BM25 + FAISS.

### CLI

```bash
python scripts/build_qasper_corpus.py
# Output: data/qasper/corpus.jsonl, data/qasper/papers.json,
#         indices/qasper_bm25/, indices/qasper_faiss/
```

### Логика

```python
from datasets import load_dataset

ds = load_dataset('allenai/qasper', split='train+validation+test')

# Each paper -> chunk into 256-token windows with 64-token overlap (same as main corpus)
# Chunk text = section_title + paragraph

chunks = []
for paper in ds:
    for section in paper['full_text']['sections']:
        for paragraph in section['paragraphs']:
            for chunk in chunk_text(paragraph, chunk_size=256, overlap=64):
                chunks.append({
                    'chunk_id': hash_chunk(paper['id'], section['title'], chunk),
                    'paper_id': paper['id'],
                    'section': section['title'],
                    'text': chunk,
                })

write_jsonl('data/qasper/corpus.jsonl', chunks)

# BM25 - переиспользовать scripts/build_indices.py с другим input
# FAISS - переиспользовать с другим input
```

**Переиспользовать `scripts/chunking.py` (есть!) и `scripts/build_indices.py` (есть!)** - им только новый input/output путь нужен.

### Тест: `tests/test_build_qasper_corpus.py`

- Загрузить QASPER, проверить что в corpus.jsonl записано >= 50000 чанков.
- Проверить что FAISS-индекс открывается, ntotal == len(corpus.jsonl).

---

## 6. `scripts/sample_qasper.py` (Phase G блокер, P2)

**Цель**: стратифицированный сэмплинг 600 вопросов из QASPER train+validation+test.

**Шаблон**: см. `scripts/sample_main_test.py` - почти идентично, только страты другие (по domain papers, по answer_type extractive/abstractive/yesno).

### CLI

```bash
python scripts/sample_qasper.py --target 600 --seed 42
# Output: data/qasper/sample.json, outputs/qasper_sampled_stratification.json
```

---

## 7. `scripts/run_statistics.py` (Phase I блокер, P3)

**Цель**: финальные таблицы и тесты гипотез H1-H3 для главы 5 диплома.

### CLI

```bash
python scripts/run_statistics.py --hypothesis H1 H2 H3
# Output: outputs/chapter5_tables/{H1,H2,H3}.csv + .tex
#         outputs/chapter5_tables/effect_sizes.csv (Cliff's δ + Cohen's d)
#         outputs/chapter5_tables/p_values_corrected.csv (Benjamini-Hochberg FDR 0.05)
```

### H1: closed-book vs RAG

- **point-biserial correlation** между (model_size, RAG_improvement_over_closed_book).
- **Steiger's z-test** для сравнения корреляций.
- **Bonferroni correction**.
- **Kendall's τ** для ранжирования моделей.

### H2: noise robustness

- **Mixed-effects regression**: `score ~ closed_book_baseline + noise_level + strategy + (1|model)`.
- Coefficient на `noise_level` - sensitivity to noise.

### H3: judge reliability

- **Wilcoxon** на perturbation pairs.
- **Krippendorff's α** между judges.
- **ECE** для self-confidence.

### Effect sizes

- **Cliff's δ** для каждой пары моделей.
- **Cohen's d** для каждой стратегии vs baseline.

### Зависимости

```python
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import scipy.stats as stats
import krippendorff
```

Возможно понадобится `pip install statsmodels krippendorff`. Добавить в `requirements-server.txt` если нет.

---

## 8. Что НЕ блокирует, но улучшит pipeline

### `scripts/cleanup_outputs.py` (помощник)

- Удаляет SC raw-сэмплы после агрегации (оставляет только majority vote).
- Удаляет `_raw.jsonl` после `compute_metrics.py`.
- Чистит `outputs/logs/*.log` старше 7 дней.
- Полезно когда диск приближается к 90%.

### `scripts/disk_audit.py`

- Печатает размер каждой подпапки `models/`, `outputs/`, `corpus/`, `indices/`.
- Выявляет лишние форматы весов в моделях (как nli-deberta с 5.8 GB onnx).

---

## 9. Workflow реализации

Для каждого блокера:

1. **Локально на Windows**: открыть `D:\Projects\Domain-Aware-...-Evaluation\`.
2. Написать скрипт + тест.
3. Запустить тест: `pytest tests/test_<name>.py -xvs`.
4. Smoke-test на dev-set локально (если возможно без GPU - mock LLM, иначе пропустить).
5. Закоммитить **отдельно каждый блокер**:
   ```bash
   git add scripts/run_rag_inference.py tests/test_run_rag_inference.py
   git commit -m "feat: rag inference runner for phase F"
   ```
6. После всех блокеров одной фазы - `git push origin dev`.
7. На сервере - `git pull origin dev` перед фазой.

---

## 10. Сводка приоритетов

| # | Скрипт | Фаза | Приоритет | Часов на реализацию |
|---|--------|------|-----------|---------------------|
| 1 | run_rag_inference.py | F | P0 | 4-6 ч |
| 2 | run_rag_experiment.py | F | P0 | 1-2 ч |
| 3 | run_judge.py + prompts | H | P1 | 6-10 ч |
| 4 | judge_aggregate.py | H | P1 | 2-3 ч |
| 5 | build_qasper_corpus.py | G | P2 | 2-3 ч |
| 6 | sample_qasper.py | G | P2 | 1 ч |
| 7 | run_statistics.py | I | P3 | 4-6 ч |

**Итого**: ~20-30 часов локальной разработки до того, как сервер сможет дойти до Phase I.

**P0/P1 (Phase F + H)** - первые две недели, пока сервер крутит D + B + E. Параллельно.
**P2/P3 (Phase G + I)** - после того как Phase F запустился (можно начать после P1).
