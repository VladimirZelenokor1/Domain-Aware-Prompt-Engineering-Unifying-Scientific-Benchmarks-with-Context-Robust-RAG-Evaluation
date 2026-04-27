# Server Runbook - Thesis Experiments on innodatahub

**Цель**: продолжить экспериментальный pipeline на сервере университета (Innopolis innodatahub) с любого ПК. Документ самодостаточный - содержит все команды, пути, замечания и план фаз до защиты диплома.

**Сервер**: `innodatahub`, JupyterLab контейнер, A100 80GB MIG 3g.40gb (40 GB VRAM выделено), 6 CPU, 32 GB RAM, 100 GB SSD.

**HF account**: тот же что локально (Llama/Gemma/Llama-3.1 уже акцептированы).

**Бюджет**: 1800 баллов/неделю = ~40 часов GPU/неделю.

---

## 0. Доступ к серверу с нового ПК

1. Открой https://innodatahub.innopolis.university (или адрес где у тебя Jupyter).
2. Логин под своим аккаунтом.
3. Создай бронирование: 1 GPU MIG 3g.40gb, 6 CPU, 32 GB RAM. Выбирай длительность по плану ниже.
4. Дождись запуска контейнера, открой JupyterLab.
5. **File → New → Terminal**.

**Persistent_volume переживает все бронирования.** venv, JDK, веса моделей, corpus, indices, репо - всё уже на месте в `~/persistent_volume/thesis/`.

---

## 1. Каждый раз в начале сессии

```bash
cd ~/persistent_volume/thesis/Domain-Aware-Prompt-Engineering-Unifying-Scientific-Benchmarks-with-Context-Robust-RAG-Evaluation
source ~/persistent_volume/thesis/venv-thesis/bin/activate
nvidia-smi
df -h ~/persistent_volume

# Stack smoke-test (10 секунд)
python -c "
import torch, faiss
from pyserini.search.lucene import LuceneSearcher
from vllm import LLM
print('stack OK | torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))
"

# Sanity check артефактов
python -c "
import faiss
idx = faiss.read_index('indices/faiss/index.faiss')
print(f'FAISS ntotal = {idx.ntotal} (expect 3939826)')
"
ls models/ | wc -l   # должно быть 11
```

Если что-то падает - см. секцию **Troubleshooting** в конце.

---

## 2. Состояние на конец последней сессии (2026-04-27)

### Что готово на сервере

- venv-thesis (7.2 GB): torch 2.4 cu121, vLLM 0.6.3, transformers 4.45.2, sentence-transformers 3.2.1, faiss-cpu, pyserini 0.43.0, autoawq 0.2.7, elasticsearch 8.15.1.
- jdk-env (555 MB): OpenJDK 21 для pyserini.
- 11 моделей в `models/` (45 GB):
  - **6 examinee LLMs**: llama-3.2-3b, qwen2.5-7b, sciphi-mistral-7b, deepseek-r1-qwen-7b, gemma-2-9b, mistral-nemo-12b (все AWQ-INT4)
  - **3 auxiliary**: bge-base-en-v1.5, bge-reranker-v2-m3, nli-deberta-v3-large
  - **2 judges**: llama-3.1-8b-judge-awq (hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4), mistral-7b-v0.3-judge-awq
- `corpus/all_chunks.jsonl` (4.4 GB, 3,939,826 чанков), `corpus/manifest.json`, `corpus/noise/{irrelevant_distractors.jsonl, injection_passages.jsonl}` (25K + 3K).
- `indices/bm25/` (5.1 GB, 3,939,825 docs), `indices/faiss/index.faiss` (218 MB, 3,939,826 vectors, dim 768).
- `data/sciknoweval/{dev.json, main_test.json, holdout.json}` (25,353 + 5,070 + 264 items).
- Репо склонирован, branch `dev`, commits c75a84c..5031b74 (12 свежих).

### Что отсутствует (генерируется на сервере по плану ниже)

- `data/sciknoweval/main_test_retrievable.json` (Phase B)
- `data/sciknoweval/main_test_sampled.json` (Phase C)
- `corpus/noise/contradictory_passages.jsonl` (Phase E)
- `outputs/closed_book_main/` (Phase D)
- `outputs/rag_main/` (Phase F)
- `data/qasper/` + индексы QASPER (Phase G)
- `outputs/judge/` (Phase H)
- `outputs/chapter5_tables/` (Phase I)

### Известные блокеры (надо починить до Phase F и H)

1. **`scripts/retriever.py`**: класс называется `Retriever` (не `HybridRetriever`), интерфейс - методы `retrieve_bm25(query, top_k)`, `retrieve_dense(query, top_k)`, `retrieve_hybrid(...)`. Использовать его, не выдумывать новый класс.
2. **`scripts/run_rag_inference.py`** и **`scripts/run_rag_experiment.py`**: НЕТ. Нужно написать (по аналогии с `run_inference.py` + `run_experiment.py`, добавив injection retrieval-контекста + noise composition). Блокер Phase F.
3. **`scripts/run_judge.py`** + `configs/judge_prompts/`: НЕТ. Блокер Phase H.
4. **`scripts/build_qasper_corpus.py`** + `scripts/sample_qasper.py`: НЕТ. Блокер Phase G.

**Стратегия**: писать эти скрипты локально на CPU, тестировать на dev-set (50-100 вопросов), пушить в origin/dev, на сервере делать `git pull` перед фазой.

### Бюджет

| Использовано в первой сессии | ~135 баллов (3 ч setup) |
| Осталось на неделе | ~1665 баллов = ~37 ч GPU |
| Полный план A-I | ~150-210 GPU-часов = 4-5 недель |

---

## 3. Phase B - Retrievability Filter (GPU, ~6-8 ч)

**Цель**: отфильтровать вопросы из main_test, для которых хотя бы один gold-passage не достаётся top-100 BM25 поиском. Без этого RAG-эксперименты будут на вопросах с гарантированным провалом retrieval.

### Перед запуском

```bash
# 1. Pull последних изменений (если на локалке что-то правил)
git fetch origin && git pull origin dev

# 2. Проверь что скрипт на месте
ls scripts/retrievability_filter.py
python scripts/retrievability_filter.py --help

# 3. Cache-каталог можно очистить (на сервере с нуля посчитается быстрее)
rm -rf data/sciknoweval/.retrievability_cache/
```

### Запуск в фоне через nohup

```bash
nohup python scripts/retrievability_filter.py \
  --split data/sciknoweval/main_test.json \
  --force \
  --flush-every 100 \
  > outputs/logs/retrievability.log 2>&1 &
disown
echo "PID: $!"

# Проверить через 30 сек
sleep 30
ps -ef | grep retrievability_filter | grep -v grep
tail -20 outputs/logs/retrievability.log
```

Можно закрыть вкладку - nohup переживает logout. Возвращайся через 6-8 часов.

### Проверка завершения

```bash
ls -la data/sciknoweval/main_test_retrievable.json
cat data/sciknoweval/.retrievability_cache/main_test/stats.json | python -m json.tool
```

Ожидаемо: `retrievable: ~7000-7500` из `total: 25353` (на локалке было 7207). Это **итоговый набор для Phase C/D/F**.

**Ориентировочно ~6-8 часов = 270-360 баллов.**

---

## 4. Phase C - Stratified Sampling (CPU, секунды)

**Цель**: из ~7200 retrievable вопросов выбрать 3000 со стратификацией по domain × level × type.

```bash
python scripts/sample_main_test.py --target 3000 --n-floor 3 --seed 42

# Проверка
ls -la data/sciknoweval/main_test_sampled.json
ls -la outputs/main_test_sampled_stratification.json
python -c "
import json
with open('data/sciknoweval/main_test_sampled.json') as f:
    d = json.load(f)
print(f'sampled = {len(d)} (expected 3000)')
"
```

**0 GPU-часов.**

---

## 5. Phase D - Closed-Book Full Matrix 6×4 (GPU, ~25-35 ч)

**Цель**: запустить все 6 моделей × 4 стратегии (DA/RAS/CTL/SC) на 3000 sampled-вопросах. SC = 5 сэмплов при temperature=0.7, остальные greedy.

### Сначала обновить configs/closed_book.yaml

Раскомментировать DeepSeek (на A100 40 GB он уже не ограничивает max_tokens) и добавить Gemma + Mistral-Nemo. На локалке открой `configs/closed_book.yaml` и приведи к виду:

```yaml
models:
  llama-3.2-3b:
    path: models/llama-3.2-3b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  qwen2.5-7b:
    path: models/qwen2.5-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  sciphi-mistral-7b:
    path: models/sciphi-mistral-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  deepseek-r1-qwen-7b:
    path: models/deepseek-r1-qwen-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  gemma-2-9b:
    path: models/gemma-2-9b-awq
    quantization: awq_marlin
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  mistral-nemo-12b:
    path: models/mistral-nemo-12b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16

data:
  dev: data/sciknoweval/dev.json
  main_test: data/sciknoweval/main_test_sampled.json   # ВАЖНО: sampled, не main_test
```

Закоммить и запушь:

```powershell
# Локально
git add configs/closed_book.yaml
git commit -m "feat: enable all 6 models for closed-book on A100 40GB"
git push origin dev
```

Также обнови `models/MODEL_REGISTRY.json` локально - сними `excluded` с DeepSeek, добавь записи для Gemma и Mistral-Nemo (можно скопировать структуру из существующих записей; точные `disk_gb`, `test_status` указать после реального теста).

### Запуск на сервере

```bash
git pull origin dev
mkdir -p outputs/logs outputs/closed_book_main

# Сначала smoke (без GPU, проверить матрицу)
python scripts/run_experiment.py --config configs/closed_book.yaml --dry-run

# Полный прогон в фоне
nohup python scripts/run_experiment.py \
  --config configs/closed_book.yaml \
  > outputs/logs/closed_book_main.log 2>&1 &
disown
echo "PID: $!"

# Проверка прогресса
tail -f outputs/logs/closed_book_main.log
```

**Длительность**: 24 ячейки × ~70-90 мин на ячейку = ~25-35 ч (Mistral-Nemo-12B самая медленная).

### Метрики после завершения

```bash
python scripts/compute_metrics.py --base-dir outputs/closed_book_main
ls outputs/closed_book_main/*/summary.json
```

**Ориентировочно 25-35 часов = 1100-1600 баллов.**

---

## 6. Phase E - Contradictory Noise Pool (GPU, ~3-4 ч)

**Цель**: сгенерировать 3000 контрадикторных passages через Qwen2.5-7B (LLM-based perturbation), затем ручная проверка 50 примеров с порогом acceptance ≥ 0.80.

```bash
nohup python scripts/build_noise.py contradictory \
  --model qwen2.5-7b \
  --target 3000 \
  > outputs/logs/noise_contradictory.log 2>&1 &
disown

# Через 3-4 ч проверь
ls corpus/noise/contradictory_passages.jsonl
wc -l corpus/noise/contradictory_passages.jsonl

# Ручная проверка 50 примеров
python scripts/build_noise.py contradictory-review --sample 50
# Откроется CSV для разметки в outputs/noise_review/contradictory_review.csv
# Проставить вручную "accept" / "reject" в колонке verdict (через JupyterLab)

# Аггрегация
python scripts/build_noise.py contradictory-stats
cat outputs/noise_review/contradictory_review_stats.json
# Если acceptance_rate >= 0.80 -> Phase F unlocked
# Если < 0.80 -> refine prompt в build_noise.py, regenerate
```

**Ориентировочно 3-4 часа GPU + 1 ч ручной работы = ~150-200 баллов.**

---

## 7. Phase F - RAG Main Matrix (GPU, ~70-90 ч дробная / 150-200 ч полная)

### ⚠ Критический блокер до запуска

Этот этап **НЕ запустится пока нет**:
1. `scripts/run_rag_inference.py` (нужно написать локально, по аналогии с `run_inference.py` + добавить retrieval + noise injection)
2. `scripts/run_rag_experiment.py` (orchestrator поверх run_rag_inference)
3. Класс `HybridRetriever` или эквивалент в `scripts/retriever.py` (на последней проверке упал импорт)

**Что делать сейчас**: написать эти 2 скрипта локально, тестировать на dev-set (50 вопросов, 1 модель, 1 retriever, 1 noise level, ~5 минут на CPU/GPU локально), пушить в origin/dev. На сервере перед Phase F - `git pull`.

### Стратегия матрицы

**Полная факториальная**: 6 моделей × 4 промпта × 3 ретривера × 4 шума = **288 ячеек** ≈ 150-200 ч.

**Дробная (рекомендую под бюджет 1800/нед)**:
- Hybrid (canonical) × 4 шума × 6 моделей × 4 промпта = 96 ячеек
- + BM25, Dense × noise=0.0 × 6 × 4 = 48 ячеек (anchor для retriever-comparison)
- = **144 ячейки** ≈ 70-90 ч

### Запуск (после написания скриптов)

```bash
git pull origin dev
mkdir -p outputs/rag_main

# Smoke на dev-set
python scripts/run_rag_experiment.py --config configs/rag.yaml --dry-run

# Дробная матрица (рекомендуемая)
nohup python scripts/run_rag_experiment.py \
  --config configs/rag.yaml \
  --matrix fractional \
  > outputs/logs/rag_main.log 2>&1 &
disown
```

**Ориентировочно 70-90 часов = 3200-4000 баллов = ~2 недели по 1800.**

---

## 8. Phase G - Track B QASPER (GPU, ~15-25 ч)

### Блокеры до запуска

1. `scripts/build_qasper_corpus.py` (paper-level chunking 1585 статей → BM25 + FAISS).
2. `scripts/sample_qasper.py` (стратифицированный сэмплинг 600 вопросов).
3. `configs/qasper.yaml` (как configs/rag.yaml, но один retriever Hybrid, два уровня шума 0/0.6).

### Запуск

```bash
# Скачать QASPER
python scripts/build_qasper_corpus.py
# Output: data/qasper/corpus.jsonl, indices/qasper_bm25/, indices/qasper_faiss/

python scripts/sample_qasper.py --target 600 --seed 42
# Output: data/qasper/sample.json

# Прогон 6 × 4 × 1 × 2 = 48 ячеек
nohup python scripts/run_rag_experiment.py \
  --config configs/qasper.yaml \
  --matrix fractional \
  > outputs/logs/qasper_main.log 2>&1 &
disown
```

**Ориентировочно 15-25 часов = 700-1100 баллов.**

---

## 9. Phase H - LLM-Judge Panel (GPU + API, ~30-50 ч + $50-100 API)

### Блокеры

1. `scripts/run_judge.py`: загружает judge-LLM (vLLM), читает все output JSONL, оценивает по rubric.
2. `configs/judge_prompts/`: ru/eng prompt templates с 6-point rubric (faithfulness, citation precision/recall, coverage, self-confidence).
3. `scripts/judge_aggregate.py`: Krippendorff's α + ECE + perturbation Wilcoxon.
4. API key для judge_c (GPT-4o или Claude Sonnet).

### Запуск

```bash
# Judge A (open-weight Llama-3.1-8B) - на всех outputs
nohup python scripts/run_judge.py \
  --judge judge_a \
  --target outputs/closed_book_main outputs/rag_main outputs/qasper_main \
  > outputs/logs/judge_a.log 2>&1 &
disown
# ~15-25 ч

# Judge B (Mistral-7B-v0.3)
nohup python scripts/run_judge.py \
  --judge judge_b \
  --target outputs/closed_book_main outputs/rag_main outputs/qasper_main \
  > outputs/logs/judge_b.log 2>&1 &
disown
# ~15-25 ч (можно после A или последовательно)

# Judge C (proprietary API) на calibration subset 1000 items
export ANTHROPIC_API_KEY=sk-...   # или OPENAI_API_KEY
python scripts/run_judge.py --judge judge_c --calibration-subset 1000
# ~10-20 минут API time, $50-100

# Аггрегация
python scripts/judge_aggregate.py
```

**Ориентировочно 30-50 GPU-часов = 1300-2200 баллов + $50-100 на API.**

---

## 10. Phase I - Statistics & Tables (CPU, 2-4 ч)

```bash
mkdir -p outputs/chapter5_tables

python scripts/compute_metrics.py --all
python scripts/run_statistics.py --hypothesis H1 H2 H3
# H1: point-biserial, Steiger's z, Bonferroni, Kendall's τ
# H2: mixed-effects regression (closed-book × noise × strategy)
# H3: Wilcoxon, Krippendorff's α, ECE
# Benjamini-Hochberg FDR 0.05
# Cliff's δ + Cohen's d рядом с каждым p-value

ls outputs/chapter5_tables/*.csv outputs/chapter5_tables/*.tex
```

**0 GPU-часов.** Скрипт `run_statistics.py` тоже надо написать локально.

---

## 11. Сводная таблица фаз и бюджета

| Phase | Что делает | GPU ч | Баллов | Зависит | Блокеры скриптов |
|-------|------------|-------|--------|---------|------------------|
| Setup | venv + JDK + 11 моделей + corpus + indices | 0 | ~135 (готово) | — | ✅ сделано |
| B | retrievability_filter | 6-8 | 270-360 | A | ✅ есть |
| C | sampling (3000) | 0 | 0 | B | ✅ есть |
| D | closed-book 6×4 на sampled | 25-35 | 1100-1600 | C, configs update | ✅ есть |
| E | contradictory noise (3K) | 3-4 | 150-200 | C | ✅ есть |
| F (дробная) | RAG 144 ячеек | 70-90 | 3200-4000 | D, E | ❌ run_rag_*.py + retriever fix |
| G | QASPER 48 ячеек | 15-25 | 700-1100 | E, F-инфра | ❌ build_qasper_*.py |
| H | judges A/B/C + аггрегация | 30-50 | 1300-2200 + API | D, F, G | ❌ run_judge.py |
| I | статистика, таблицы | 0 | 0 | D, F, G, H | ❌ run_statistics.py |
| **Итого** | | **150-210** | **~6700-9500** | | |

При 1800 баллов/нед = 40 ч GPU/нед: **полный план = ~5 недель**.

---

## 12. Что писать локально перед сервером (порядок приоритета)

См. **detailed spec в `docs/blockers.md`** - каждый скрипт описан с интерфейсом, аргументами, ожидаемым выходом и тестами. Порядок приоритета:

1. **`scripts/run_rag_inference.py`** + **`scripts/run_rag_experiment.py`** - Phase F блокер.
2. **`scripts/run_judge.py`** + judge prompts - Phase H блокер.
3. **`scripts/build_qasper_corpus.py`** + **`scripts/sample_qasper.py`** - Phase G блокер.
4. **`scripts/run_statistics.py`** - Phase I блокер (можно последним).

Каждый - отдельный коммит, обязательно тесты в `tests/test_*.py`. Всё закидывать в `origin/dev`, на сервере `git pull`.

---

## 13. Дисковый бюджет (важно следить)

**Сейчас**: 73/100 GB занято, 28 GB свободно.

**Ожидаемое потребление по фазам**:

| После фазы | Что добавится | Свободно после |
|------------|----------------|----------------|
| B | main_test_retrievable.json (~5 MB) | 28 GB |
| C | main_test_sampled.json (~3 MB) + cache | 28 GB |
| D | outputs/closed_book_main 24 cells × ~50-100 MB = ~2 GB | 26 GB |
| E | contradictory_passages.jsonl (~20 MB) | 26 GB |
| F | outputs/rag_main 144 cells × ~150-300 MB = ~30-45 GB | **−5 до −20 GB ⚠** |
| G | QASPER corpus + indices (~3 GB) + outputs (~3 GB) | проблема |

**Стратегия дисковой экономии**:
- После Phase D → запустить `scripts/compute_metrics.py` → удалить SC-сэмплы (только итоговый ответ оставить, не 5 raw samples).
- После Phase F каждой ячейки → удалять `_raw.jsonl`, оставлять только `_aggregated.jsonl` + metrics.
- Чистить `outputs/logs/*.log` старше 7 дней.
- Если упрёшься в потолок - временно удалить `corpus/{wikipedia,pubmed,openstax}_chunks.jsonl` (4.5 GB, нужны были только при сборке all_chunks.jsonl, для retrieval не используются).

---

## 14. Troubleshooting

### `pyserini` падает с `Unable to find javac`

JAVA_HOME не подхватился. Проверь activate:

```bash
tail -6 ~/persistent_volume/thesis/venv-thesis/bin/activate
```

Должно быть:
```
export JAVA_HOME=$HOME/persistent_volume/thesis/jdk-env/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
```

Если нет - дописать через `cat >> ... <<'EOF'` (см. секцию 1).

### vLLM падает на загрузке Mistral-Nemo-12B (OOM)

На MIG 3g.40gb 40 GB VRAM хватает, но если вместе с reranker/embedder - может не хватить. Снизь `gpu_memory_utilization` до 0.75 в configs/closed_book.yaml для mistral-nemo-12b. Или временно убери BGE-reranker из RAM.

### `huggingface_hub` сломал transformers (как было в первой сессии)

```bash
pip install "huggingface_hub==0.26.2"
```

### nohup-процесс не виден в `ps`

Проверь PPID=1 (сирота от убитого родителя):

```bash
ps -ef | grep -E "python|huggingface" | grep -v grep
```

### Закончилось дисковое место

```bash
# Топ-10 каталогов по размеру в репо
du -h --max-depth=2 . 2>/dev/null | sort -hr | head -10

# Удалить логи > 7 дней
find outputs/logs -name "*.log" -mtime +7 -delete

# Удалить .cache в моделях
find models -type d -name ".cache" -exec rm -rf {} + 2>/dev/null
```

### Контейнер забыл JAVA_HOME / PATH после нового бронирования

`venv-thesis/bin/activate` пишется один раз в persistent_volume, переживает все бронирования. Если не подхватывается - реактивируй: `deactivate && source ~/persistent_volume/thesis/venv-thesis/bin/activate`.

---

## 15. Контакты и ссылки

- **Репо**: https://github.com/VladimirZelenokor1/Domain-Aware-Prompt-Engineering-Unifying-Scientific-Benchmarks-with-Context-Robust-RAG-Evaluation
- **Branch**: `dev`
- **HF account**: тот же что локально (Llama/Gemma/Llama-3.1 акцептированы).
- **Server**: innodatahub (Innopolis University), 1800 баллов/нед = 40 ч GPU/нед на A100 40GB MIG.

**Конец runbook.**
