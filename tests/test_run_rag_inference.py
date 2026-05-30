"""Tests for RAG inference runner (scripts/run_rag_inference.py).

All tests use MockLLM + MockRetriever - no GPU, no corpus indices required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_inference import (  # noqa: E402
    load_config,
)

from run_rag_inference import (  # noqa: E402
    get_rag_output_path,
    make_retriever,
    resolve_split_path,
    run_rag_cell,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
DEV_DATA = PROJECT_ROOT / "data" / "sciknoweval" / "dev.json"


# =========================================================================
# make_retriever: split -> index set
# =========================================================================


class TestMakeRetriever:
    """Retriever must point at QASPER indices for track_b, main otherwise."""

    @pytest.fixture(autouse=True)
    def _require_faiss(self) -> None:
        # retriever module imports faiss at module load; only present on the
        # GPU server. Skip locally where faiss is not installed.
        pytest.importorskip("faiss")

    @staticmethod
    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    def test_track_b_uses_qasper_indices(self) -> None:
        r = make_retriever("track_b", device="cpu")
        assert self._norm(r._bm25_index_dir).endswith("indices/qasper_bm25")
        assert self._norm(r._faiss_index_path).endswith(
            "indices/qasper_faiss/index.faiss"
        )
        assert self._norm(r._faiss_id_map_path).endswith(
            "indices/qasper_faiss/id_map.json"
        )

    def test_main_test_uses_default_indices(self) -> None:
        r = make_retriever("main_test", device="cpu")
        assert self._norm(r._bm25_index_dir).endswith("indices/bm25")
        assert "qasper" not in self._norm(r._faiss_index_path)


# =========================================================================
# MockRetriever
# =========================================================================


class MockRetriever:
    """Deterministic mock for Retriever, returns canned passages.

    Provides the same interface as ``scripts.retriever.Retriever`` but
    returns pre-built passages without any index/model loading.
    """

    def __init__(self, passages: list[dict] | None = None, top_k: int = 10) -> None:
        self._passages = passages or [
            {
                "chunk_id": f"chunk_{i:04d}",
                "text": f"Mock passage {i} about science and research.",
                "score": round(1.0 - i * 0.05, 4),
                "rank": i + 1,
            }
            for i in range(top_k)
        ]
        self.call_count = 0

    def _retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        self.call_count += 1
        return self._passages[:top_k]

    def retrieve_bm25(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 retrieval mock."""
        return self._retrieve(query, top_k)

    def retrieve_dense(self, query: str, top_k: int = 10) -> list[dict]:
        """Dense retrieval mock."""
        return self._retrieve(query, top_k)

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> list[dict]:
        """Hybrid retrieval mock."""
        return self._retrieve(query, top_k)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def noise_pools(tmp_path: Path) -> dict[str, str]:
    """Create minimal JSONL pool files for testing noise assembly.

    Returns:
        Dict mapping pool name to file path string.
    """
    pools_dir = tmp_path / "corpus" / "noise"
    pools_dir.mkdir(parents=True)

    pool_files: dict[str, str] = {}
    for pool_name, id_prefix in [
        ("irrelevant", "irr"),
        ("injection", "inj"),
        ("contradictory", "con"),
    ]:
        path = pools_dir / f"{pool_name}_passages.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(20):
                rec = {
                    "chunk_id"
                    if pool_name == "irrelevant"
                    else "noise_id": f"{id_prefix}_{i:03d}",
                    "text": f"{pool_name} passage {i} with scientific content.",
                    "domain": ["biology", "chemistry", "physics", "materials_science"][
                        i % 4
                    ]
                    if pool_name == "irrelevant"
                    else pool_name,
                }
                if pool_name == "injection":
                    rec["template_id"] = i % 3
                f.write(json.dumps(rec) + "\n")
        pool_files[pool_name] = str(path)

    return pool_files


@pytest.fixture
def rag_config() -> dict:
    """Load real rag.yaml config."""
    return load_config(RAG_CONFIG_PATH)


# =========================================================================
# TestRagConfig
# =========================================================================


class TestRagConfig:
    """Verify rag.yaml has expected structure."""

    def test_has_six_models(self, rag_config: dict) -> None:
        assert len(rag_config["models"]) == 6

    def test_has_noise_levels(self, rag_config: dict) -> None:
        levels = rag_config["noise"]["levels"]
        assert levels == [0.0, 0.2, 0.4, 0.6]

    def test_has_three_retrievers(self, rag_config: dict) -> None:
        assert rag_config["retrievers"] == ["bm25", "dense", "hybrid"]

    def test_has_four_strategies(self, rag_config: dict) -> None:
        assert set(rag_config["strategies"]) == {"da", "ras", "ctl", "sc"}

    def test_has_top_k_passages(self, rag_config: dict) -> None:
        assert rag_config["inference"]["top_k_passages"] == 10

    def test_has_checkpoint_every(self, rag_config: dict) -> None:
        assert rag_config["inference"]["checkpoint_every"] == 500


# =========================================================================
# TestRagOutputPath
# =========================================================================


class TestRagOutputPath:
    """Tests for RAG output path construction."""

    def test_path_format_bm25_noise0(self) -> None:
        path = get_rag_output_path(
            Path("outputs/rag_main"),
            "llama-3.2-3b",
            "bm25",
            0.0,
            "da",
        )
        assert path == Path("outputs/rag_main/llama-3.2-3b/bm25_noise0.0_da.jsonl")

    def test_path_format_hybrid_noise04(self) -> None:
        path = get_rag_output_path(
            Path("outputs/rag_main"),
            "qwen2.5-7b",
            "hybrid",
            0.4,
            "ras",
        )
        assert path == Path("outputs/rag_main/qwen2.5-7b/hybrid_noise0.4_ras.jsonl")

    def test_path_format_dense_noise06_sc(self) -> None:
        path = get_rag_output_path(
            Path("out"),
            "model-x",
            "dense",
            0.6,
            "sc",
        )
        assert path == Path("out/model-x/dense_noise0.6_sc.jsonl")


# =========================================================================
# TestResolveSplitPath
# =========================================================================


class TestResolveSplitPath:
    """Tests for split path resolution from RAG config."""

    def test_dev_split(self, rag_config: dict) -> None:
        path = resolve_split_path(rag_config, "dev")
        assert path == PROJECT_ROOT / "data" / "sciknoweval" / "dev.json"

    def test_main_test_split(self, rag_config: dict) -> None:
        path = resolve_split_path(rag_config, "main_test")
        assert path == PROJECT_ROOT / "data" / "sciknoweval" / "main_test_sampled.json"

    def test_unknown_split_raises(self, rag_config: dict) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            resolve_split_path(rag_config, "nonexistent")


# =========================================================================
# TestRunRagCell
# =========================================================================


class TestRunRagCell:
    """End-to-end RAG inference tests with MockLLM + MockRetriever."""

    def test_mock_rag_da_produces_correct_records(
        self,
        rag_config: dict,
        noise_pools: dict[str, str],
    ) -> None:
        """5 records with MockLLM + MockRetriever, noise=0.0."""
        mock_retriever = MockRetriever()

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_config["output"]["base_dir"] = tmpdir
            summary = run_rag_cell(
                model_name="MOCK",
                strategy="da",
                retriever_mode="bm25",
                noise_level=0.0,
                config=rag_config,
                split="dev",
                limit=5,
                mock=True,
                retriever=mock_retriever,
                noise_pool_paths=noise_pools,
            )

            assert summary["status"] == "complete"
            assert summary["processed"] == 5

            # Verify retriever was called for each record
            assert mock_retriever.call_count == 5

            # Read output and verify RAG-specific fields
            output_path = get_rag_output_path(
                Path(tmpdir),
                "MOCK",
                "bm25",
                0.0,
                "da",
            )
            with open(output_path) as f:
                records = [json.loads(line) for line in f]

            assert len(records) == 5
            for rec in records:
                assert "passages_used" in rec
                assert "retriever" in rec
                assert "noise_level" in rec
                assert rec["retriever"] == "bm25"
                assert rec["noise_level"] == 0.0
                assert isinstance(rec["passages_used"], list)
                assert len(rec["passages_used"]) > 0
                # At noise=0.0 all passages should be real
                for p in rec["passages_used"]:
                    assert p["noise_type"] == "real"
                    assert "chunk_id" in p

    def test_mock_rag_sc_produces_sc_result(
        self,
        rag_config: dict,
        noise_pools: dict[str, str],
    ) -> None:
        """SC strategy works in RAG mode."""
        mock_retriever = MockRetriever()

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_config["output"]["base_dir"] = tmpdir
            summary = run_rag_cell(
                model_name="MOCK",
                strategy="sc",
                retriever_mode="hybrid",
                noise_level=0.0,
                config=rag_config,
                split="dev",
                limit=3,
                mock=True,
                retriever=mock_retriever,
                noise_pool_paths=noise_pools,
            )

            assert summary["status"] == "complete"
            assert summary["processed"] == 3

            output_path = get_rag_output_path(
                Path(tmpdir),
                "MOCK",
                "hybrid",
                0.0,
                "sc",
            )
            with open(output_path) as f:
                records = [json.loads(line) for line in f]

            assert len(records) == 3
            for rec in records:
                assert rec["sc_result"] is not None
                assert rec["sc_result"]["total_samples"] == 5
                assert rec["retriever"] == "hybrid"

    def test_resume_skips_processed_records(
        self,
        rag_config: dict,
        noise_pools: dict[str, str],
    ) -> None:
        """Second run with same params returns status='skipped'."""
        mock_retriever = MockRetriever()

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_config["output"]["base_dir"] = tmpdir

            # First run
            run_rag_cell(
                model_name="MOCK",
                strategy="da",
                retriever_mode="dense",
                noise_level=0.0,
                config=rag_config,
                split="dev",
                limit=5,
                mock=True,
                retriever=mock_retriever,
                noise_pool_paths=noise_pools,
            )

            # Second run - same params, should skip
            mock_retriever_2 = MockRetriever()
            summary = run_rag_cell(
                model_name="MOCK",
                strategy="da",
                retriever_mode="dense",
                noise_level=0.0,
                config=rag_config,
                split="dev",
                limit=5,
                mock=True,
                retriever=mock_retriever_2,
                noise_pool_paths=noise_pools,
            )

            assert summary["status"] == "skipped"
            assert summary["processed"] == 0
            assert summary["skipped"] == 5

    def test_citations_extracted_in_rag_mode(
        self,
        rag_config: dict,
        noise_pools: dict[str, str],
    ) -> None:
        """Parsed output has 'citations' key in RAG mode."""
        mock_retriever = MockRetriever()

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_config["output"]["base_dir"] = tmpdir
            run_rag_cell(
                model_name="MOCK",
                strategy="da",
                retriever_mode="bm25",
                noise_level=0.0,
                config=rag_config,
                split="dev",
                limit=5,
                mock=True,
                retriever=mock_retriever,
                noise_pool_paths=noise_pools,
            )

            output_path = get_rag_output_path(
                Path(tmpdir),
                "MOCK",
                "bm25",
                0.0,
                "da",
            )
            with open(output_path) as f:
                records = [json.loads(line) for line in f]

        for rec in records:
            assert "citations" in rec["parsed"]

    def test_noisy_context_has_noise_passages(
        self,
        rag_config: dict,
        noise_pools: dict[str, str],
    ) -> None:
        """At noise_level=0.4, some passages should have noise_type != 'real'."""
        mock_retriever = MockRetriever()

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_config["output"]["base_dir"] = tmpdir
            run_rag_cell(
                model_name="MOCK",
                strategy="da",
                retriever_mode="hybrid",
                noise_level=0.4,
                config=rag_config,
                split="dev",
                limit=5,
                mock=True,
                retriever=mock_retriever,
                noise_pool_paths=noise_pools,
            )

            output_path = get_rag_output_path(
                Path(tmpdir),
                "MOCK",
                "hybrid",
                0.4,
                "da",
            )
            with open(output_path) as f:
                records = [json.loads(line) for line in f]

        assert len(records) == 5
        for rec in records:
            passages = rec["passages_used"]
            noise_types = Counter(p["noise_type"] for p in passages)
            # At 0.4 noise level, 4 out of 10 passages replaced
            assert noise_types.get("real", 0) <= 10
            # Must have at least some noise passages
            non_real = sum(v for k, v in noise_types.items() if k != "real")
            assert non_real >= 1, f"Expected noise passages but got: {noise_types}"


# =========================================================================
# Error path tests
# =========================================================================


class TestRunRagCellErrors:
    """Error-path tests for run_rag_cell."""

    def test_invalid_strategy_raises(self, rag_config: dict) -> None:
        with pytest.raises(ValueError, match="Invalid strategy"):
            run_rag_cell(
                model_name="MOCK",
                strategy="bogus",
                retriever_mode="bm25",
                noise_level=0.0,
                config=rag_config,
                mock=True,
                retriever=MockRetriever(),
            )

    def test_invalid_retriever_raises(self, rag_config: dict) -> None:
        with pytest.raises(ValueError, match="Invalid retriever"):
            run_rag_cell(
                model_name="MOCK",
                strategy="da",
                retriever_mode="bogus",
                noise_level=0.0,
                config=rag_config,
                mock=True,
                retriever=MockRetriever(),
            )
