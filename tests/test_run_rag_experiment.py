"""Tests for RAG experiment orchestrator (scripts/run_rag_experiment.py).

All tests are unit-level - no GPU, no corpus indices, no real LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_inference import load_config  # noqa: E402
from run_rag_experiment import build_rag_matrix  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def rag_config() -> dict:
    """Load real rag.yaml config."""
    return load_config(RAG_CONFIG_PATH)


# =========================================================================
# TestBuildRagMatrix
# =========================================================================


class TestBuildRagMatrix:
    """Tests for build_rag_matrix function."""

    def test_full_matrix_size(self, rag_config: dict) -> None:
        """Full mode: 6 models x 4 strategies x 3 retrievers x 4 noise = 288."""
        matrix = build_rag_matrix(rag_config, mode="full")
        assert len(matrix) == 288

    def test_fractional_matrix_size(self, rag_config: dict) -> None:
        """Fractional mode: hybrid x 4 noise (96) + bm25/dense x noise=0.0 (48) = 144."""
        matrix = build_rag_matrix(rag_config, mode="fractional")
        assert len(matrix) == 144

    def test_fractional_matrix_no_duplicates(self, rag_config: dict) -> None:
        """All tuples in fractional matrix must be unique."""
        matrix = build_rag_matrix(rag_config, mode="fractional")
        assert len(matrix) == len(set(matrix))

    def test_full_matrix_tuple_structure(self, rag_config: dict) -> None:
        """Each tuple contains (model, strategy, retriever, noise_level) - 4 elements."""
        matrix = build_rag_matrix(rag_config, mode="full")
        for cell in matrix:
            assert len(cell) == 4
            model, strategy, retriever, noise_level = cell
            assert isinstance(model, str)
            assert isinstance(strategy, str)
            assert isinstance(retriever, str)
            assert isinstance(noise_level, float)

    def test_unknown_mode_raises(self, rag_config: dict) -> None:
        """ValueError raised for unrecognised mode string."""
        with pytest.raises(ValueError, match="Unknown mode"):
            build_rag_matrix(rag_config, mode="bogus_mode")

    def test_outer_loop_is_model(self, rag_config: dict) -> None:
        """All cells for the first model appear before any cell for the second model."""
        matrix = build_rag_matrix(rag_config, mode="full")
        models_in_order = list(rag_config["models"].keys())
        first_model = models_in_order[0]
        second_model = models_in_order[1]

        # Find the last occurrence of the first model and first occurrence of the second
        last_first = max(i for i, cell in enumerate(matrix) if cell[0] == first_model)
        first_second = min(
            i for i, cell in enumerate(matrix) if cell[0] == second_model
        )

        assert last_first < first_second, (
            f"Expected all {first_model} cells before {second_model} cells, "
            f"but last {first_model} at index {last_first}, "
            f"first {second_model} at index {first_second}"
        )

    def test_full_matrix_no_duplicates(self, rag_config: dict) -> None:
        """Full matrix has no duplicate cells."""
        matrix = build_rag_matrix(rag_config, mode="full")
        assert len(matrix) == len(set(matrix))

    def test_full_matrix_all_models_present(self, rag_config: dict) -> None:
        """Every model from config appears in the full matrix."""
        matrix = build_rag_matrix(rag_config, mode="full")
        models_in_matrix = {cell[0] for cell in matrix}
        assert models_in_matrix == set(rag_config["models"].keys())

    def test_full_matrix_all_strategies_present(self, rag_config: dict) -> None:
        """Every strategy from config appears in the full matrix."""
        matrix = build_rag_matrix(rag_config, mode="full")
        strategies_in_matrix = {cell[1] for cell in matrix}
        assert strategies_in_matrix == set(rag_config["strategies"])

    def test_full_matrix_all_retrievers_present(self, rag_config: dict) -> None:
        """Every retriever from config appears in the full matrix."""
        matrix = build_rag_matrix(rag_config, mode="full")
        retrievers_in_matrix = {cell[2] for cell in matrix}
        assert retrievers_in_matrix == set(rag_config["retrievers"])

    def test_full_matrix_all_noise_levels_present(self, rag_config: dict) -> None:
        """Every noise level from config appears in the full matrix."""
        matrix = build_rag_matrix(rag_config, mode="full")
        noise_in_matrix = {cell[3] for cell in matrix}
        assert noise_in_matrix == set(rag_config["noise"]["levels"])

    def test_fractional_hybrid_has_all_noise_levels(self, rag_config: dict) -> None:
        """Fractional mode: hybrid retriever has all 4 noise levels."""
        matrix = build_rag_matrix(rag_config, mode="fractional")
        hybrid_cells = [cell for cell in matrix if cell[2] == "hybrid"]
        hybrid_noise_levels = {cell[3] for cell in hybrid_cells}
        expected = set(rag_config["noise"]["levels"])
        assert hybrid_noise_levels == expected

    def test_fractional_bm25_dense_only_noise0(self, rag_config: dict) -> None:
        """Fractional mode: bm25 and dense only appear at noise_level=0.0."""
        matrix = build_rag_matrix(rag_config, mode="fractional")
        for cell in matrix:
            model, strategy, retriever, noise_level = cell
            if retriever in ("bm25", "dense"):
                assert noise_level == 0.0, (
                    f"Expected bm25/dense only at noise=0.0, "
                    f"got {retriever} at noise={noise_level}"
                )

    def test_filter_by_models_subset(self, rag_config: dict) -> None:
        """Passing models list restricts matrix to those models only."""
        subset = ["llama-3.2-3b", "qwen2.5-7b"]
        matrix = build_rag_matrix(rag_config, mode="full", models=subset)
        models_in_matrix = {cell[0] for cell in matrix}
        assert models_in_matrix == set(subset)
        # 2 models x 4 strategies x 3 retrievers x 4 noise = 96
        assert len(matrix) == 96

    def test_filter_by_strategies_subset(self, rag_config: dict) -> None:
        """Passing strategies list restricts matrix to those strategies only."""
        subset = ["da", "ctl"]
        matrix = build_rag_matrix(rag_config, mode="full", strategies=subset)
        strategies_in_matrix = {cell[1] for cell in matrix}
        assert strategies_in_matrix == set(subset)

    def test_filter_by_retrievers_subset(self, rag_config: dict) -> None:
        """Passing retrievers list restricts matrix to those retrievers only."""
        subset = ["hybrid"]
        matrix = build_rag_matrix(rag_config, mode="full", retrievers=subset)
        retrievers_in_matrix = {cell[2] for cell in matrix}
        assert retrievers_in_matrix == {"hybrid"}

    def test_filter_by_noise_levels_subset(self, rag_config: dict) -> None:
        """Passing noise_levels list restricts matrix to those levels only."""
        subset = [0.0, 0.4]
        matrix = build_rag_matrix(rag_config, mode="full", noise_levels=subset)
        noise_in_matrix = {cell[3] for cell in matrix}
        assert noise_in_matrix == {0.0, 0.4}

    def test_unknown_model_raises(self, rag_config: dict) -> None:
        """ValueError raised for model name not in config."""
        with pytest.raises(ValueError, match="Unknown model"):
            build_rag_matrix(rag_config, mode="full", models=["not-a-model"])

    def test_unknown_strategy_raises(self, rag_config: dict) -> None:
        """ValueError raised for strategy not in config."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_rag_matrix(rag_config, mode="full", strategies=["not-a-strategy"])

    def test_unknown_retriever_raises(self, rag_config: dict) -> None:
        """ValueError raised for retriever not in config."""
        with pytest.raises(ValueError, match="Unknown retriever"):
            build_rag_matrix(rag_config, mode="full", retrievers=["not-a-retriever"])

    def test_fractional_matrix_hybrid_count(self, rag_config: dict) -> None:
        """Fractional: hybrid contributes 6 models x 4 strategies x 4 noise = 96 cells."""
        matrix = build_rag_matrix(rag_config, mode="fractional")
        hybrid_cells = [cell for cell in matrix if cell[2] == "hybrid"]
        assert len(hybrid_cells) == 96

    def test_fractional_matrix_bm25_count(self, rag_config: dict) -> None:
        """Fractional: bm25 contributes 6 models x 4 strategies x 1 noise = 24 cells."""
        matrix = build_rag_matrix(rag_config, mode="fractional")
        bm25_cells = [cell for cell in matrix if cell[2] == "bm25"]
        assert len(bm25_cells) == 24

    def test_fractional_matrix_dense_count(self, rag_config: dict) -> None:
        """Fractional: dense contributes 6 models x 4 strategies x 1 noise = 24 cells."""
        matrix = build_rag_matrix(rag_config, mode="fractional")
        dense_cells = [cell for cell in matrix if cell[2] == "dense"]
        assert len(dense_cells) == 24
