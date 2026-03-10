"""Shared fixtures for prompt builder and response parser tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def sample_mcq4_record() -> dict:
    """Realistic MC-4 question from SciKnowEval."""
    return {
        "question": (
            "Which of the following cellular components is primarily "
            "responsible for governing cell expansion and wall thickening "
            "in plant cells?"
        ),
        "answer": "",
        "type": "mcq-4-choices",
        "domain": "Biology",
        "details": {
            "level": "L1",
            "task": "literature_multi_choice_question",
            "subtask": "PubMed_QA",
            "source": "PubMed",
        },
        "answerKey": "A",
        "choices": {
            "text": [
                "KNOTTED ARABIDOPSIS THALIANA7 (KNAT7)",
                "Growth-Regulating Factor 4 (GRF4)",
                "sclerenchyma fiber cells",
                "NAC31",
            ],
            "label": ["A", "B", "C", "D"],
        },
    }


@pytest.fixture()
def sample_mcq2_record() -> dict:
    """MC-2 question with only two choices."""
    return {
        "question": "Is water a polar molecule?",
        "answer": "",
        "type": "mcq-2-choices",
        "domain": "Chemistry",
        "details": {"level": "L1", "task": "basic_qa", "subtask": "basic_qa", "source": "test"},
        "answerKey": "A",
        "choices": {
            "text": ["Yes, due to its bent geometry", "No, it is nonpolar"],
            "label": ["A", "B"],
        },
    }


@pytest.fixture()
def sample_open_ended_record() -> dict:
    """Open-ended question from SciKnowEval."""
    return {
        "question": (
            "Below is a user's experimental design requirement:\n"
            "I aimed to identify novel biomarkers for predicting responses "
            "to immunotherapy in human cancers.\n\n"
            "Design the experimental procedure."
        ),
        "answer": "1. Install R packages.\n2. Acquire datasets.\n3. Analyze.",
        "type": "open-ended-qa",
        "domain": "Biology",
        "details": {
            "level": "L5",
            "task": "procedure_generation",
            "subtask": "procedure_generation",
            "source": "star-protocol",
        },
        "answerKey": "",
        "choices": {"text": [], "label": []},
    }


@pytest.fixture()
def sample_tf_record() -> dict:
    """True/false question from SciKnowEval."""
    return {
        "question": "After using the gas cylinder, the valve can be left open.",
        "answer": "No",
        "type": "true_or_false",
        "domain": "Chemistry",
        "details": {
            "level": "L4",
            "task": "laboratory_safety_test",
            "subtask": "laboratory_safety_test_judgement",
            "source": "real university test",
        },
        "answerKey": "",
        "choices": {"text": [], "label": []},
    }


@pytest.fixture()
def sample_filling_record() -> dict:
    """Filling (chemical equation balancing) question."""
    return {
        "question": (
            "Here is a unbalanced chemical equation:\n"
            "NH4 + NO2 = H2O + N\n"
            "The balanced chemical equation is:"
        ),
        "answer": "NH4 + NO2 = 2H2O + 2N",
        "type": "filling",
        "domain": "Chemistry",
        "details": {"level": "L3", "task": "balancing_chemical_equation",
                     "subtask": "balancing_chemical_equation", "source": "test"},
        "answerKey": "",
        "choices": {"text": [], "label": []},
    }


@pytest.fixture()
def sample_relation_extraction_record() -> dict:
    """Relation extraction question."""
    return {
        "question": (
            "Graphene transistors were prepared using p-doped silicon (Si) "
            "substrates with a SiO2 layer.\n\n"
            "Extract doping information from this sentence.\n\n###"
        ),
        "answer": "The host 'silicon' was doped.\n\nEND",
        "type": "relation_extraction",
        "domain": "Chemistry",
        "details": {"level": "L2", "task": "L2_Chemistry",
                     "subtask": "extract_doping", "source": "Bohrium"},
        "answerKey": "",
        "choices": {"text": [], "label": []},
    }


@pytest.fixture()
def choices_4() -> dict[str, list]:
    """Standard 4-choice options."""
    return {
        "text": ["Fire", "Water", "Air", "Earth"],
        "label": ["A", "B", "C", "D"],
    }


@pytest.fixture()
def choices_2() -> dict[str, list]:
    """Binary choice options."""
    return {
        "text": ["Yes, it is polar", "No, it is nonpolar"],
        "label": ["A", "B"],
    }


@pytest.fixture()
def sample_passages() -> list[str]:
    """Three sample RAG passages."""
    return [
        "Water (H2O) is a polar inorganic compound that is a tasteless and odorless liquid at room temperature.",
        "Carbon dioxide (CO2) is a chemical compound occurring as a colorless gas.",
        "Sodium chloride (NaCl) is an ionic compound commonly known as table salt.",
    ]
