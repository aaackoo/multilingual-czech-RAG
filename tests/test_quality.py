"""
Thin pytest wrapper around evaluate.py's quality metrics, for CI gating.

evaluate.py stays the source of truth for the human-readable report (rich
table, per-question breakdown, eval_report.json) -- deliberately not
duplicated here. This file only asserts on the aggregate averages
evaluate.py already computes, so a quality regression fails the build
instead of only showing up if someone remembers to read the report.

Runs the full pipeline (embedding + generation + judge calls) against every
golden question, same as `uv run python evaluate.py` -- real API calls,
not fast or free. Marked "quality" so it's opt-in rather than part of every
default `pytest` invocation:

    uv run pytest -m quality -v
"""

import json

import pytest

from evaluate import GOLDEN_SET_PATH, _avg, evaluate_one

# Thresholds taken from this repo's own baseline eval run (see README's
# "Key design decisions" section), not arbitrary round numbers. Tightening
# these over time as the pipeline improves is the actual point of gating
# on them in CI.
MIN_FAITHFULNESS = 0.9
MIN_ANSWER_RELEVANCY_LLM = 0.9
MIN_ANSWER_RELEVANCY_EMBEDDING = 0.8
# Deliberately low -- see README's "Known limitations": context precision is
# structurally capped on this demo's small corpus (top_k=5 over ~18 chunks),
# so a high bar here would fail for corpus-size reasons, not retrieval bugs.
MIN_CONTEXT_PRECISION = 0.2

pytestmark = pytest.mark.quality


@pytest.fixture(scope="session")
def eval_results():
    """Runs the whole golden set through the live pipeline once per test
    session and shares the result across every assertion below, rather
    than re-running the (expensive) pipeline once per metric."""
    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    return [evaluate_one(item) for item in golden_set]


def test_faithfulness_above_threshold(eval_results):
    avg = _avg([r.faithfulness for r in eval_results])
    assert avg is not None and avg >= MIN_FAITHFULNESS, (
        f"average faithfulness {avg} fell below {MIN_FAITHFULNESS}"
    )


def test_context_precision_above_threshold(eval_results):
    avg = _avg([r.context_precision for r in eval_results])
    assert avg is not None and avg >= MIN_CONTEXT_PRECISION, (
        f"average context precision {avg} fell below {MIN_CONTEXT_PRECISION}"
    )


def test_answer_relevancy_llm_above_threshold(eval_results):
    avg = _avg([r.answer_relevancy_llm for r in eval_results])
    assert avg is not None and avg >= MIN_ANSWER_RELEVANCY_LLM, (
        f"average LLM answer relevancy {avg} fell below {MIN_ANSWER_RELEVANCY_LLM}"
    )


def test_answer_relevancy_embedding_above_threshold(eval_results):
    avg = _avg([r.answer_relevancy_embedding for r in eval_results])
    assert avg is not None and avg >= MIN_ANSWER_RELEVANCY_EMBEDDING, (
        f"average embedding answer relevancy {avg} fell below {MIN_ANSWER_RELEVANCY_EMBEDDING}"
    )
