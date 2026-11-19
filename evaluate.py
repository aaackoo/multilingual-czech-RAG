"""
Stage 6: measure RAG pipeline quality against a golden Czech Q&A set.

    uv run python evaluate.py

This is a custom, lightweight LLM-as-judge implementation rather than a
framework like RAGAS -- deliberately, so every metric here is a short,
readable prompt + JSON parse, not a black box. It computes three
complementary signals per question:

  - faithfulness / groundedness: does the generated answer only assert
    things the retrieved context actually supports? This measures the
    GENERATION step -- a pipeline can retrieve perfectly and still
    hallucinate on top of good context.
  - context precision: of the chunks retrieval handed to the model, what
    fraction were actually relevant to the question? This measures the
    RETRIEVAL (+ rerank) step in isolation, independent of what generation
    did with them -- so a low score here points at retrieval/chunking, not
    the prompt.
  - answer relevancy: does the answer actually address what was asked (is
    it on-topic, not just truthful)? Reported as two independent signals
    side by side: an LLM-judge score, and a cheap non-LLM sanity check --
    cosine similarity between the generated answer's embedding and the
    golden answer's embedding. When they disagree sharply, that's usually
    worth a manual look.

Known bias, worth saying out loud: using an LLM to judge a pipeline that
itself calls an LLM risks the judge favoring the generator's own style
("self-preference bias"). EVAL_JUDGE_MODEL is a separate config knob from
GENERATION_MODEL for exactly this reason -- point it at a different model
to reduce (not eliminate) that risk.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from rag.generate import build_context_block
from rag.llm import embed, judge
from rag.pipeline import answer_query
from rag.retrieval import RetrievedChunk

GOLDEN_SET_PATH = Path("data/golden_eval.json")
REPORT_PATH = Path("eval_report.json")


def _parse_json(text: str) -> dict | None:
    """Judge models occasionally wrap JSON in markdown fences despite being
    asked not to -- strip those before parsing rather than failing the
    whole metric over a formatting quirk."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def evaluate_faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float | None:
    """Asks the judge to break the answer into claims and check each one
    against the retrieved context. Score = fraction of claims supported."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict fact-checking judge. You will be given CONTEXT "
                "(numbered excerpts) and an ANSWER. Break the answer into individual "
                "factual claims and decide whether each is supported by the context. "
                'Respond with ONLY a JSON object: {"claims": [{"claim": str, '
                '"supported": bool}], "faithfulness_score": float} where '
                "faithfulness_score is the fraction of claims that are supported "
                "(1.0 if there are no claims to check)."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{build_context_block(chunks)}\n\nANSWER:\n{answer}",
        },
    ]
    parsed = _parse_json(judge(messages, response_format={"type": "json_object"}))
    return parsed.get("faithfulness_score") if parsed else None


def evaluate_context_precision(question: str, chunks: list[RetrievedChunk]) -> float | None:
    """Asks the judge to mark each retrieved chunk relevant/irrelevant to
    the question. Score = fraction of retrieved chunks that were relevant."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a relevance judge. Given a QUESTION and numbered CONTEXT "
                "chunks, decide which chunks contain information relevant to "
                "answering the question. Respond with ONLY a JSON object: "
                '{"relevance": [{"index": int, "relevant": bool}]}, one entry per '
                "chunk."
            ),
        },
        {
            "role": "user",
            "content": f"QUESTION: {question}\n\nCONTEXT:\n{build_context_block(chunks)}",
        },
    ]
    parsed = _parse_json(judge(messages, response_format={"type": "json_object"}))
    if not parsed or not parsed.get("relevance"):
        return None
    flags = [item["relevant"] for item in parsed["relevance"]]
    return sum(flags) / len(flags)


def evaluate_answer_relevancy_llm(question: str, answer: str) -> float | None:
    """Asks the judge whether the answer is on-topic for the question --
    deliberately independent of factual correctness (that's faithfulness's
    job), this only checks "did it answer what was asked"."""
    messages = [
        {
            "role": "system",
            "content": (
                "You judge whether an ANSWER actually addresses the QUESTION asked, "
                "regardless of whether the answer is factually correct. Respond with "
                'ONLY a JSON object: {"relevancy_score": float} from 0.0 (off-topic) '
                "to 1.0 (fully addresses the question)."
            ),
        },
        {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {answer}"},
    ]
    parsed = _parse_json(judge(messages, response_format={"type": "json_object"}))
    return parsed.get("relevancy_score") if parsed else None


@dataclass
class EvalResult:
    question: str
    answer: str
    ground_truth_answer: str
    faithfulness: float | None
    context_precision: float | None
    answer_relevancy_llm: float | None
    answer_relevancy_embedding: float | None


def evaluate_one(item: dict) -> EvalResult:
    result = answer_query(item["question"])

    faithfulness = evaluate_faithfulness(result.answer, result.retrieved_chunks)
    context_precision = evaluate_context_precision(item["question"], result.retrieved_chunks)
    relevancy_llm = evaluate_answer_relevancy_llm(item["question"], result.answer)

    answer_emb, truth_emb = embed([result.answer, item["ground_truth_answer"]])
    relevancy_embedding = _cosine_similarity(answer_emb, truth_emb)

    return EvalResult(
        question=item["question"],
        answer=result.answer,
        ground_truth_answer=item["ground_truth_answer"],
        faithfulness=faithfulness,
        context_precision=context_precision,
        answer_relevancy_llm=relevancy_llm,
        answer_relevancy_embedding=relevancy_embedding,
    )


def _avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def print_report(results: list[EvalResult]) -> None:
    console = Console()
    table = Table(title="RAG Evaluation Report", show_lines=True)
    table.add_column("Question", overflow="fold", max_width=40)
    table.add_column("Faithfulness", justify="right")
    table.add_column("Context Precision", justify="right")
    table.add_column("Answer Relevancy (LLM)", justify="right")
    table.add_column("Answer Relevancy (embedding)", justify="right")

    def fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    for r in results:
        table.add_row(
            r.question,
            fmt(r.faithfulness),
            fmt(r.context_precision),
            fmt(r.answer_relevancy_llm),
            fmt(r.answer_relevancy_embedding),
        )

    table.add_section()
    table.add_row(
        "AVERAGE",
        fmt(_avg([r.faithfulness for r in results])),
        fmt(_avg([r.context_precision for r in results])),
        fmt(_avg([r.answer_relevancy_llm for r in results])),
        fmt(_avg([r.answer_relevancy_embedding for r in results])),
        style="bold",
    )

    console.print(table)


def main() -> None:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    results = [evaluate_one(item) for item in golden_set]

    print_report(results)

    REPORT_PATH.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
