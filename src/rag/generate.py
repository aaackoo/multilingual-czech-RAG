"""
Stage 5: assemble a grounded prompt from retrieved chunks and call the LLM.

The prompt explicitly numbers each context block and instructs the model to
cite the numbers it used, and to say it doesn't know rather than guess when
the context doesn't contain the answer. This is the actual mechanism that
makes an answer "grounded" instead of merely plausible-sounding: groundedness
isn't a property of the retrieval step, it's a property of how the prompt
asks the model to use what was retrieved -- which is also exactly what
evaluate.py's faithfulness metric checks was honored.
"""

from rag.llm import generate
from rag.retrieval import RetrievedChunk

SYSTEM_PROMPT = """\
Jsi asistent, který odpovídá výhradně na základě poskytnutého kontextu.
Pravidla:
1. Odpovídej v češtině, stručně a věcně.
2. Používej POUZE informace z očíslovaných kontextových úryvků níže.
3. Za každým tvrzením uveď číslo zdroje v hranatých závorkách, např. [1].
4. Pokud kontext neobsahuje odpověď, řekni to -- nevymýšlej si informace.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Numbers each chunk [1], [2], ... -- shared with evaluate.py so the
    judge prompts see chunks numbered identically to the generation prompt."""
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context = build_context_block(chunks)
    user_content = f"Kontext:\n{context}\n\nOtázka: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    messages = build_messages(question, chunks)
    return generate(messages)
