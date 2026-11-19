"""
Single choke point for all LLM and embedding calls.

Every other module calls `generate()` or `embed()` from here instead of
importing litellm directly. Two reasons:

1. It's where the "model-agnostic" promise actually lives -- swapping
   providers means editing GENERATION_MODEL / EMBEDDING_MODEL in .env, not
   touching retrieval.py, generate.py, or evaluate.py.
2. It's where a LiteLLM Router with fallbacks belongs. We only have an
   OpenAI key for this demo, so the router currently has one model per role,
   but the commented block below shows exactly how a second provider would
   be wired in for automatic failover -- the point being that the failover
   *mechanism* is real infrastructure, not something bolted on only once a
   second key exists.
"""

import logging

import litellm
from litellm import Router

from rag.config import get_settings

# LiteLLM logs a WARNING every time it can't find our custom "generation"/
# "embedding" router aliases in its built-in per-token cost table -- noisy
# and irrelevant to this demo (we don't rely on LiteLLM's cost tracking).
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

settings = get_settings()

# LiteLLM Router model aliases. Code elsewhere calls Router with these
# logical names ("generation", "embedding") rather than a concrete
# provider/model string, which is what makes the fallback list below a
# config change instead of a code change.
_model_list = [
    {
        "model_name": "generation",
        "litellm_params": {
            "model": settings.generation_model,
            "api_key": settings.openai_api_key,
        },
    },
    {
        "model_name": "embedding",
        "litellm_params": {
            "model": settings.embedding_model,
            "api_key": settings.openai_api_key,
        },
    },
    {
        # Separate alias from "generation" so EVAL_JUDGE_MODEL can be set to
        # a different model than GENERATION_MODEL -- using the pipeline's
        # own generation model to grade the pipeline's own output risks
        # self-preference bias (see evaluate.py).
        "model_name": "judge",
        "litellm_params": {
            "model": settings.eval_judge_model,
            "api_key": settings.openai_api_key,
        },
    },
]

# --- To demonstrate provider fallback once you have a second API key ---
# 1. Add ANTHROPIC_API_KEY to .env
# 2. Add a second entry with the same model_name="generation" but a
#    different underlying model, e.g.:
#      {
#          "model_name": "generation",
#          "litellm_params": {
#              "model": "anthropic/claude-3-5-haiku-latest",
#              "api_key": settings.anthropic_api_key,
#          },
#      }
# LiteLLM Router treats multiple entries sharing a model_name as a pool and
# will retry against the next one on failure -- no application code changes
# needed, which is the whole point of routing through one abstraction.

router = Router(model_list=_model_list)

# Keep raw exceptions instead of swallowing them behind a generic wrapper --
# a demo should surface provider errors (bad key, rate limit) plainly rather
# than hiding them.
litellm.suppress_debug_info = True


def generate(messages: list[dict], **kwargs) -> str:
    """Chat completion via the 'generation' model alias. Returns plain text."""
    response = router.completion(model="generation", messages=messages, **kwargs)
    return response.choices[0].message.content


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the 'embedding' model alias."""
    response = router.embedding(model="embedding", input=texts)
    return [item["embedding"] for item in response.data]


def judge(messages: list[dict], **kwargs) -> str:
    """Chat completion via the 'judge' model alias, used only by evaluate.py."""
    response = router.completion(model="judge", messages=messages, **kwargs)
    return response.choices[0].message.content
