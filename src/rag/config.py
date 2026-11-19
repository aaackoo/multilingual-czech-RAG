"""
Central config, loaded once from environment variables / .env.

Using pydantic-settings (rather than scattering os.environ.get() calls
through every module) gives us validation, defaults, and type coercion in
one place, and makes every tunable knob (models, chunk size, top_k, rerank
toggle) visible from a single file -- important for a demo someone else
will read.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM / embedding provider access (all calls go through LiteLLM) ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # LiteLLM model strings use a "provider/model-name" format, which is
    # exactly what lets us swap providers by editing .env instead of code.
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    generation_model: str = "openai/gpt-4o-mini"
    eval_judge_model: str = "openai/gpt-4o-mini"

    # --- Postgres / pgvector ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"
    postgres_user: str = "rag"
    postgres_password: str = "rag"

    # --- Reranking (stage 4) ---
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Retrieval (stage 3) ---
    retrieval_top_k: int = 5  # final number of chunks used for generation
    retrieval_candidates: int = 20  # candidates pulled before reranking

    # --- Chunking (stage 1) ---
    chunk_size_chars: int = 800
    chunk_overlap_chars: int = 150

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    # lru_cache makes this a cheap process-wide singleton -- Settings() re-parses
    # the environment every call otherwise, and every module just wants one
    # consistent view of config.
    return Settings()
