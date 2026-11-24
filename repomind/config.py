"""Central configuration, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- storage ---
    database_url: str = "postgresql://repomind:repomind@localhost:5433/repomind"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- LLM ---
    llm_provider: str = "anthropic"  # anthropic | fake
    llm_model: str = "claude-opus-5-5"
    # Used for decompose / grade / rewrite / memory extraction. Defaults to the main model;
    # set to e.g. claude-haiku-5-5 to cut cost.
    llm_fast_model: str | None = None
    llm_effort: str = "medium"  # synthesis effort: low | medium | high | xhigh | max
    llm_fast_effort: str = "low"  # effort for the small structured steps
    llm_fallbacks: str = "default"  # server-side refusal fallback: "default" or "off"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3

    # --- embeddings ---
    embedding_provider: str = "sentence-transformers"  # sentence-transformers | hash
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # --- ingestion ---
    max_file_bytes: int = 400_000
    chunk_max_lines: int = 80
    window_lines: int = 50
    window_overlap: int = 10
    index_wiki: bool = True

    # --- retrieval ---
    retrieval_candidates: int = 40  # per-arm candidates before fusion
    rrf_k: int = 60
    test_path_penalty: float = 0.85  # multiply fused score of test files (1.0 disables)

    # --- MCP ---
    mcp_transport: str = "http"  # http | stdio | inmemory
    mcp_code_search_url: str = "http://localhost:8101/mcp"
    mcp_git_history_url: str = "http://localhost:8102/mcp"
    mcp_docs_url: str = "http://localhost:8103/mcp"
    mcp_tool_timeout_s: float = 25.0

    # --- GitHub ---
    github_token: str | None = None
    github_webhook_secret: str | None = None
    github_api_url: str = "https://api.github.com"
    github_cache_ttl_s: int = 120

    # --- live sync ---
    sync_poll_interval_s: int = 0  # 0 disables the polling loop
    auto_ingest_repo: str | None = None  # e.g. https://github.com/psf/requests@v2.32.3

    # --- agent guardrails ---
    max_subquestions: int = 5
    max_retries: int = 2  # query-rewrite retries per sub-question
    max_tool_calls: int = 24  # per question
    max_calls_per_subquestion: int = 4  # per retrieval round
    evidence_per_call: int = 6
    max_evidence_for_synthesis: int = 16
    grade_threshold: float = 0.5  # relevance score (0-1) for an item to count as supporting evidence
    request_timeout_s: float = 240.0

    # --- memory ---
    memory_importance_threshold: float = 0.6
    memory_half_life_days: float = 30.0
    memory_duplicate_similarity: float = 0.92
    memory_top_k: int = 5
    memory_min_weight: float = 0.15

    # --- tracing ---
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def fast_model(self) -> str:
        return self.llm_fast_model or self.llm_model

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"


@lru_cache
def get_settings() -> Settings:
    return Settings()
