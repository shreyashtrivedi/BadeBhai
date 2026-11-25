"""Central configuration, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Applied when LLM_PROVIDER=ollama (small local models are slower and have small context windows).
# Any of these set explicitly in the environment wins.
LOCAL_PROFILE = {
    "max_subquestions": 3,
    "evidence_per_call": 4,
    "max_calls_per_subquestion": 4,  # tool calls cost no LLM time; grading stays capped per round
    "max_evidence_for_synthesis": 10,
    "request_timeout_s": 1200.0,
    "max_retries": 1,
    "grade_items_per_subquestion": 6,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- storage ---
    database_url: str = "postgresql://badebhai:badebhai@localhost:5433/badebhai_dev"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- LLM ---
    # ollama (default): free, local, open models. anthropic: optional paid API (`uv sync --extra anthropic`).
    llm_provider: str = "ollama"  # ollama | anthropic | fake
    # Overrides the provider's default model (OLLAMA_MODEL for ollama, claude-opus-5-5 for anthropic).
    llm_model: str | None = None
    # Model for decompose / grade / rewrite / memory extraction; defaults to the main model.
    llm_fast_model: str | None = None

    # Ollama (free, self-hosted)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"  # fits a 4 GB GPU; 7-8B models are stronger but slower there
    ollama_fast_model: str | None = None
    ollama_num_ctx: int = 8192  # context window; evidence budgets scale with it
    ollama_max_output_tokens: int = 2048
    ollama_temperature: float = 0.2
    ollama_think: bool = False  # "thinking" mode of reasoning models: slower, sometimes better plans
    ollama_keep_alive: str = "30m"
    ollama_timeout_s: float = 600.0

    # Anthropic (optional, paid; not installed by default)
    # Read from .env too (pydantic-settings does not export .env into os.environ, so pass it explicitly).
    anthropic_api_key: SecretStr | None = None
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
    grade_items_per_subquestion: int = 10  # new items graded per sub-question per round
    request_timeout_s: float = 240.0

    # --- memory ---
    memory_importance_threshold: float = 0.6
    memory_half_life_days: float = 30.0
    memory_duplicate_similarity: float = 0.92
    memory_top_k: int = 5
    memory_min_weight: float = 0.15

    # --- tracing (LangSmith reads os.environ; configure_tracing() exports these) ---
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "badebhai"
    langsmith_endpoint: str | None = None
    log_level: str = "INFO"
    log_json: bool = True

    @model_validator(mode="after")
    def _local_profile(self) -> Settings:
        """Leaner agent defaults for small local models, unless a value was set explicitly."""
        if self.llm_provider == "ollama":
            for name, value in LOCAL_PROFILE.items():
                if name not in self.model_fields_set:
                    object.__setattr__(self, name, value)
        return self

    @property
    def main_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {"ollama": self.ollama_model, "anthropic": "claude-opus-5-5"}.get(self.llm_provider, "fake")

    @property
    def fast_model(self) -> str:
        if self.llm_fast_model:
            return self.llm_fast_model
        if self.llm_provider == "ollama" and self.ollama_fast_model:
            return self.ollama_fast_model
        return self.main_model

    @property
    def evidence_char_budget(self) -> int:
        """Characters of evidence that fit in one prompt next to instructions and the answer."""
        if self.llm_provider == "ollama":
            return int(self.ollama_num_ctx * 3.0 * 0.55)  # ~3 chars/token; leave room for prompt + output
        return 60_000

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_tracing(s: Settings) -> bool:
    """Export LangSmith settings (possibly from .env) so LangGraph / langsmith pick them up."""
    import os

    if not (s.langsmith_tracing and s.langsmith_api_key):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", s.langsmith_api_key.get_secret_value())
    os.environ.setdefault("LANGSMITH_PROJECT", s.langsmith_project)
    if s.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", s.langsmith_endpoint)
    return True
