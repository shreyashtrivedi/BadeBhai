"""Embedding providers behind a small interface."""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

from badebhai.config import Settings, get_settings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


_TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing. No downloads; used in tests and CI.

    Quality is far below a neural model, but it is stable and gives keyword-like similarity.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text):
            t = tok.lower()
            h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class SentenceTransformerEmbedder:
    # bge models expect this instruction on queries (not on passages).
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, dim: int, batch_size: int = 32):
        self.name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name, device="cpu")
                dim_fn = (
                    getattr(self._model, "get_embedding_dimension", None)
                    or self._model.get_sentence_embedding_dimension
                )
                actual = dim_fn()
                if actual != self.dim:
                    raise ValueError(f"{self.name} produces {actual}-d vectors but EMBEDDING_DIM={self.dim}")
            return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.BGE_QUERY_PREFIX if "bge" in self.name.lower() else ""
        return self.model.encode([prefix + text], normalize_embeddings=True)[0].astype(np.float32)


class OpenAICompatEmbedder:
    """Embeddings from an OpenAI-compatible /embeddings endpoint (Gemini's free tier, OpenAI, Ollama /v1).
    For hosts too small to load a local model (a 512 MB free web service). Vectors are L2-normalized."""

    def __init__(
        self, model: str, dim: int, base_url: str, api_key: str | None, batch: int = 64, transport=None
    ):
        import httpx

        self.name, self.dim, self.batch = model, dim, max(1, min(batch, 100))
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/", headers=headers, timeout=60.0, transport=transport
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        import time

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            body = {"model": self.name, "input": texts[i : i + self.batch], "dimensions": self.dim}
            for attempt in range(6):
                r = self._http.post("embeddings", json=body)
                if r.status_code == 429 and attempt < 5:  # free tiers: wait and retry
                    time.sleep(min(60.0, float(r.headers.get("retry-after") or 2 ** (attempt + 1))))
                    continue
                if r.status_code >= 400:
                    raise RuntimeError(f"embedding API error {r.status_code}: {r.text[:200]}")
                break
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            out.extend(d["embedding"] for d in data)
        vecs = np.asarray(out, dtype=np.float32)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"{self.name} returned {vecs.shape[1]}-d vectors but EMBEDDING_DIM={self.dim}")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


@lru_cache(maxsize=4)
def _build(
    provider: str, model: str, dim: int, batch: int, base_url: str | None, api_key: str | None
) -> Embedder:
    if provider == "hash":
        return HashingEmbedder(dim)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model, dim, batch)
    if provider == "openai":
        if not base_url:
            raise ValueError("EMBEDDING_PROVIDER=openai needs EMBEDDING_BASE_URL (or OPENAI_BASE_URL)")
        return OpenAICompatEmbedder(model, dim, base_url, api_key, batch)
    raise ValueError(f"unknown EMBEDDING_PROVIDER {provider!r}")


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    base = s.embedding_base_url or s.openai_base_url
    key = s.embedding_api_key or s.openai_api_key
    return _build(s.embedding_provider, s.embedding_model, s.embedding_dim, s.embedding_batch_size, base, key)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0 or math.isnan(na) or math.isnan(nb):
        return 0.0
    return float(np.dot(a, b) / (na * nb))
