"""Optional cross-encoder reranking of hybrid-search candidates (local, free, CPU).

Hybrid search ranks with two cheap signals (embedding similarity, keyword match) fused by rank. A
cross-encoder reads the question and each candidate *together*, which judges relevance much better,
but is too slow to run over a whole repo; so it reorders only the top RERANK_CANDIDATES.

The final score blends the reranker with the fused rank (`rerank_weight`), so a reranker that is weak
on code cannot throw away a candidate both cheap signals agree on.
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache
from typing import TYPE_CHECKING

from badebhai.log import get_logger

if TYPE_CHECKING:
    from badebhai.retrieval.hybrid import Hit

log = get_logger(__name__)

_PASSAGE_CHARS = 1500


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.name, device="cpu", max_length=512)
                log.info("reranker loaded", extra={"model": self.name})
            return self._model

    def scores(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self.model.predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)
        return [float(x) for x in raw]


@lru_cache(maxsize=4)
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


def passage(hit: Hit) -> str:
    head = hit.path + (f" :: {hit.symbol}" if hit.symbol else "")
    return f"{head}\n{hit.content[:_PASSAGE_CHARS]}"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def rerank(query: str, hits: list[Hit], reranker, weight: float, top_n: int) -> list[Hit]:
    """Reorder the first `top_n` hits (already sorted by fused score); the rest keep their order."""
    head, tail = hits[:top_n], hits[top_n:]
    if len(head) < 2:
        return hits
    ce = reranker.scores(query, [passage(h) for h in head])
    n = len(head)
    blended = []
    for i, (h, c) in enumerate(zip(head, ce, strict=True)):
        prior = 1.0 - i / n  # fused rank, 1.0 for the top candidate
        blended.append((weight * _sigmoid(c) + (1.0 - weight) * prior, h))
    blended.sort(key=lambda t: t[0], reverse=True)
    for score, h in blended:
        h.rerank_score = round(score, 4)
    return [h for _, h in blended] + tail
