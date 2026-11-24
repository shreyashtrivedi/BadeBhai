"""Hybrid retrieval: pgvector cosine search + Postgres full-text search, fused with RRF."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from psycopg.rows import dict_row

from repomind.config import Settings, get_settings
from repomind.embeddings import Embedder
from repomind.ingestion.chunker import split_identifier
from repomind.ingestion.languages import is_test_path

# Question words that dominate keyword matching without saying anything about the code.
_STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "did",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "any",
        "all",
        "also",
        "but",
        "each",
        "get",
        "got",
        "just",
        "more",
        "most",
        "much",
        "no",
        "not",
        "only",
        "other",
        "out",
        "over",
        "same",
        "some",
        "such",
        "too",
        "very",
        "via",
        "handled",
        "handle",
        "handles",
        "handling",
        "implemented",
        "implementation",
        "implement",
        "located",
        "location",
        "defined",
        "define",
        "code",
        "file",
        "files",
        "function",
        "functions",
        "logic",
        "work",
        "works",
        "working",
        "used",
        "use",
        "uses",
        "using",
        "find",
        "show",
        "explain",
        "tell",
        "happen",
        "happens",
        "repo",
        "repository",
        "project",
        "module",
        "please",
    ]
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def keyword_terms(query: str, max_terms: int = 16) -> list[str]:
    terms: list[str] = []
    for tok in _WORD.findall(query):
        cands = [tok.lower()]
        if "_" in tok or any(c.isupper() for c in tok[1:]):
            cands += split_identifier(tok)
        for t in cands:
            t = t.strip("_")
            if len(t) >= 2 and t not in _STOP and t not in terms and t.replace("_", "").isalnum():
                terms.append(t)
    return terms[:max_terms]


def build_tsquery(query: str) -> str | None:
    terms = keyword_terms(query)
    if not terms:
        return None
    return " | ".join(f"{t}:*" if len(t) >= 4 else t for t in terms)


@dataclass
class Hit:
    chunk_id: int
    path: str
    kind: str
    language: str | None
    symbol: str | None
    symbol_type: str | None
    start_line: int
    end_line: int
    content: str
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    similarity: float | None = None

    def to_dict(self, max_chars: int | None = None) -> dict:
        d = asdict(self)
        if max_chars and len(d["content"]) > max_chars:
            d["content"] = d["content"][:max_chars] + "\n…"
        return d


_SELECT = "id AS chunk_id, path, kind, language, symbol, symbol_type, start_line, end_line, content"


def hybrid_search(
    pool,
    embedder: Embedder,
    repo_id: int,
    query: str,
    *,
    k: int = 8,
    kinds: tuple[str, ...] = ("code",),
    path_prefix: str | None = None,
    language: str | None = None,
    settings: Settings | None = None,
) -> list[Hit]:
    s = settings or get_settings()
    n = max(s.retrieval_candidates, k * 3)
    filters = ["repo_id = %(repo)s", "kind = ANY(%(kinds)s)"]
    params: dict = {"repo": repo_id, "kinds": list(kinds), "n": n}
    if path_prefix:
        filters.append("path LIKE %(prefix)s")
        params["prefix"] = path_prefix.rstrip("*").replace("%", r"\%") + "%"
    if language:
        filters.append("language = %(lang)s")
        params["lang"] = language
    where = " AND ".join(filters)

    qvec = embedder.embed_query(query)
    params["qvec"] = qvec
    tsq = build_tsquery(query)
    params["tsq"] = tsq

    rows: dict[int, dict] = {}
    vec_rank: dict[int, int] = {}
    kw_rank: dict[int, int] = {}
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {_SELECT}, 1 - (embedding <=> %(qvec)s) AS similarity FROM chunks "
            f"WHERE {where} ORDER BY embedding <=> %(qvec)s LIMIT %(n)s",
            params,
        )
        for i, r in enumerate(cur.fetchall()):
            rows[r["chunk_id"]] = r
            vec_rank[r["chunk_id"]] = i + 1
        if tsq:
            cur.execute(
                f"SELECT {_SELECT}, NULL::float AS similarity, ts_rank_cd(tsv, q, 32) AS kw "
                f"FROM chunks, to_tsquery('english', %(tsq)s) AS q "
                f"WHERE {where} AND tsv @@ q ORDER BY kw DESC LIMIT %(n)s",
                params,
            )
            for i, r in enumerate(cur.fetchall()):
                rows.setdefault(r["chunk_id"], r)
                kw_rank[r["chunk_id"]] = i + 1

    hits: list[Hit] = []
    for cid, r in rows.items():
        score = 0.0
        if cid in vec_rank:
            score += 1.0 / (s.rrf_k + vec_rank[cid])
        if cid in kw_rank:
            score += 1.0 / (s.rrf_k + kw_rank[cid])
        if is_test_path(r["path"]) and not (path_prefix and is_test_path(path_prefix + "x")):
            score *= s.test_path_penalty
        hits.append(
            Hit(
                chunk_id=cid,
                path=r["path"],
                kind=r["kind"],
                language=r["language"],
                symbol=r["symbol"],
                symbol_type=r["symbol_type"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                content=r["content"],
                score=round(score * 1000, 3),
                vector_rank=vec_rank.get(cid),
                keyword_rank=kw_rank.get(cid),
                similarity=round(float(r["similarity"]), 4) if r.get("similarity") is not None else None,
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return _dedupe_overlaps(hits)[:k]


def _dedupe_overlaps(hits: list[Hit]) -> list[Hit]:
    """Drop a hit that mostly overlaps a better-ranked hit from the same file (sliding windows)."""
    kept: list[Hit] = []
    for h in hits:
        dup = False
        for o in kept:
            if o.path == h.path:
                overlap = min(o.end_line, h.end_line) - max(o.start_line, h.start_line) + 1
                if overlap > 0.5 * (h.end_line - h.start_line + 1):
                    dup = True
                    break
        if not dup:
            kept.append(h)
    return kept
