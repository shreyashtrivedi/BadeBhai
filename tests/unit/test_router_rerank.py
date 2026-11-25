"""Auto-mode routing and reranker blending."""

import pytest

from badebhai.app.router import route
from badebhai.retrieval.hybrid import Hit
from badebhai.retrieval.rerank import rerank


@pytest.mark.parametrize(
    "q",
    [
        "Where is authentication handled?",
        "Which data structure stores HTTP headers case-insensitively?",
        "How do I install requests?",
        "Does requests support GraphQL?",
        "Where does requests load credentials from a .netrc file?",
    ],
)
def test_simple_lookups_take_the_fast_path(q):
    assert route(q).mode == "naive"


@pytest.mark.parametrize(
    "q",
    [
        "How does a Session decide which transport adapter handles a URL?",  # how it works
        "Why was Proxy-Authorization handling changed in 2.31.0?",  # history + version
        "What changed in src/requests during May 2024?",
        "Which files would I touch to add a new authentication scheme?",  # planning
        "Where is Response.json implemented and what does it raise for invalid JSON?",  # two parts
        "Are there GitHub issues discussing retries?",
        "What are the main parts of this codebase?",
        "Where is X? And where is it tested?",
    ],
)
def test_anything_needing_reasoning_goes_to_the_agent(q):
    r = route(q)
    assert r.mode == "agent" and r.reason


def test_follow_ups_that_refer_back_use_the_agent():
    assert route("Where is it tested?", follow_up=True).mode == "agent"
    assert route("Where is it tested?").mode == "naive"


def _hit(i: int, path: str) -> Hit:
    return Hit(i, path, "code", "python", None, None, 1, 10, f"content {path}", score=1.0 - i / 10)


class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def scores(self, query, passages):
        return [self._scores[p.split("\n")[0]] for p in passages]


def test_reranker_reorders_the_head_and_keeps_the_tail():
    hits = [_hit(0, "a.py"), _hit(1, "b.py"), _hit(2, "c.py"), _hit(3, "d.py")]
    fake = FakeReranker({"a.py": -5.0, "b.py": 6.0, "c.py": 0.0})
    out = rerank("q", hits, fake, weight=1.0, top_n=3)
    assert [h.path for h in out] == ["b.py", "c.py", "a.py", "d.py"]
    assert out[0].rerank_score is not None and out[-1].rerank_score is None


def test_blend_weight_zero_keeps_the_fused_order():
    hits = [_hit(0, "a.py"), _hit(1, "b.py"), _hit(2, "c.py")]
    fake = FakeReranker({"a.py": -5.0, "b.py": 6.0, "c.py": 0.0})
    assert [h.path for h in rerank("q", hits, fake, weight=0.0, top_n=3)] == ["a.py", "b.py", "c.py"]
