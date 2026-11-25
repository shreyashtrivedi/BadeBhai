"""Citation verifier: bad citations must be caught programmatically."""

from badebhai.app.citations import UNVERIFIED_MARK, check_evidence, verify_answer
from badebhai.app.schemas import Evidence

FILE = "\n".join(f"line {i}" for i in range(1, 101))


class FakeOracle:
    def __init__(self):
        self.files = {"src/auth.py": FILE}
        self.commits = {"abc1234def5678"}

    def line_count(self, path):
        return len(self.files[path].split("\n")) if path in self.files else None

    def read_lines(self, path, start, end):
        return "\n".join(self.files[path].split("\n")[start - 1 : end]) if path in self.files else None

    def commit_exists(self, sha):
        return any(c.startswith(sha) for c in self.commits)

    def find_path(self, partial):
        return next((p for p in self.files if p == partial or p.endswith("/" + partial)), None)


def code_ev(start, end, content=None, path="src/auth.py"):
    if content is None:
        content = "\n".join(f"line {i}" for i in range(start, end + 1))
    return Evidence(
        key=f"code:{path}#L{start}-L{end}",
        type="code",
        source_tool="t",
        title=path,
        content=content,
        path=path,
        start_line=start,
        end_line=end,
    )


def test_valid_citations_are_kept_and_renumbered_in_order():
    ev = [code_ev(1, 5), code_ev(10, 12), code_ev(20, 22)]
    out = verify_answer("Auth lives in auth.py [3]. It checks tokens [1][3].", ev, FakeOracle())
    assert out.answer == "Auth lives in auth.py [1]. It checks tokens [2][1]."
    assert [c.evidence_key for c in out.citations] == [ev[2].key, ev[0].key]
    assert out.report.total_citations == 3 and out.report.valid_citations == 3
    assert out.citations[0].label == "src/auth.py#L20-L22"


def test_unknown_evidence_id_is_stripped_and_claim_flagged():
    ev = [code_ev(1, 5)]
    out = verify_answer(
        "Retries are configured in the adapter [7]. Auth is in auth.py [1].", ev, FakeOracle()
    )
    assert "[7]" not in out.answer
    assert UNVERIFIED_MARK.strip() in out.answer
    assert out.report.invalid_citations[0]["marker"] == 7
    assert "Retries are configured in the adapter" in out.report.flagged_claims[0]
    assert out.report.valid_citations == 1


def test_line_range_beyond_file_is_invalid():
    ev = [code_ev(95, 140, content="whatever")]
    out = verify_answer("The handler is defined at the end of the file [1].", ev, FakeOracle())
    assert out.report.valid_citations == 0
    assert "exceeds file length" in out.report.invalid_citations[0]["problem"]
    assert out.citations == []


def test_content_mismatch_is_invalid():
    ev = [code_ev(1, 3, content="def fabricated():\n    pass\n")]
    problem = check_evidence(ev[0], FakeOracle())
    assert problem and "do not match" in problem


def test_truncated_snippet_prefix_is_accepted():
    ev = code_ev(1, 10, content="line 1\nline 2\n…")
    assert check_evidence(ev, FakeOracle()) is None


def test_unindexed_file_and_missing_commit():
    assert "not in the index" in check_evidence(code_ev(1, 2, path="nope.py"), FakeOracle())
    commit = Evidence(
        key="commit:deadbeef00", type="commit", source_tool="t", title="c", content="", sha="deadbeef00"
    )
    assert "not found" in check_evidence(commit, FakeOracle())
    good = commit.model_copy(update={"sha": "abc1234def5678", "key": "commit:abc1234def5678"})
    assert check_evidence(good, FakeOracle()) is None


def test_comma_separated_markers_and_partial_validity():
    ev = [code_ev(1, 2), code_ev(3, 4)]
    out = verify_answer("Both helpers exist in the module [1, 2, 9].", ev, FakeOracle())
    assert out.answer == "Both helpers exist in the module [1][2]."
    assert out.report.total_citations == 3 and out.report.valid_citations == 2


def test_uncited_claims_reported_but_meta_sentences_ignored():
    ev = [code_ev(1, 2)]
    text = "The retry logic uses exponential backoff with jitter.\n\nI couldn't find evidence for rate limits in the repo."
    out = verify_answer(text, ev, FakeOracle())
    assert out.report.uncited_claims == ["The retry logic uses exponential backoff with jitter."]


def test_bad_inline_reference_is_flagged():
    ev = [code_ev(1, 2)]
    out = verify_answer("See src/auth.py:500 and src/auth.py:10 for details [1].", ev, FakeOracle())
    assert out.report.bad_inline_refs == ["src/auth.py:500"]
    assert "reference not found" in out.answer


def test_code_blocks_are_not_segmented_as_claims():
    ev = [code_ev(1, 2)]
    text = "Example usage [1]:\n\n```python\nsession.auth = ('user', 'pass')  # sets basic auth for all requests\n```\n"
    out = verify_answer(text, ev, FakeOracle())
    assert out.report.uncited_claims == []


def test_trailing_references_line_is_removed_before_counting():
    ev = [code_ev(1, 2), code_ev(3, 4)]
    out = verify_answer("Auth helpers live in the module [1].\n\nCitations: [1][2]", ev, FakeOracle())
    assert out.answer == "Auth helpers live in the module [1]."
    assert out.report.total_citations == 1
