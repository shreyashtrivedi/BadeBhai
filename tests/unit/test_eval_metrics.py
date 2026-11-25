from repomind.evals.metrics import judge_score, summarize


def _r(**kw):
    base = {
        "id": "q",
        "category": "location",
        "answerable": True,
        "has_expectations": True,
        "hits": {"hit@1": True, "hit@3": True, "hit@5": True, "hit@10": True},
        "recall_any": True,
        "cited_hit": True,
        "citations_total": 1,
        "citations_valid": 1,
        "said_insufficient": False,
        "fully_abstained": False,
        "latency_ms": 1000,
        "cost_usd": 0.0,
        "llm_calls": 1,
        "tool_calls": 0,
        "judge_score": 1.0,
    }
    return {**base, **kw}


def test_abstaining_on_an_answerable_question_scores_zero_whatever_the_judge_said():
    assert judge_score(_r(fully_abstained=True, said_insufficient=True)) == 0.0
    assert judge_score(_r(answerable=False, fully_abstained=True)) == 1.0  # correct abstention
    assert judge_score(_r(judge_score=0.75)) == 0.75


def test_summary_uses_the_adjusted_judge_score():
    s = summarize([_r(), _r(id="q2", fully_abstained=True, said_insufficient=True, citations_total=0)])
    assert s["judge_correctness"] == 0.5
    assert s["over_abstention"] == 0.5
    assert s["judge_by_category"] == {"location": 0.5}


def test_refusals_phrased_as_does_not_manage_count_as_insufficient():
    from repomind.evals.metrics import said_insufficient

    assert said_insufficient("Requests does not manage WebSocket connections or heartbeat intervals.")
    assert not said_insufficient("Requests manages connection pools through urllib3 [1].")
