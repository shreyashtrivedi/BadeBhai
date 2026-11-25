"""The eval harness end to end (scripted LLM, tiny repo): both systems, judge, metrics, report."""

import pytest
import yaml

from repomind import compat
from repomind.evals.report import markdown, write
from repomind.evals.runner import EvalRunner
from repomind.ingestion.indexer import Indexer
from repomind.llm.fake import FakeLLM
from tests.integration.helpers import evidence_ids, grade_all, plan

pytestmark = pytest.mark.db


def test_eval_runner_compares_naive_and_agent(db_settings, tiny_repo, tmp_path):
    rep = Indexer(db_settings).ingest(str(tiny_repo), name="acme/tiny")
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "repo": "acme/tiny",
                "ref": "main",
                "commit": rep.commit,
                "questions": [
                    {
                        "id": "auth",
                        "category": "location",
                        "question": "Where is basic authentication handled?",
                        "expected_files": ["app/auth.py"],
                        "reference": "BasicAuth in app/auth.py",
                    },
                    {
                        "id": "graphql",
                        "category": "unanswerable",
                        "question": "Where is the GraphQL resolver?",
                        "answerable": False,
                        "reference": "There is none.",
                    },
                ],
            }
        )
    )

    def naive(system, messages, schema):
        ids = evidence_ids(messages[-1]["content"])
        if "GraphQL" in messages[-1]["content"]:
            return "I couldn't find evidence of a GraphQL resolver in this repository."
        return f"BasicAuth in app/auth.py handles it [{ids['code'][0]}]."

    def decompose(system, messages, schema):
        q = messages[-1]["content"]
        text = q.split("Question: ")[-1]
        return plan((text, ["code"], text, [("search_code", {"query": text})]))

    def grade(system, messages, schema):
        weak = "GraphQL" in messages[-1]["content"]
        return grade_all(0.1 if weak else 0.9, sufficient=not weak)(system, messages, schema)

    def rewrite(system, messages, schema):
        return {"replans": [{"sub_question_id": "q1", "query": "graphql", "calls": []}]}

    llm = FakeLLM(
        {
            "naive": naive,
            "decompose": decompose,
            "grade": grade,
            "rewrite": rewrite,
            "synthesize": lambda s, m, sc: (
                f"BasicAuth lives in app/auth.py [{evidence_ids(m[-1]['content'])['code'][0]}]."
            ),
        }
    )
    judge = FakeLLM({"judge": lambda s, m, sc: {"score": 1.0, "verdict": "correct", "rationale": "matches"}})
    report = compat.run(
        EvalRunner(db_settings, llm=llm, judge_llm=judge).evaluate(golden, ("naive", "agent"))
    )
    naive_s, agent_s = report["summary"]["naive"], report["summary"]["agent"]
    assert naive_s["hit@1"] == 1.0 and agent_s["hit@1"] == 1.0
    assert naive_s["citation_validity"] == 1.0 and agent_s["citation_validity"] == 1.0
    assert naive_s["insufficient_accuracy"] == 1.0 and agent_s["insufficient_accuracy"] == 1.0
    assert agent_s["judge_correctness"] == 1.0 and agent_s["errors"] == 0
    assert agent_s["tool_calls_mean"] >= 1 and agent_s["llm_calls_mean"] > naive_s["llm_calls_mean"]
    md = markdown(report)
    assert "| Retrieval hit@1 | 100.0% | 100.0% |" in md and "LLM-judge" in md
    raw, page = write(report, tmp_path / "reports")
    assert raw.exists() and page.exists() and (tmp_path / "reports" / "latest.md").exists()


def test_eval_checkpoint_and_resume(db_settings, tiny_repo, tmp_path):
    rep = Indexer(db_settings).ingest(str(tiny_repo), name="acme/tiny")
    golden = tmp_path / "golden.yaml"
    qs = [
        {
            "id": f"q{i}",
            "category": "location",
            "question": "Where is basic authentication handled?",
            "expected_files": ["app/auth.py"],
            "reference": "app/auth.py",
        }
        for i in range(3)
    ]
    golden.write_text(
        yaml.safe_dump({"repo": "acme/tiny", "ref": "main", "commit": rep.commit, "questions": qs})
    )
    calls = {"n": 0}

    def naive(system, messages, schema):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")  # second question fails mid-run
        return f"BasicAuth in app/auth.py [{evidence_ids(messages[-1]['content'])['code'][0]}]."

    runner = EvalRunner(db_settings, llm=FakeLLM({"naive": naive}))
    ck = tmp_path / "ck"
    first = compat.run(runner.evaluate(golden, ("naive",), checkpoint_dir=ck))
    assert first["summary"]["naive"]["errors"] == 1
    run_id = first["run_id"]
    assert len((ck / f"{run_id}.partial.jsonl").read_text().splitlines()) == 3
    second = compat.run(runner.evaluate(golden, ("naive",), checkpoint_dir=ck, resume=run_id))
    assert calls["n"] == 4  # only the failed question was re-run
    assert second["summary"]["naive"]["errors"] == 0 and second["summary"]["naive"]["questions"] == 3
