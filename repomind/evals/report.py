"""Write eval results as JSON (raw) and a Markdown summary table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).parent / "reports"

ROWS = [
    ("hit@1", "Retrieval hit@1", "pct"),
    ("hit@3", "Retrieval hit@3", "pct"),
    ("hit@5", "Retrieval hit@5", "pct"),
    ("hit@10", "Retrieval hit@10", "pct"),
    ("recall_any", "Expected source found anywhere in gathered evidence", "pct"),
    ("cited_expected", "Answer cites an expected file/commit/issue", "pct"),
    ("citation_validity", "Citation validity rate", "pct"),
    ("judge_correctness", "LLM-judge answer correctness (0-1; abstaining when answerable = 0)", "num"),
    ("insufficient_accuracy", "Says 'insufficient evidence' on unanswerable questions", "pct"),
    ("over_abstention", "Abstains entirely on answerable questions", "pct"),
    ("latency_p50_ms", "Latency p50", "ms"),
    ("latency_p95_ms", "Latency p95", "ms"),
    ("cost_mean_usd", "Cost per question (mean)", "usd"),
    ("llm_calls_mean", "LLM calls per question", "num"),
    ("tool_calls_mean", "MCP tool calls per question", "num"),
]


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "ms":
        return f"{v / 1000:.1f} s"
    if kind == "usd":
        return f"${v:.4f}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def markdown(report: dict[str, Any]) -> str:
    systems = list(report["summary"])
    head = "| Metric | " + " | ".join(systems) + " |\n|---|" + "---|" * len(systems) + "\n"
    rows = []
    for key, label, kind in ROWS:
        vals = [report["summary"][s].get(key) for s in systems]
        if all(v is None for v in vals):
            continue
        rows.append(f"| {label} | " + " | ".join(_fmt(v, kind) for v in vals) + " |")
    n = max(report["summary"][s]["questions"] for s in systems)
    errs = {s: report["summary"][s]["errors"] for s in systems}
    lines = [
        f"## Eval run {report['run_id']}",
        "",
        f"Repo `{report['repo']}` @ `{report['ref']}` ({(report['commit'] or '')[:10]}), {n} questions "
        f"from `{report['golden']}`. Models: {report['models']}. Errors: {errs}.",
        "",
        head + "\n".join(rows),
    ]
    cats = {
        s: report["summary"][s].get("judge_by_category")
        for s in systems
        if report["summary"][s].get("judge_by_category")
    }
    if cats:
        all_cats = sorted({c for v in cats.values() for c in v})
        lines += [
            "",
            "Judge correctness by category:",
            "",
            "| Category | " + " | ".join(cats) + " |\n|---|" + "---|" * len(cats),
        ]
        lines += [
            f"| {c} | " + " | ".join(_fmt(cats[s].get(c), "num") for s in cats) + " |" for c in all_cats
        ]
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], out_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    raw = out_dir / "raw" / f"{report['run_id']}.json"
    raw.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = out_dir / f"{report['run_id']}.md"
    text = markdown(report)
    md.write_text(text, encoding="utf-8")
    (out_dir / "latest.md").write_text(text, encoding="utf-8")
    return raw, md
