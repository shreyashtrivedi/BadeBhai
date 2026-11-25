"""Golden question set loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

GOLDEN_DIR = Path(__file__).parent / "golden"
DEFAULT_GOLDEN = GOLDEN_DIR / "requests_v2_32_3.yaml"


@dataclass
class GoldenQuestion:
    id: str
    category: str
    question: str
    reference: str
    expected_files: list[str] = field(default_factory=list)
    expected_commits: list[str] = field(default_factory=list)
    expected_issue: bool = False
    sources: list[str] = field(default_factory=list)
    answerable: bool = True
    notes: str = ""  # free text for maintainers (e.g. where a question came from)

    @property
    def has_expectations(self) -> bool:
        return bool(self.expected_files or self.expected_commits or self.expected_issue)


@dataclass
class GoldenSet:
    repo: str
    ref: str
    commit: str
    questions: list[GoldenQuestion]


def load_golden(path: Path | str = DEFAULT_GOLDEN) -> GoldenSet:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    qs = [GoldenQuestion(**q) for q in raw["questions"]]
    ids = [q.id for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in golden set")
    return GoldenSet(repo=raw["repo"], ref=raw["ref"], commit=raw["commit"], questions=qs)
