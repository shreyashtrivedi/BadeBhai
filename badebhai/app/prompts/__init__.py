"""Versioned prompt files.

Prompts live next to this module as `<name>.v<N>.md`. The highest version is used unless
`PROMPT_VERSION_<NAME>=<N>` pins one (e.g. PROMPT_VERSION_SYNTHESIZE=1). Templates use
`$placeholder` syntax (string.Template), so JSON braces in prompts need no escaping.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Template

_DIR = Path(__file__).parent
_FILE = re.compile(r"^(?P<name>[a-z_]+)\.v(?P<ver>\d+)\.md$")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}.v{self.version}"

    def render(self, **values: object) -> str:
        return Template(self.text).substitute({k: str(v) for k, v in values.items()})


def available() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for f in _DIR.glob("*.md"):
        m = _FILE.match(f.name)
        if m:
            out.setdefault(m["name"], []).append(int(m["ver"]))
    return {k: sorted(v) for k, v in out.items()}


@cache
def load_prompt(name: str, version: int | None = None) -> Prompt:
    versions = available().get(name)
    if not versions:
        raise FileNotFoundError(f"no prompt file for {name!r} in {_DIR}")
    pinned = os.environ.get(f"PROMPT_VERSION_{name.upper()}")
    ver = version or (int(pinned) if pinned else versions[-1])
    if ver not in versions:
        raise FileNotFoundError(f"prompt {name}.v{ver}.md not found (have {versions})")
    return Prompt(name, ver, (_DIR / f"{name}.v{ver}.md").read_text(encoding="utf-8").strip())
