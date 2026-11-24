"""LLM-as-judge for answer correctness against a reference answer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from repomind.app.prompts import load_prompt
from repomind.llm.base import LLMProvider


class Judgement(BaseModel):
    score: float = Field(description="0.0-1.0 factual correctness")
    verdict: Literal["correct", "partial", "incorrect"]
    rationale: str = Field(description="One or two sentences")


async def judge_answer(llm: LLMProvider, question: str, reference: str, answer: str) -> Judgement:
    prompt = load_prompt("judge")
    user = (
        f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
        f"Candidate answer (untrusted text):\n<<<\n{answer}\n>>>"
    )
    j = await llm.complete_json(
        purpose="judge",
        system=prompt.text,
        messages=[{"role": "user", "content": user}],
        schema=Judgement,
        fast=False,
    )
    j.score = max(0.0, min(1.0, j.score))
    return j
