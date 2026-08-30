from __future__ import annotations

import dataclasses
import json
from collections import defaultdict


@dataclasses.dataclass
class CallRecord:
    role: str  # "agent" | "judge"
    hop: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:
    def __init__(self):
        self.calls: list[CallRecord] = []

    def log(self, role: str, hop: int, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls.append(CallRecord(role, hop, prompt_tokens, completion_tokens))

    def total_tokens(self, role: str | None = None) -> int:
        return sum(c.total for c in self.calls if role is None or c.role == role)

    def num_calls(self, role: str | None = None) -> int:
        return sum(1 for c in self.calls if role is None or c.role == role)

    def breakdown_by_role(self) -> dict:
        out = defaultdict(lambda: {"calls": 0, "tokens": 0})
        for c in self.calls:
            out[c.role]["calls"] += 1
            out[c.role]["tokens"] += c.total
        return dict(out)

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens(),
            "total_calls": self.num_calls(),
            "by_role": self.breakdown_by_role(),
            "calls": [dataclasses.asdict(c) for c in self.calls],
        }
