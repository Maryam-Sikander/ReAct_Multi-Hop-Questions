from __future__ import annotations

import dataclasses
import re

from src.agents.prompts import build_prompt
from src.tools.wiki_search import WikiEnv
from src.utils.llm_client import LLMClient
from src.utils.token_tracker import TokenTracker

ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\nAction:|\Z)", re.DOTALL)


@dataclasses.dataclass
class HopRecord:
    hop: int
    thought: str
    action_type: str
    action_arg: str
    observation: str


@dataclasses.dataclass
class TrajectoryResult:
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str | None
    hops: list[HopRecord]
    stopped_reason: str  # "finished" | "max_hops" | "parse_error"
    token_usage: dict


class ReActAgent:
    def __init__(self, llm: LLMClient, max_hops: int = 7):
        self.llm = llm
        self.max_hops = max_hops

    def run(self, question_id: str, question: str, gold_answer: str) -> TrajectoryResult:
        env = WikiEnv()
        tracker = TokenTracker()
        trajectory_text = ""
        hops: list[HopRecord] = []
        predicted_answer = None
        stopped_reason = "max_hops"

        for hop_idx in range(1, self.max_hops + 1):
            prompt = build_prompt(question, trajectory_text)
            response = self.llm.complete(prompt, stop=["\nObservation"])
            tracker.log("agent", hop_idx, response.prompt_tokens, response.completion_tokens)

            generated = response.text.strip()
            thought_match = THOUGHT_RE.search(generated)
            action_match = ACTION_RE.search(generated)

            if not action_match:
                stopped_reason = "parse_error"
                hops.append(HopRecord(
                    hop=hop_idx,
                    thought=thought_match.group(1).strip() if thought_match else generated,
                    action_type="PARSE_ERROR",
                    action_arg="",
                    observation=f"Could not parse an Action from: {generated!r}",
                ))
                break

            thought = thought_match.group(1).strip() if thought_match else ""
            action_type = action_match.group(1).strip()
            action_arg = action_match.group(2).strip()

            if action_type == "Finish":
                predicted_answer = action_arg
                stopped_reason = "finished"
                hops.append(HopRecord(hop_idx, thought, action_type, action_arg, observation=""))
                break

            if action_type == "Search":
                observation = env.search(action_arg)
            elif action_type == "Lookup":
                observation = env.lookup(action_arg)
            else:
                observation = f"Unrecognized action type: {action_type}"

            hops.append(HopRecord(hop_idx, thought, action_type, action_arg, observation))
            trajectory_text += (
                f"Thought: {thought}\n"
                f"Action: {action_type}[{action_arg}]\n"
                f"Observation: {observation}\n"
            )

        return TrajectoryResult(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
            hops=hops,
            stopped_reason=stopped_reason,
            token_usage=tracker.to_dict(),
        )
