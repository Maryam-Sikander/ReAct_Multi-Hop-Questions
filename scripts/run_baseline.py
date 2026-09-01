"""
Runs the baseline ReAct agent over a sampled question set and writes one
JSON line per question to results/logs/. Nothing gets aggregated here —
that's a separate analysis step, once both conditions exist and I actually
want to compare them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.agents.react_agent import ReActAgent
from src.utils.llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "configs" / "models.yaml") as f:
        return yaml.safe_load(f)


def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config = load_config()
    parser.add_argument("--model", default="gemma", choices=list(config["models"].keys()),
                         help="which model config to use, from configs/models.yaml")
    parser.add_argument("--questions", type=str, default=None,
                         help="path to a processed .jsonl question file; "
                              "defaults to the most recent one in data/processed/")
    parser.add_argument("--n", type=int, default=None,
                         help="only run the first N questions (for smoke tests)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model_cfg = config["models"][args.model]
    run_cfg = config["run"]

    if args.questions:
        q_path = Path(args.questions)
    else:
        candidates = sorted((ROOT / "data" / "processed").glob("hotpot_sample_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                "No sampled question file found in data/processed/. "
                "Run `python -m src.data.hotpotqa --n <count>` first."
            )
        q_path = candidates[-1]

    questions = load_questions(q_path)
    if args.n:
        questions = questions[: args.n]

    llm = LLMClient(
        model_id=model_cfg["id"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        timeout_s=run_cfg["request_timeout_s"],
        retry_attempts=run_cfg["retry_attempts"],
        retry_backoff_s=run_cfg["retry_backoff_s"],
    )
    agent = ReActAgent(llm=llm, max_hops=run_cfg["max_hops"])

    out_name = args.out or f"baseline_{args.model}_{q_path.stem}.jsonl"
    out_path = ROOT / "results" / "logs" / out_name

    print(f"running baseline ReAct | model={model_cfg['id']} | questions={len(questions)} | -> {out_path}")

    with open(out_path, "w") as f:
        for q in tqdm(questions):
            result = agent.run(
                question_id=q["id"],
                question=q["question"],
                gold_answer=q["answer"],
            )
            f.write(json.dumps(dataclasses.asdict(result)) + "\n")

    print("done.")


if __name__ == "__main__":
    main()
