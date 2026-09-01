import argparse
import dataclasses
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.agents.react_agent import ReActAgent
from src.utils.llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(ROOT / "configs" / "models.yaml") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma", choices=list(config["models"].keys()))
    parser.add_argument("--questions", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model_cfg = config["models"][args.model]
    run_cfg = config["run"]

    if args.questions:
        q_path = Path(args.questions)
    else:
        candidates = sorted((ROOT / "data" / "processed").glob("hotpot_sample_*.jsonl"))
        if not candidates:
            raise FileNotFoundError("run src/data/hotpotqa.py first to generate a question sample")
        q_path = candidates[-1]

    with open(q_path) as f:
        questions = [json.loads(line) for line in f]
    if args.n:
        questions = questions[: args.n]

    llm = LLMClient(model_cfg["id"], model_cfg["temperature"], model_cfg["max_tokens"],
                     run_cfg["request_timeout_s"], run_cfg["retry_attempts"], run_cfg["retry_backoff_s"])
    agent = ReActAgent(llm, max_hops=run_cfg["max_hops"])

    out_path = ROOT / "results" / "logs" / (args.out or f"baseline_{args.model}_{q_path.stem}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"running {len(questions)} questions with {model_cfg['id']} -> {out_path}")
    with open(out_path, "w") as f:
        for q in tqdm(questions):
            result = agent.run(q["id"], q["question"], q["answer"])
            f.write(json.dumps(dataclasses.asdict(result)) + "\n")

    print("done")


if __name__ == "__main__":
    main()