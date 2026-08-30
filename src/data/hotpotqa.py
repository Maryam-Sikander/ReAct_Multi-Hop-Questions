from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def sample_questions(n: int, seed: int, hard_only: bool = True) -> list[dict]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    if hard_only:
        pool = [q for q in ds if q["level"] == "hard"]
        if len(pool) < n:
            print(f"only {len(pool)} hard questions in the set, using the full pool instead")
            pool = list(ds)
    else:
        pool = list(ds)

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    return [
        {
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "type": q["type"],
            "level": q["level"],
            "supporting_facts": q["supporting_facts"],
        }
        for q in sample
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-only", action="store_true", default=True)
    parser.add_argument("--out", type=str, default=None, help="output filename override")
    args = parser.parse_args()

    sample = sample_questions(n=args.n, seed=args.seed, hard_only=args.hard_only)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out or f"hotpot_sample_n{args.n}_seed{args.seed}.jsonl"
    out_path = PROCESSED_DIR / out_name
    with open(out_path, "w") as f:
        for q in sample:
            f.write(json.dumps(q) + "\n")

    print(f"wrote {len(sample)} questions to {out_path}")


if __name__ == "__main__":
    main()