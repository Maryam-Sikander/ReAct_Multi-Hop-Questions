import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def sample_questions(n, seed, hard_only=True):
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    pool = [q for q in ds if q["level"] == "hard"] if hard_only else list(ds)
    if hard_only and len(pool) < n:
        print(f"only {len(pool)} hard questions available, using full pool")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    sample = sample_questions(args.n, args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / (args.out or f"hotpot_sample_n{args.n}_seed{args.seed}.jsonl")
    with open(out_path, "w") as f:
        for q in sample:
            f.write(json.dumps(q) + "\n")

    print(f"wrote {len(sample)} questions to {out_path}")


if __name__ == "__main__":
    main()