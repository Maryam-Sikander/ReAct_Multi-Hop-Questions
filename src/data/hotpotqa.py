from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests
from tqdm import tqdm

HOTPOTQA_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def download_raw(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "hotpot_dev_distractor_v1.json"
    if out_path.exists() and not force:
        print(f"already have it: {out_path}")
        return out_path

    print("downloading HotpotQA distractor dev set (~40MB)...")
    resp = requests.get(HOTPOTQA_DEV_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return out_path


def sample_questions(raw_path: Path, n: int, seed: int, hard_only: bool = True) -> list[dict]:
    with open(raw_path) as f:
        data = json.load(f)

    if hard_only:
        # 'hard' level questions are the ones that actually need multi-hop
        # reasoning rather than being answerable from a single paragraph.
        pool = [q for q in data if q.get("level") == "hard"]
        if len(pool) < n:
            print(f"only {len(pool)} hard questions available, falling back to full pool")
            pool = data
    else:
        pool = data

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    # keep only what the agent and the analysis actually need
    trimmed = []
    for q in sample:
        trimmed.append({
            "id": q["_id"],
            "question": q["question"],
            "answer": q["answer"],
            "type": q.get("type"),
            "level": q.get("level"),
            "supporting_facts": q.get("supporting_facts", []),
        })
    return trimmed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-only", action="store_true", default=True)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--out", type=str, default=None, help="output filename override")
    args = parser.parse_args()

    raw_path = download_raw(force=args.force_download)
    sample = sample_questions(raw_path, n=args.n, seed=args.seed, hard_only=args.hard_only)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out or f"hotpot_sample_n{args.n}_seed{args.seed}.jsonl"
    out_path = PROCESSED_DIR / out_name
    with open(out_path, "w") as f:
        for q in sample:
            f.write(json.dumps(q) + "\n")

    print(f"wrote {len(sample)} questions to {out_path}")


if __name__ == "__main__":
    main()
