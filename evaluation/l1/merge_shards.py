#!/usr/bin/env python3
"""Merge N shard prediction JSONLs (from run_vllm.py --shard i/N) and score once."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--format", choices=["action", "native"], default="native")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = []
    for s in args.shards:
        records.extend(load_jsonl(s))
    print(f"Merged {len(records)} predictions from {len(args.shards)} shards", file=sys.stderr)

    summary, _ = evaluate(records, use_semantic=not args.no_semantic, fmt=args.format)
    print("\n===== L-1 STEP-LEVEL SUMMARY (merged) =====")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()
