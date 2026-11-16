#!/usr/bin/env python3
"""
CLI: load evaluation/l1/data/test.jsonl -> build step-level datapoints (datapoints.py)
-> run inference via Fireworks (fireworks_client.py) -> score (metrics.py) -> report
overall + by_path_type (happy/unhappy/impossible).

Usage:
    export FIREWORKS_API_KEY=...

    # pre-fine-tune base model: v3 <action>-grammar prompt
    python evaluation/l1/run.py --model accounts/fireworks/models/<base-model-id> \\
        --format action --out evaluation/l1/results/base.json

    # Fireworks-deployed fine-tuned checkpoint: native SFT prompt + tools=
    python evaluation/l1/run.py --model accounts/<you>/deployedModels/<ft-id> \\
        --format native --out evaluation/l1/results/finetuned.json

    # smoke test on a handful of conversations before the full 1,881-conversation run
    python evaluation/l1/run.py --model <id> --format native \\
        --n-conversations 5 --out /tmp/smoke.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datapoints import build_step_datapoints  # noqa: E402
from fireworks_client import run_inference  # noqa: E402
from metrics import evaluate  # noqa: E402

DEFAULT_TEST_FILE = Path(__file__).resolve().parent / "data" / "test.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    ap.add_argument("--model", required=True, help="Fireworks model or deployment ID")
    ap.add_argument("--format", choices=["action", "native"], default="native",
                     help="action: v3 <action>-grammar prompt (pre-FT base model). "
                          "native: SFT prompt + tools= (Fireworks-deployed fine-tune).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-conversations", type=int, default=None,
                     help="Subsample the first N conversations (smoke tests); default: all")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--predictions", default=None,
                     help="Save raw per-step predictions here; reused (skipping "
                          "inference) if the file already exists")
    ap.add_argument("--out", required=True, help="Where to write the summary JSON")
    args = ap.parse_args()

    conversations = load_jsonl(args.test_file)
    if args.n_conversations:
        conversations = conversations[: args.n_conversations]
    print(f"Loaded {len(conversations)} conversations from {args.test_file}", file=sys.stderr)

    datapoints = []
    for ex in conversations:
        datapoints.extend(build_step_datapoints(ex, fmt=args.format))
    n_tool = sum(1 for d in datapoints if d["gold_type"] == "tool")
    n_say = len(datapoints) - n_tool
    print(f"Built {len(datapoints)} step-level datapoints (tool={n_tool}, say={n_say})",
          file=sys.stderr)
    if args.format == "native":
        missing = sum(1 for d in datapoints if not d.get("tools"))
        if missing:
            print(f"  WARNING: {missing} datapoint(s) have no resolved tool schema -- "
                  f"check _meta.domain against data/scenario/<domain>/tools.json", file=sys.stderr)

    pred_path = Path(args.predictions) if args.predictions else None
    if pred_path and pred_path.exists():
        print(f"Reusing existing predictions at {pred_path}", file=sys.stderr)
        records = load_jsonl(pred_path)
    else:
        print(f"Calling Fireworks model={args.model} format={args.format} "
              f"concurrency={args.concurrency} ...", file=sys.stderr)
        preds = run_inference(datapoints, args.model, fmt=args.format,
                               concurrency=args.concurrency, max_tokens=args.max_tokens)
        records = [
            {
                "gold_type": dp["gold_type"],
                "gold_tool_calls": dp["gold_tool_calls"],
                "gold_response": dp["gold_response"],
                "prediction": pred,
                "format": args.format,
                "meta": dp.get("meta") or {},
            }
            for dp, pred in zip(datapoints, preds)
        ]
        if pred_path:
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved predictions -> {pred_path}", file=sys.stderr)

    summary, _ = evaluate(records, use_semantic=not args.no_semantic, fmt=args.format)

    print("\n===== L-1 STEP-LEVEL SUMMARY =====")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()
