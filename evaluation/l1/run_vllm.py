# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm==0.6.6.post1",
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "transformers==4.47.1",
#     "datasets",
#     "sentence-transformers",
#     "numpy",
#     "tqdm",
# ]
# ///

"""
Local vLLM L-1 evaluation runner, for open-weight HF checkpoints that aren't a
Fireworks deployment (e.g. Salesforce/xLAM-2-3b-fc-r) -- mirrors evaluation/l1/run.py
(the Fireworks path) but for local GPU inference.

Reuses evaluation/apigen_L1.py::run_inference (the existing vLLM batched-inference
path -- tokenizer.apply_chat_template + tools=) as-is: evaluation/l1/datapoints.py's
step-level datapoints use the identical {"instruction","input","messages","tools"}
shape apigen_L1.py's own datapoints do, so no vLLM setup needs duplicating here.

Usage:
    uv run evaluation/l1/run_vllm.py --model Salesforce/xLAM-2-3b-fc-r \\
        --format native --tp 8 \\
        --predictions evaluation/l1/results/xlam_2_3b_preds.jsonl \\
        --out evaluation/l1/results/xlam_2_3b_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "evaluation"))

from datapoints import build_step_datapoints  # noqa: E402
from metrics import evaluate  # noqa: E402
from apigen_L1 import run_inference  # noqa: E402

DEFAULT_TEST_FILE = Path(__file__).resolve().parent / "data" / "test.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    ap.add_argument("--format", choices=["action", "native"], default="native")
    ap.add_argument("--tp", type=int, default=8, help="tensor_parallel_size (number of GPUs)")
    ap.add_argument("--cpu-offload-gb", type=float, default=0,
                     help="GB of weights to keep on CPU RAM (vLLM streams them to GPU "
                          "per-layer). Use when a model doesn't fit in one GPU's memory "
                          "and NCCL/tensor-parallel isn't viable in this environment.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                     help="gpu_memory_utilization passed to vLLM")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--n-conversations", type=int, default=None,
                     help="Subsample the first N conversations (smoke tests); default: all")
    ap.add_argument("--shard", default=None,
                     help="Data-parallel shard as 'i/N' (e.g. '3/8'): process only "
                          "conversations[i::N]. For splitting one model across "
                          "several single-GPU processes instead of tensor-parallel "
                          "(avoids NCCL) -- pair with CUDA_VISIBLE_DEVICES and --tp 1.")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--predictions", default=None,
                     help="Save raw per-step predictions here; reused (skipping "
                          "inference) if the file already exists")
    ap.add_argument("--out", required=True, help="Where to write the summary JSON")
    args = ap.parse_args()

    conversations = load_jsonl(args.test_file)
    if args.n_conversations:
        conversations = conversations[: args.n_conversations]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        conversations = conversations[i::n]
        print(f"Shard {i}/{n}: {len(conversations)} conversations", file=sys.stderr)
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
        print(f"Loading vLLM engine: {args.model} (tp={args.tp}) ...", file=sys.stderr)
        preds = run_inference(args.model, datapoints, args.max_new_tokens, args.tp,
                               args.max_model_len, fmt=args.format,
                               cpu_offload_gb=args.cpu_offload_gb,
                               gpu_memory_utilization=args.gpu_mem_util)
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
