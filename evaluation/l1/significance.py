#!/usr/bin/env python3
"""
Paired bootstrap significance testing between two (or more) L1 eval runs on the
same evaluation/l1/data/test.jsonl -> step-level datapoints. Datapoint i means the
same gold step for every model (build_step_datapoints is deterministic over the
same test file), so per-datapoint scores are directly paired across models --
no need to match on content, just index.

For each metric, restricts to indices where the metric is defined (non-None) --
tool_f1/params_match_full only exist for gold_type=="tool" steps, final_answer_f1/
final_answer_sim only for gold_type=="say" steps. This mask is identical across
models since gold_type depends on the dataset, not the prediction.

Paired bootstrap: resample datapoint indices with replacement B times, take the
mean of (score_A - score_B) under each resample -> null-free two-sided p-value is
2 * min(P(mean_diff <= 0), P(mean_diff >= 0)) and a 95% CI is the 2.5/97.5
percentiles of the resampled mean-diff distribution.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRIC_KEYS = ["action_recall", "tool_f1", "params_match_full",
               "final_answer_f1", "final_answer_sim"]

MODELS = {
    "base": "base_qwen3_4b_preds.jsonl",
    "apigen": "ft_qwen3_4b_apigen_preds.jsonl",
    "happy": "ft_qwen3_4b_happytrain_preds.jsonl",
    "unhappy": "ft_qwen3_4b_unhappytrain_preds.jsonl",
    "madshu": "ft_qwen3_4b_madshu_preds.jsonl",
    "madsfull": "ft_qwen3_4b_madsfull_preds.jsonl",
    "mix": "ft_qwen3_4b_mix_preds.jsonl",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_for(name, cache_dir):
    cache_path = cache_dir / f"{name}_rows.json"
    if cache_path.exists():
        print(f"  reusing cached rows for {name}", file=sys.stderr)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    print(f"  scoring {name} ...", file=sys.stderr)
    records = load_jsonl(RESULTS_DIR / MODELS[name])
    _, rows = evaluate(records, use_semantic=True, fmt="native")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return rows


def paired_bootstrap(a_vals, b_vals, n_boot=10000, seed=0):
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    n = len(a)
    diffs = a - b
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    p_le = float(np.mean(boot_means <= 0))
    p_ge = float(np.mean(boot_means >= 0))
    p_value = min(1.0, 2 * min(p_le, p_ge))
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": observed,
        "ci95": [float(ci_lo), float(ci_hi)],
        "p_value": p_value,
        "significant": bool(ci_lo > 0 or ci_hi < 0),
    }


def compare(rows_a, rows_b, path_type=None, n_boot=10000, seed=0):
    out = {}
    for key in METRIC_KEYS:
        a_vals, b_vals = [], []
        for ra, rb in zip(rows_a, rows_b):
            if path_type and (ra.get("_meta") or {}).get("path_type") != path_type:
                continue
            va, vb = ra.get(key), rb.get(key)
            if va is None or vb is None:
                continue
            a_vals.append(va)
            b_vals.append(vb)
        if not a_vals:
            out[key] = None
            continue
        out[key] = paired_bootstrap(a_vals, b_vals, n_boot=n_boot, seed=seed)
    return out


def main():
    cache_dir = RESULTS_DIR / "sig_cache"
    rows = {name: rows_for(name, cache_dir) for name in MODELS}

    pairs = [
        ("happy", "unhappy"),
        ("happy", "base"), ("unhappy", "base"), ("apigen", "base"),
        ("madshu", "base"), ("madsfull", "base"), ("mix", "base"),
        ("happy", "apigen"), ("unhappy", "apigen"),
        ("happy", "madshu"), ("unhappy", "madshu"),
        ("mix", "madshu"), ("mix", "madsfull"),
        ("happy", "mix"), ("unhappy", "mix"),
        ("madshu", "madsfull"),
    ]

    all_results = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        print(f"comparing {key} (overall) ...", file=sys.stderr)
        all_results[key] = {"overall": compare(rows[a], rows[b])}

    for a, b in [("happy", "unhappy")]:
        for pt in ("happy", "unhappy", "impossible"):
            key = f"{a}_vs_{b}"
            print(f"comparing {key} ({pt}) ...", file=sys.stderr)
            all_results[key][pt] = compare(rows[a], rows[b], path_type=pt)

    out_path = RESULTS_DIR / "significance_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
