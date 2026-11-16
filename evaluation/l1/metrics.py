#!/usr/bin/env python3
"""
Step-level L-1 metrics, matching the paper's per-action definition:

    action_recall always; then, conditioned on the GOLD action's type:
        tool  -> tool_f1 (set-based name match) + params_match_full (exact args match)
        say   -> final_answer_f1 (token F1) + final_answer_sim (embedding cosine)

Unlike evaluation/apigen_L1.py::evaluate (which aggregates per human turn and skips
tool_name_acc/params_match_full to `None` -- excluded from the average -- whenever
the model predicted zero tool calls), every tool-type and say-type datapoint here is
scored unconditionally: a model that predicts zero tool calls when one was required
scores 0, not "not counted".
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "evaluation"))

from apigen_L1 import extract_pred, normalize_params, token_f1  # noqa: E402

METRIC_KEYS = ["action_recall", "tool_f1", "params_match_full",
               "final_answer_f1", "final_answer_sim"]


def _tool_f1(gold_calls, pred_calls):
    """Set-based (multiset) precision/recall over tool names. Reduces to a plain
    correct/incorrect indicator for the single-gold-call case, which is every
    tool-type step in this dataset, but stays meaningful if a model over- or
    under-generates calls in one completion."""
    gold_names = [c["name"] for c in gold_calls]
    pred_names = [c["name"] for c in pred_calls]
    if not gold_names and not pred_names:
        return 1.0
    if not pred_names or not gold_names:
        return 0.0

    remaining = list(pred_names)
    matched = 0
    for n in gold_names:
        if n in remaining:
            remaining.remove(n)
            matched += 1

    precision = matched / len(pred_names)
    recall = matched / len(gold_names)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _params_match_full(gold_calls, pred_calls):
    """1.0 iff every gold call has an exact name+params counterpart among the
    predicted calls (position-independent, each predicted call usable once),
    else 0."""
    if not gold_calls:
        return 1.0 if not pred_calls else 0.0

    remaining = list(pred_calls)
    for g in gold_calls:
        gp = normalize_params(g["params"])
        hit = None
        for p in remaining:
            if p["name"] == g["name"] and normalize_params(p["params"]) == gp:
                hit = p
                break
        if hit is None:
            return 0.0
        remaining.remove(hit)
    return 1.0


def evaluate(predictions, use_semantic=True, fmt="native"):
    if use_semantic:
        import torch.nn.functional as F
        from sentence_transformers import SentenceTransformer

        sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        def semantic_similarity(a, b):
            if not a.strip() or not b.strip():
                return 0.0
            ea = sim_model.encode(a, convert_to_tensor=True)
            eb = sim_model.encode(b, convert_to_tensor=True)
            return float(F.cosine_similarity(ea, eb, dim=0).item())
    else:
        def semantic_similarity(a, b):
            return None

    results = []
    for item in predictions:
        gold_type = item["gold_type"]
        gold_calls = item.get("gold_tool_calls") or []
        gold_resp = item.get("gold_response") or ""
        pred_calls, pred_resp = extract_pred(item["prediction"], item.get("format", fmt))
        pred_type = "tool" if pred_calls else ("say" if pred_resp else None)

        row = {
            "action_recall": 1.0 if pred_type == gold_type else 0.0,
            "tool_f1": None,
            "params_match_full": None,
            "final_answer_f1": None,
            "final_answer_sim": None,
            "_meta": item.get("meta") or {},
        }

        if gold_type == "tool":
            row["tool_f1"] = _tool_f1(gold_calls, pred_calls)
            row["params_match_full"] = _params_match_full(gold_calls, pred_calls)
        else:
            resp = pred_resp or ""
            row["final_answer_f1"] = token_f1(resp, gold_resp) if resp else 0.0
            row["final_answer_sim"] = semantic_similarity(resp, gold_resp) if resp else 0.0

        results.append(row)

    def avg_over(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {k: avg_over(results, k) for k in METRIC_KEYS}
    summary["n_datapoints"] = len(results)
    summary["n_tool_steps"] = sum(1 for r in results if r["tool_f1"] is not None)
    summary["n_say_steps"] = sum(1 for r in results if r["final_answer_f1"] is not None)

    buckets = defaultdict(list)
    for r in results:
        cell = (r["_meta"] or {}).get("path_type")
        if cell:
            buckets[cell].append(r)
    if buckets:
        summary["by_path_type"] = {
            cell: {**{k: avg_over(rows, k) for k in METRIC_KEYS},
                   "n_datapoints": len(rows),
                   "n_tool_steps": sum(1 for r in rows if r["tool_f1"] is not None),
                   "n_say_steps": sum(1 for r in rows if r["final_answer_f1"] is not None)}
            for cell, rows in sorted(buckets.items())
        }

    return summary, results
