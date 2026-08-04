#!/usr/bin/env python3
"""
Data quality assessment: MADS-Bench vs APIGen-MT-5k.

Metrics:
  1. Avg tool calls per conversation
  2. Avg turns per conversation
  3. Coherence  : Entailment Rate (EnR) + Semantic Similarity (SS)
                  between consecutive turns (Wang et al., 2025)
  4. Diversity  : Shannon entropy (H) over word frequencies +
                  Distinct-N (N=3)

APIGen is subsampled (seed-fixed) to match MADS-Bench size for fair comparison.

Requirements:
  pip install datasets sentence-transformers transformers torch numpy

Usage:
  python scripts/quality_assessment.py \
      --mads data/apigen/apigen_non_reasoning.jsonl \
      --sample-size 146 \
      --out data/analysis/quality_report.json
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mads(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with open(path) as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        else:
            entries = json.load(f)
    return entries


def load_apigen(sample_size: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/APIGen-MT-5k", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:sample_size]]


# ---------------------------------------------------------------------------
# Basic structure metrics
# ---------------------------------------------------------------------------

def conversation_turns(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    return entry.get("conversations", [])


def count_metrics(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    tool_calls, turns, human_turns, gpt_turns = [], [], [], []
    for e in entries:
        conv = conversation_turns(e)
        tool_calls.append(sum(1 for t in conv if t["from"] == "function_call"))
        # A "turn" = one human or one gpt message (dialogue turns, excludes fc/obs)
        turns.append(sum(1 for t in conv if t["from"] in ("human", "gpt")))
        human_turns.append(sum(1 for t in conv if t["from"] == "human"))
        gpt_turns.append(sum(1 for t in conv if t["from"] == "gpt"))
    return {
        "num_conversations": len(entries),
        "avg_tool_calls": float(np.mean(tool_calls)),
        "std_tool_calls": float(np.std(tool_calls)),
        "avg_turns": float(np.mean(turns)),
        "std_turns": float(np.std(turns)),
        "avg_human_turns": float(np.mean(human_turns)),
        "avg_gpt_turns": float(np.mean(gpt_turns)),
    }


# ---------------------------------------------------------------------------
# Coherence: consecutive dialogue-turn pairs (human/gpt text only)
# ---------------------------------------------------------------------------

def dialogue_texts(entry: Dict[str, Any]) -> List[str]:
    """Ordered human/gpt utterances (tool calls & observations excluded)."""
    return [t["value"] for t in conversation_turns(entry)
            if t["from"] in ("human", "gpt") and t["value"].strip()]


def consecutive_pairs(entries: List[Dict[str, Any]]) -> List[tuple]:
    pairs = []
    for e in entries:
        texts = dialogue_texts(e)
        pairs.extend(zip(texts[:-1], texts[1:]))
    return pairs


def entailment_rate(pairs: List[tuple], model_name: str, batch_size: int = 16) -> float:
    """Rate of entailment (premise=turn_i, hypothesis=turn_{i+1}) using an NLI model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    # roberta-large-mnli label order: 0=contradiction, 1=neutral, 2=entailment
    entail_idx = 2
    if hasattr(model.config, "label2id"):
        for k, v in model.config.label2id.items():
            if "entail" in k.lower():
                entail_idx = v

    n_entail = 0
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = tok([p[0] for p in batch], [p[1] for p in batch],
                      truncation=True, max_length=512, padding=True,
                      return_tensors="pt").to(device)
            preds = model(**enc).logits.argmax(dim=-1)
            n_entail += (preds == entail_idx).sum().item()
    return n_entail / max(len(pairs), 1)


def semantic_similarity(pairs: List[tuple], model_name: str) -> float:
    """Mean cosine similarity between consecutive turn embeddings."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    a = model.encode([p[0] for p in pairs], batch_size=64,
                     normalize_embeddings=True, show_progress_bar=False)
    b = model.encode([p[1] for p in pairs], batch_size=64,
                     normalize_embeddings=True, show_progress_bar=False)
    return float(np.mean(np.sum(a * b, axis=1)))


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def shannon_entropy(entries: List[Dict[str, Any]]) -> float:
    """H over the word-frequency distribution of all dialogue text."""
    counts = Counter()
    for e in entries:
        for text in dialogue_texts(e):
            counts.update(tokenize(text))
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def distinct_n(entries: List[Dict[str, Any]], n: int = 3) -> float:
    """|unique n-grams| / |total n-grams| over all dialogue text."""
    total, unique = 0, set()
    for e in entries:
        for text in dialogue_texts(e):
            toks = tokenize(text)
            grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
            total += len(grams)
            unique.update(grams)
    return len(unique) / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def assess(name: str, entries: List[Dict[str, Any]],
           nli_model: str, embed_model: str, skip_coherence: bool) -> Dict[str, Any]:
    print(f"\n=== {name} ({len(entries)} conversations) ===", file=sys.stderr)
    report = count_metrics(entries)

    print("  computing diversity...", file=sys.stderr)
    report["shannon_entropy"] = shannon_entropy(entries)
    report["distinct_3"] = distinct_n(entries, n=3)

    if not skip_coherence:
        pairs = consecutive_pairs(entries)
        report["num_consecutive_pairs"] = len(pairs)
        print(f"  computing semantic similarity over {len(pairs)} pairs...", file=sys.stderr)
        report["semantic_similarity"] = semantic_similarity(pairs, embed_model)
        print(f"  computing entailment rate over {len(pairs)} pairs...", file=sys.stderr)
        report["entailment_rate"] = entailment_rate(pairs, nli_model)
    return report


def main():
    ap = argparse.ArgumentParser(description="Quality assessment: MADS-Bench vs APIGen-MT-5k")
    ap.add_argument("--mads", default="data/MADS-bench/banking/madsbench_banking_non_reasoning.jsonl",
                    help="MADS-Bench dataset (jsonl/json, APIGen format)")
    ap.add_argument("--sample-size", type=int, default=None,
                    help="APIGen subsample size (default: match MADS size)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nli-model", default="roberta-large-mnli")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--skip-coherence", action="store_true",
                    help="Skip EnR/SS (no model downloads); structure+diversity only")
    ap.add_argument("--out", default="data/analysis/quality_report.json")
    args = ap.parse_args()

    mads = load_mads(Path(args.mads))
    sample_size = args.sample_size or len(mads)
    print(f"Loading APIGen-MT-5k (subsample={sample_size}, seed={args.seed})...", file=sys.stderr)
    apigen = load_apigen(sample_size, args.seed)

    report = {
        "config": {"mads_path": args.mads, "apigen_sample_size": sample_size,
                   "seed": args.seed, "nli_model": args.nli_model,
                   "embed_model": args.embed_model},
        "MADS-Bench": assess("MADS-Bench", mads, args.nli_model,
                             args.embed_model, args.skip_coherence),
        "APIGen-MT": assess("APIGen-MT", apigen, args.nli_model,
                            args.embed_model, args.skip_coherence),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    # Comparison table
    keys = ["num_conversations", "avg_tool_calls", "avg_turns",
            "entailment_rate", "semantic_similarity",
            "shannon_entropy", "distinct_3"]
    print(f"\n{'Metric':<25}{'MADS-Bench':>15}{'APIGen-MT':>15}")
    print("-" * 55)
    for k in keys:
        m = report["MADS-Bench"].get(k)
        a = report["APIGen-MT"].get(k)
        fm = f"{m:.4f}" if isinstance(m, float) else str(m or "-")
        fa = f"{a:.4f}" if isinstance(a, float) else str(a or "-")
        print(f"{k:<25}{fm:>15}{fa:>15}")
    print(f"\nFull report saved to: {out}")


if __name__ == "__main__":
    main()