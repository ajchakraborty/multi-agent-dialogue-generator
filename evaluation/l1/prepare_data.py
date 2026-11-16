#!/usr/bin/env python3
"""
Build the L-1 eval corpus and its stratified train/test split.

Reuses src/distribution.py's classify_output()/_pair_key() for the canonical,
deduplicated (file, domain, path_type) mapping -- NOT scripts/data_conversion.py's
resolve_domain(), which sources path_type from the data/scenario/<domain>/<path_type>/
directory index instead of meta/task flags, and doesn't dedupe v2/v3 reruns of the
same (scenario_id, persona_id) pair. Confirmed against
`python3 src/distribution.py --report --all-domains`: 5,643 total
(happy 1,888 / unhappy 1,956 / impossible 1,799).

Turn conversion itself (messages -> APIGen `conversations` list, tool schema
resolution) reuses scripts/data_conversion.py's build_entry()/ToolsRegistry as-is.

Usage:
    python evaluation/l1/prepare_data.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import distribution as dist_mod  # noqa: E402
from data_conversion import ToolsRegistry, build_entry, load_json  # noqa: E402

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "data"
TOOLS_ROOT = REPO / "data/scenario"


def collect_canonical_files():
    """-> [(path, domain, path_type), ...], deduped by (scenario_id, persona_id)
    exactly as src/distribution.py's count_existing() does."""
    seen_pairs = set()
    rows = []
    dup_skipped = 0
    for vdir in dist_mod.FLAT_VALID_DIRS:
        if not vdir.exists():
            continue
        for f in sorted(vdir.glob("*.json")):
            key = dist_mod._pair_key(f)
            if key is not None:
                if key in seen_pairs:
                    dup_skipped += 1
                    continue
                seen_pairs.add(key)
            try:
                domain, path_type = dist_mod.classify_output(f)
            except Exception as e:
                print(f"  WARN: cannot classify {f}: {e}", file=sys.stderr)
                continue
            rows.append((f, domain, path_type))
    print(f"  deduped {dup_skipped} rerun scenario+persona pair(s)", file=sys.stderr)
    return rows


def build_dataset(rows):
    registry = ToolsRegistry(TOOLS_ROOT)
    entries = []
    skipped = 0
    for f, domain, path_type in rows:
        convo = load_json(f)
        if not convo or "messages" not in convo:
            skipped += 1
            continue
        tools_str = registry.get(domain)
        if tools_str is None:
            skipped += 1
            continue
        entry = build_entry(convo, tools_str, reasoning=False, source_file=f,
                             include_meta=True, domain=domain, path_type=path_type)
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)
    print(f"  built {len(entries)} entries, skipped {skipped}", file=sys.stderr)
    return entries


def stratified_split(entries, seed=SEED):
    by_pt = defaultdict(list)
    for e in entries:
        by_pt[e["_meta"]["path_type"]].append(e)

    rng = random.Random(seed)
    train, test = [], []
    for pt in sorted(by_pt):
        items = list(by_pt[pt])
        rng.shuffle(items)
        n_test = round(len(items) / 3)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


def pt_counts(entries):
    c = defaultdict(int)
    for e in entries:
        c[e["_meta"]["path_type"]] += 1
    return dict(sorted(c.items()))


def domain_pt_counts(entries):
    c = defaultdict(lambda: defaultdict(int))
    for e in entries:
        m = e["_meta"]
        c[m["domain"]][m["path_type"]] += 1
    return {d: dict(sorted(v.items())) for d, v in sorted(c.items())}


def write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    print("Collecting canonical (deduped) conversation files from v2+v3 ...",
          file=sys.stderr)
    rows = collect_canonical_files()
    raw_by_pt = defaultdict(int)
    for _, _, pt in rows:
        raw_by_pt[pt] += 1
    print(f"  canonical total: {len(rows)}  by path_type: {dict(sorted(raw_by_pt.items()))}",
          file=sys.stderr)

    entries = build_dataset(rows)

    train, test = stratified_split(entries)
    print(f"  train={len(train)}  test={len(test)}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    summary = {
        "seed": SEED,
        "train": {"total": len(train), "by_path_type": pt_counts(train),
                   "by_domain_path_type": domain_pt_counts(train)},
        "test": {"total": len(test), "by_path_type": pt_counts(test),
                  "by_domain_path_type": domain_pt_counts(test)},
    }
    with open(OUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SPLIT SUMMARY =====")
    print(json.dumps({"train": summary["train"]["by_path_type"],
                       "test": summary["test"]["by_path_type"]}, indent=2))
    print(f"\nWrote {OUT_DIR / 'train.jsonl'} ({len(train)})")
    print(f"Wrote {OUT_DIR / 'test.jsonl'} ({len(test)})")
    print(f"Wrote {OUT_DIR / 'split_summary.json'}")


if __name__ == "__main__":
    main()
