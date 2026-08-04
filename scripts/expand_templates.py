#!/usr/bin/env python3
"""
Derive scenario templates to fill under-populated (domain, path_type) cells.

Reads:  data/scenario/<domain>/<src_type>/*.json
Writes: data/scenario/<domain>/<to>/*.json  (same depth, so "../tools.json" resolves)

type_ids come from data/catalogs/behavior_types.json;
instruction text from data/catalogs/behavior_instructions.json.

Usage:
    uv run scripts/derive_scenarios.py --domain online_shopping --to impossible --per-base 4 --dry-run
    uv run scripts/derive_scenarios.py --domain online_shopping --to impossible --per-base 4
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO / "data/scenario"
TYPES = json.loads((REPO / "data/catalogs/behavior_types.json").read_text())["types"]
INSTR = json.loads((REPO / "data/catalogs/behavior_instructions.json").read_text())
KNOWN = {t["type_id"] for t in TYPES}

SEARCH_HINTS = ("list_", "search", "find", "lookup", "get_", "check_",
                "available", "recent", "query")


def has_instructions(behaviors) -> bool:
    return isinstance(behaviors, list) and any(
        isinstance(b, dict) and (b.get("instructions") or "").strip() for b in behaviors
    )


def path_type_of(d: dict) -> str:
    if d.get("task", {}).get("impossible") or \
       has_instructions(d.get("tool_agent", {}).get("injected_behaviors")):
        return "impossible"
    if has_instructions(d.get("user_agent", {}).get("injected_behaviors")):
        return "unhappy"
    return "happy"


def capabilities(d: dict) -> set:
    caps = set()
    if d.get("task", {}).get("success_criteria", {}).get("action"):
        caps.add("action_step")
    blob = json.dumps(d).lower()
    if any(h in blob for h in SEARCH_HINTS):
        caps.add("search_step")
    return caps


def base_id_of(d: dict, path: Path) -> str:
    return (d.get("metadata") or {}).get("scenario_id") or path.stem


def _strip_persona(v: dict) -> dict:
    """Derived templates rotate personas via --persona-id."""
    v.pop("persona", None)
    return v


def derive_impossible(d, base_id, fm, n):
    v = deepcopy(d)
    sid = f"{base_id}_imp{n:02d}"
    meta = v.setdefault("metadata", {})
    meta.update({"scenario_id": sid, "derived_from": base_id,
                 "failure_mode": fm["type_id"]})

    task = v.setdefault("task", {})
    task["impossible"] = True
    task["success_criteria"] = {
        "action": "graceful_decline",
        "notes": fm["expected_assistant"] +
                 " Success requires NO fabricated confirmation and a clear explanation to the user.",
    }
    v.setdefault("user_agent", {})["injected_behaviors"] = []
    ta = v.setdefault("tool_agent", {})
    ta["injected_behaviors"] = [
        {"type_id": fm["type_id"], "instructions": fm["instructions"]}
    ]
    return sid, _strip_persona(v)


def derive_unhappy(d, base_id, beh, n):
    v = deepcopy(d)
    sid = f"{base_id}_unh{n:02d}"
    meta = v.setdefault("metadata", {})
    meta.update({"scenario_id": sid, "derived_from": base_id,
                 "behavior": beh["type_id"]})

    v.setdefault("task", {})["impossible"] = False
    v.setdefault("user_agent", {})["injected_behaviors"] = [
        {"type_id": beh["type_id"], "instructions": beh["instructions"]}
    ]
    v.setdefault("tool_agent", {}).setdefault("injected_behaviors", [])
    return sid, _strip_persona(v)


def derive_happy(d, base_id):
    v = deepcopy(d)
    sid = f"{base_id}_hap01"
    meta = v.setdefault("metadata", {})
    meta.update({"scenario_id": sid, "derived_from": base_id})
    meta.pop("failure_mode", None)
    meta.pop("behavior", None)

    v.setdefault("task", {})["impossible"] = False
    v.setdefault("user_agent", {})["injected_behaviors"] = []
    v.setdefault("tool_agent", {})["injected_behaviors"] = []
    return sid, _strip_persona(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--to", required=True, choices=["impossible", "unhappy", "happy"])
    ap.add_argument("--per-base", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # fail fast if the companion catalog drifts from behavior_types.json
    for group in ("impossible", "unhappy"):
        for e in INSTR.get(group, []):
            if e["type_id"] not in KNOWN:
                raise SystemExit(f"Unknown type_id '{e['type_id']}' "
                                 f"— add it to data/catalogs/behavior_types.json first")

    root = SCENARIO_DIR / a.domain
    if not root.exists():
        raise SystemExit(f"Not found: {root}  (run scripts/sort_scenarios.py first)")
    if not (root / "tools.json").exists():
        print(f"WARN: {root/'tools.json'} missing — '../tools.json' will not resolve",
              file=sys.stderr)

    src_types = {"impossible": ["happy", "unhappy"],
                 "unhappy":    ["happy"],
                 "happy":      ["unhappy", "impossible"]}[a.to]

    sources = []
    for st in src_types:
        for f in sorted((root / st).glob("*.json")):
            d = json.loads(f.read_text())
            if (d.get("metadata") or {}).get("derived_from"):
                continue                     # never derive from a derivative
            sources.append((f, st, d))
    if not sources:
        raise SystemExit(f"No source templates in {root}/{{{','.join(src_types)}}}")

    out_dir, made = root / a.to, 0
    print(f"{a.domain}: {len(sources)} source(s) from {src_types} -> {a.to}\n", file=sys.stderr)

    for src, st, d in sources:
        base_id, caps = base_id_of(d, src), capabilities(d)
        if a.to == "impossible":
            pool = [f for f in INSTR["impossible"]
                    if set(f.get("requires", [])) <= caps][: a.per_base]
            results = [derive_impossible(d, base_id, fm, n) for n, fm in enumerate(pool, 1)]
        elif a.to == "unhappy":
            pool = [b for b in INSTR["unhappy"]
                    if set(b.get("requires", [])) <= caps][: a.per_base]
            results = [derive_unhappy(d, base_id, b, n) for n, b in enumerate(pool, 1)]
        else:
            results = [derive_happy(d, base_id)]

        print(f"{src.relative_to(REPO)}  (caps={sorted(caps) or 'none'})", file=sys.stderr)
        for sid, v in results:
            got = path_type_of(v)
            if got != a.to:
                print(f"  !! {sid} classified as {got}, skipped", file=sys.stderr)
                continue
            tag = v["metadata"].get("failure_mode") or v["metadata"].get("behavior") or ""
            warn = "   <-- REVIEW (from impossible)" if (a.to == "happy" and st == "impossible") else ""
            print(f"  + {a.to}/{sid}.json  {tag}{warn}", file=sys.stderr)
            if not a.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{sid}.json").write_text(
                    json.dumps(v, indent=2, ensure_ascii=False) + "\n")
            made += 1

    print(f"\n{'[dry-run] would generate' if a.dry_run else 'generated'} {made} template(s) "
          f"in {out_dir.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()