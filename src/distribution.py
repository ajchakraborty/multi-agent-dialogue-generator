# """
# Scans data/valid_outputs/v2 and reports per-domain, per-use-case
# path type distribution. Shows gap to target and flags skew.
# Run this before every orchestrate.py batch to keep scaling balanced.
# """
# import json
# from pathlib import Path
# from collections import defaultdict

# V2_DIR   = Path("data/valid_outputs/v2")
# TARGET   = 833          # per domain
# PER_TYPE = TARGET // 3  # ~278 per path type per domain
# SKEW_TOL = 15           # % tolerance before flagging imbalance

# PREFIX_MAP   = {"ca":"calendar_assistant","os":"online_shopping","tr":"travel",
#                 "ba":"banking","rb":"restaurant_booking","hs":"home_services"}
# USE_CASE_MAP = {"oe":"open_ended","rm":"reschedule_meeting","sm":"schedule_meeting",
#                 "ro":"return_order","co":"cancel_order","bf":"book_flight",
#                 "sf":"search_flights","cb":"cancel_booking","mb":"make_booking",
#                 "pb":"pay_bill","cb2":"check_balance","tf":"transfer_funds",
#                 "mr":"make_reservation","mo":"modify_reservation","cr":"cancel_reservation",
#                 "bc":"book_cleaner","sr":"schedule_repair"}

# ALL_DOMAINS = ["calendar_assistant","online_shopping","travel",
#                "banking","restaurant_booking","home_services"]

# def classify(f: Path) -> dict:
#     d      = json.load(open(f))
#     meta   = d.get("meta", {})
#     config = d.get("config", {})
#     task   = config.get("task", {})

#     domain   = meta.get("domain_id") or meta.get("domain")
#     use_case = meta.get("use_case")
#     scenario = config.get("scenario_name", "")

#     if not domain or not use_case:
#         parts = scenario.split("_")
#         if len(parts) >= 2:
#             domain   = domain   or PREFIX_MAP.get(parts[0], parts[0])
#             use_case = use_case or USE_CASE_MAP.get(parts[1], parts[1])

#     if not domain:
#         stem  = f.stem.split("__")[1] if "__" in f.stem else f.stem
#         parts = stem.split("_")
#         domain   = PREFIX_MAP.get(parts[0], parts[0])
#         use_case = USE_CASE_MAP.get(parts[1], parts[1]) if len(parts) > 1 else "unknown"

#     impossible = task.get("impossible", False)
#     fallback   = task.get("fallback_behavior", "")

#     # Previous logic treated any injected behavior (user OR tool) as 'unhappy'.
#     # Change to explicitly separate user vs tool injected behaviors so that:
#     #  - impossible: explicit impossible flag OR tool_agent injected behaviors
#     #  - unhappy: user_agent injected behaviors OR fallback_behavior (but not impossible)
#     #  - happy: none of the above
#     ua_behaviors = config.get("user_agent_config", {}).get("injected_behaviors", [])
#     ta_behaviors = config.get("tool_agent_config", {}).get("injected_behaviors", [])

#     def _has_instructions(beh_list):
#         return any((b.get("instructions") or "").strip() for b in beh_list)

#     ua_injected = _has_instructions(ua_behaviors)
#     ta_injected = _has_instructions(ta_behaviors)

#     if impossible or ta_injected:
#         path_type = "impossible"
#     elif fallback or ua_injected:
#         path_type = "unhappy"
#     else:
#         path_type = "happy"
#     persona    = config.get("persona", {}).get("id") or meta.get("persona_id", "?")

#     return {"domain": domain, "use_case": use_case,
#             "path_type": path_type, "persona": persona,
#             "turns": len(d.get("messages", []))}


# def main():
#     files = sorted(V2_DIR.glob("*.json"))
#     stats = defaultdict(lambda: defaultdict(
#         lambda: {"happy": 0, "unhappy": 0, "impossible": 0, "personas": set()}))

#     for f in files:
#         try:
#             row = classify(f)
#             bucket = stats[row["domain"]][row["use_case"]]
#             bucket[row["path_type"]] += 1
#             bucket["personas"].add(row["persona"])
#         except Exception as e:
#             print(f"  WARN: {f.name} — {e}")

#     # ── Per-domain per-use-case table ──────────────────────────────────────
#     print(f"\n{'='*80}")
#     print(f"  VALID OUTPUTS v2  —  DISTRIBUTION REPORT  ({len(files)} total files)")
#     print(f"{'='*80}")

#     domain_totals = {}
#     for domain in ALL_DOMAINS:
#         use_cases = stats.get(domain, {})
#         d_h = sum(v["happy"]      for v in use_cases.values())
#         d_u = sum(v["unhappy"]    for v in use_cases.values())
#         d_i = sum(v["impossible"] for v in use_cases.values())
#         d_t = d_h + d_u + d_i
#         domain_totals[domain] = {"h": d_h, "u": d_u, "i": d_i, "t": d_t}

#         status = "✅" if d_t >= TARGET else f"❌  gap={TARGET - d_t}"
#         print(f"\n  DOMAIN: {domain:<25}  total={d_t:>4}  {status}")
#         print(f"  {'USE CASE':<22} {'H':>5} {'U':>5} {'I':>5} {'TOT':>5} "
#               f"{'H-gap':>7} {'U-gap':>7} {'I-gap':>7}")
#         print(f"  {'-'*68}")

#         for uc in sorted(use_cases):
#             v   = use_cases[uc]
#             h,u,i = v["happy"], v["unhappy"], v["impossible"]
#             t   = h + u + i
#             hg  = max(0, PER_TYPE - h)
#             ug  = max(0, PER_TYPE - u)
#             ig  = max(0, PER_TYPE - i)
#             print(f"  {uc:<22} {h:>5} {u:>5} {i:>5} {t:>5} "
#                   f"{hg:>7} {ug:>7} {ig:>7}")

#         if not use_cases:
#             print(f"  (no conversations yet)")

#     # ── Balance check ──────────────────────────────────────────────────────
#     print(f"\n{'='*80}")
#     print(f"  BALANCE CHECK  (target: ~{PER_TYPE} each | skew tolerance: ±{SKEW_TOL}%)")
#     print(f"{'='*80}")
#     print(f"  {'Domain':<25} {'H%':>5} {'U%':>5} {'I%':>5}  {'Status'}")
#     print(f"  {'-'*55}")

#     for domain in ALL_DOMAINS:
#         t = domain_totals[domain]["t"]
#         if t == 0:
#             print(f"  {domain:<25}   —     —     —    ❌ NO DATA")
#             continue
#         h  = domain_totals[domain]["h"]
#         u  = domain_totals[domain]["u"]
#         i  = domain_totals[domain]["i"]
#         hp = h/t*100; up = u/t*100; ip = i/t*100
#         ok = (abs(hp-33) < SKEW_TOL and
#               abs(up-33) < SKEW_TOL and
#               abs(ip-33) < SKEW_TOL)
#         flag = "✅ balanced" if ok else "❌ SKEWED"
#         print(f"  {domain:<25} {hp:>4.0f}% {up:>4.0f}% {ip:>4.0f}%  {flag}")

#     # ── Progress bar ───────────────────────────────────────────────────────
#     print(f"\n{'='*80}")
#     print(f"  PROGRESS TO TARGET ({TARGET}/domain)")
#     print(f"{'='*80}")
#     for domain in ALL_DOMAINS:
#         t    = domain_totals[domain]["t"]
#         pct  = t / TARGET * 100
#         done = int(pct / 2.5)
#         bar  = "█" * done + "░" * (40 - done)
#         print(f"  {domain:<25} {t:>4}/{TARGET} [{bar}] {pct:>5.1f}%")

#     grand = sum(v["t"] for v in domain_totals.values())
#     gtotal = TARGET * len(ALL_DOMAINS)
#     gpct   = grand / gtotal * 100
#     print(f"\n  {'GRAND TOTAL':<25} {grand:>4}/{gtotal}  {gpct:.1f}% complete")

#     # ── What to generate next (biggest gap first) ──────────────────────────
#     print(f"\n{'='*80}")
#     print(f"  NEXT BATCH PRIORITY  (generate these to stay balanced)")
#     print(f"{'='*80}")
#     queue = []
#     for domain in ALL_DOMAINS:
#         for uc, v in stats.get(domain, {}).items():
#             for pt in ("happy", "unhappy", "impossible"):
#                 gap = max(0, PER_TYPE - v[pt])
#                 if gap > 0:
#                     queue.append((gap, domain, uc, pt))
#     # Also add missing domains
#     for domain in ALL_DOMAINS:
#         if domain not in stats:
#             queue.append((PER_TYPE, domain, "ALL USE CASES", "all types"))

#     queue.sort(reverse=True)
#     for gap, domain, uc, pt in queue[:15]:
#         print(f"  gap={gap:>4}  {domain:<25} {uc:<22} {pt}")


# if __name__ == "__main__":
#     main()

# """
# Scenario template distribution report.

# Scans data/domains/**.json (excluding tools.json) and reports per-domain, per-use-case
# counts of scenario templates by path type:
#   - happy: user_agent.injected_behaviors empty AND tool_agent.injected_behaviors empty
#   - unhappy: user_agent.injected_behaviors non-empty (regardless of tool)
#   - impossible: user empty AND tool non-empty

# This complements src/distribution.py which scans validated outputs (data/valid_outputs/v2).
# """

# import json
# from pathlib import Path
# from collections import defaultdict

# DOMAINS_DIR = Path("data/domains")

# ALL_DOMAINS = [
#     "calendar_assistant",
#     "online_shopping",
#     "travel",
#     "banking",
#     "restaurant_booking",
#     "home_services",
# ]

# def _has_instructions(beh_list) -> bool:
#     if not isinstance(beh_list, list):
#         return False
#     for b in beh_list:
#         if not isinstance(b, dict):
#             continue
#         if (b.get("instructions") or "").strip():
#             return True
#     return False

# def classify_scenario(scenario_path: Path) -> dict:
#     d = json.load(open(scenario_path, "r", encoding="utf-8"))

#     # Prefer folder structure for domain/use_case because metadata can be inconsistent
#     # data/domains/<domain>/<use_case>/<file>.json
#     parts = scenario_path.parts
#     try:
#         idx = parts.index("domains")
#         domain = parts[idx + 1]
#         use_case = parts[idx + 2]
#     except Exception:
#         domain = d.get("metadata", {}).get("domain") or "unknown"
#         use_case = d.get("metadata", {}).get("use_case") or "unknown"

#     ua = d.get("user_agent", {}).get("injected_behaviors", [])
#     ta = d.get("tool_agent", {}).get("injected_behaviors", [])

#     ua_has = _has_instructions(ua)
#     ta_has = _has_instructions(ta)

#     if ua_has:
#         path_type = "unhappy"
#     elif ta_has:
#         path_type = "impossible"
#     else:
#         path_type = "happy"

#     return {
#         "domain": domain,
#         "use_case": use_case,
#         "path_type": path_type,
#         "file": str(scenario_path),
#     }

# def main():
#     files = sorted([p for p in DOMAINS_DIR.rglob("*.json") if p.name != "tools.json"])
#     stats = defaultdict(lambda: defaultdict(lambda: {"happy": 0, "unhappy": 0, "impossible": 0}))
#     domain_totals = defaultdict(lambda: {"happy": 0, "unhappy": 0, "impossible": 0, "total": 0})

#     for f in files:
#         try:
#             row = classify_scenario(f)
#             stats[row["domain"]][row["use_case"]][row["path_type"]] += 1
#             domain_totals[row["domain"]][row["path_type"]] += 1
#             domain_totals[row["domain"]]["total"] += 1
#         except Exception as e:
#             print(f"WARN: failed to parse {f}: {e}")

#     print("\n" + "=" * 90)
#     print(f"SCENARIO TEMPLATE DISTRIBUTION  (data/domains)   total_files={len(files)}")
#     print("=" * 90)

#     # Per-domain summary
#     print("\nPER-DOMAIN SUMMARY")
#     print(f"{'Domain':<22} {'Happy':>6} {'Unhappy':>8} {'Impossible':>11} {'Total':>7}")
#     print("-" * 60)
#     for domain in ALL_DOMAINS:
#         v = domain_totals.get(domain, {"happy": 0, "unhappy": 0, "impossible": 0, "total": 0})
#         print(f"{domain:<22} {v['happy']:>6} {v['unhappy']:>8} {v['impossible']:>11} {v['total']:>7}")

#     # Unknown domains (if any)
#     for domain in sorted(domain_totals.keys()):
#         if domain not in ALL_DOMAINS:
#             v = domain_totals[domain]
#             print(f"{domain:<22} {v['happy']:>6} {v['unhappy']:>8} {v['impossible']:>11} {v['total']:>7}")

#     # Per-domain breakdown
#     print("\n" + "=" * 90)
#     print("PER-DOMAIN / PER-USE-CASE BREAKDOWN")
#     print("=" * 90)
#     for domain in ALL_DOMAINS:
#         use_cases = stats.get(domain, {})
#         if not use_cases:
#             print(f"\nDOMAIN: {domain} (no scenario files found)")
#             continue

#         print(f"\nDOMAIN: {domain}")
#         print(f"  {'Use case':<28} {'Happy':>6} {'Unhappy':>8} {'Impossible':>11} {'Total':>7}")
#         print(f"  {'-'*70}")
#         for uc in sorted(use_cases.keys()):
#             v = use_cases[uc]
#             total = v["happy"] + v["unhappy"] + v["impossible"]
#             print(f"  {uc:<28} {v['happy']:>6} {v['unhappy']:>8} {v['impossible']:>11} {total:>7}")

#     print("\nDONE.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Builds a balanced batch of scenario files for orchestrate.py.

Target: TOTAL_PER_CELL conversations per (domain, path_type) cell.
Reads existing counts from valid outputs, computes gaps, and cycles
through templates in each cell to fill the gap.

Usage:
    uv run src/distribution.py --report                # gap table only
    uv run src/distribution.py --batch-size 300 > batch.txt
    nohup python src/orchestrate.py $(cat batch.txt) --run-eval > gen.log 2>&1 &
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────
DOMAINS_DIR = Path("data/scenario")
VALID_DIR = Path("data/valid_outputs/v2")
TOTAL_PER_CELL = 300  # 500 x 6 domains x 3 types = 9000
PATH_TYPES = ["happy", "unhappy", "impossible"]

ALL_DOMAINS = [
    "calendar_assistant",
    "online_shopping",
    "travel",
    "banking",
    "restaurant_booking",
    "home_services",
]

PREFIX_MAP = {"ca": "calendar_assistant", "os": "online_shopping", "tr": "travel",
              "ba": "banking", "rb": "restaurant_booking", "hs": "home_services"}


# ── Template classification (data/domains) ────────────────────────────────
def _has_instructions(beh_list) -> bool:
    if not isinstance(beh_list, list):
        return False
    for b in beh_list:
        if isinstance(b, dict) and (b.get("instructions") or "").strip():
            return True
    return False


def classify_scenario(scenario_path: Path) -> dict:
    """Classify a scenario TEMPLATE by path type, using folder structure for domain."""
    d = json.load(open(scenario_path, "r", encoding="utf-8"))

    parts = scenario_path.parts
    try:
        idx = parts.index("domains")
        domain = parts[idx + 1]
        use_case = parts[idx + 2]
    except (ValueError, IndexError):
        domain = d.get("metadata", {}).get("domain") or "unknown"
        use_case = d.get("metadata", {}).get("use_case") or "unknown"

    ua_has = _has_instructions(d.get("user_agent", {}).get("injected_behaviors", []))
    ta_has = _has_instructions(d.get("tool_agent", {}).get("injected_behaviors", []))

    if ua_has:
        path_type = "unhappy"
    elif ta_has:
        path_type = "impossible"
    else:
        path_type = "happy"

    return {"domain": domain, "use_case": use_case,
            "path_type": path_type, "file": str(scenario_path)}


# ── Output classification (data/valid_outputs/v2) ─────────────────────────
def classify_output(f: Path) -> tuple:
    """Return (domain, path_type) for a generated conversation output."""
    d = json.load(open(f, "r", encoding="utf-8"))
    meta = d.get("meta", {})
    config = d.get("config", {})
    task = config.get("task", {})

    domain = meta.get("domain_id") or meta.get("domain")
    if not domain:
        scenario = config.get("scenario_name", "") or (
            f.stem.split("__")[1] if "__" in f.stem else f.stem)
        prefix = scenario.split("_")[0] if scenario else ""
        domain = PREFIX_MAP.get(prefix, prefix or "unknown")

    pt = meta.get("path_type")
    if not pt:
        ua = config.get("user_agent_config", {}).get("injected_behaviors", [])
        ta = config.get("tool_agent_config", {}).get("injected_behaviors", [])
        impossible = task.get("impossible", False)
        fallback = task.get("fallback_behavior", "")
        if impossible or _has_instructions(ta):
            pt = "impossible"
        elif fallback or _has_instructions(ua):
            pt = "unhappy"
        else:
            pt = "happy"
    return domain, pt


def count_existing() -> dict:
    counts = defaultdict(int)
    if not VALID_DIR.exists():
        return counts
    for f in VALID_DIR.glob("*.json"):
        try:
            domain, pt = classify_output(f)
            counts[(domain, pt)] += 1
        except Exception as e:
            print(f"WARN: skipping {f.name}: {e}", file=sys.stderr)
    return counts


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=300,
                    help="Total scenarios in this batch (allocated by gap size)")
    ap.add_argument("--report", action="store_true", help="Only print gap table")
    args = ap.parse_args()

    # 1. Bucket templates by (domain, path_type)
    templates = defaultdict(list)
    for f in sorted(DOMAINS_DIR.rglob("*.json")):
        if f.name in ("tools.json", "catalog.json"):
            continue
        try:
            row = classify_scenario(f)
            templates[(row["domain"], row["path_type"])].append(f)
        except Exception as e:
            print(f"WARN: failed to parse template {f}: {e}", file=sys.stderr)

    existing = count_existing()

    # 2. Compute gaps per cell
    cells = [(d, pt) for d in ALL_DOMAINS for pt in PATH_TYPES]
    gaps = {c: max(0, TOTAL_PER_CELL - existing.get(c, 0)) for c in cells}

    print("=" * 62, file=sys.stderr)
    print(f"{'Domain':<22}{'Type':<12}{'Have':>6}{'Gap':>6}{'Templates':>11}",
          file=sys.stderr)
    print("-" * 62, file=sys.stderr)
    for c in cells:
        print(f"{c[0]:<22}{c[1]:<12}{existing.get(c, 0):>6}{gaps[c]:>6}"
              f"{len(templates[c]):>11}", file=sys.stderr)
        if gaps[c] > 0 and not templates[c]:
            print(f"  !! No templates for {c} — create some first", file=sys.stderr)
    total_have = sum(existing.get(c, 0) for c in cells)
    total_gap = sum(gaps.values())
    print("-" * 62, file=sys.stderr)
    print(f"{'TOTAL':<34}{total_have:>6}{total_gap:>6}", file=sys.stderr)

    if args.report:
        return

    # 3. Allocate this batch proportionally to gaps
    if total_gap == 0:
        print("All cells complete.", file=sys.stderr)
        return
    batch = []
    per_cell = {c: round(args.batch_size * gaps[c] / total_gap) for c in cells}
    for c, n in per_cell.items():
        if n == 0 or not templates[c]:
            continue
        cyc = itertools.cycle(templates[c])
        batch.extend(str(next(cyc)) for _ in range(n))

    # 4. Emit file list to stdout (one per line)
    for p in batch:
        print(p)


if __name__ == "__main__":
    main()