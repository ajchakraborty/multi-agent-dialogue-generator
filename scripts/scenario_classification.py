#!/usr/bin/env python3
"""
Sort scenario templates from data/domains/<domain>/<use_case>/*.json
into data/scenario/<domain>/<happy|unhappy|impossible>/*.json

Classification matches src/distribution.py exactly:
  unhappy    -> user_agent.injected_behaviors has instructions
  impossible -> tool_agent.injected_behaviors has instructions (and user has none)
  happy      -> neither

Usage:
    uv run scripts/sort_scenarios.py --dry-run
    uv run scripts/sort_scenarios.py
    uv run scripts/sort_scenarios.py --domain travel
    uv run scripts/sort_scenarios.py --mode symlink     # instead of copy
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from distribution import classify_scenario, ALL_DOMAINS  # noqa: E402

DOMAINS_DIR = REPO / "data/domains"
SCENARIO_DIR = REPO / "data/scenario"
SKIP_NAMES = {"tools.json", "catalog.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", help="Only this domain (default: all)")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="Delete existing data/scenario/<domain> dirs first")
    a = ap.parse_args()

    domains = [a.domain] if a.domain else ALL_DOMAINS
    stats = defaultdict(lambda: defaultdict(int))
    collisions = []
    written = 0

    for domain in domains:
        src_root = DOMAINS_DIR / domain
        if not src_root.exists():
            print(f"WARN: no such domain dir: {src_root}", file=sys.stderr)
            continue

        if a.clean and not a.dry_run:
            shutil.rmtree(SCENARIO_DIR / domain, ignore_errors=True)

        for f in sorted(src_root.rglob("*.json")):
            if f.name in SKIP_NAMES:
                continue
            try:
                row = classify_scenario(f)
            except Exception as e:
                print(f"WARN: failed to parse {f}: {e}", file=sys.stderr)
                continue

            pt = row["path_type"]
            dest_dir = SCENARIO_DIR / domain / pt
            dest = dest_dir / f.name

            # avoid name clashes across use_case folders
            if dest.exists() or any(c[1] == dest for c in collisions):
                dest = dest_dir / f"{row['use_case']}__{f.name}"
                collisions.append((f, dest))

            stats[domain][pt] += 1
            print(f"{pt:<11} {f.relative_to(REPO)}  ->  {dest.relative_to(REPO)}")

            if not a.dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                if a.mode == "symlink":
                    if dest.is_symlink() or dest.exists():
                        dest.unlink()
                    dest.symlink_to(f.resolve())
                else:
                    shutil.copy2(f, dest)
            written += 1

    print("\n" + "=" * 58, file=sys.stderr)
    print(f"{'Domain':<22}{'Happy':>8}{'Unhappy':>10}{'Impossible':>12}", file=sys.stderr)
    print("-" * 58, file=sys.stderr)
    for domain in domains:
        s = stats[domain]
        print(f"{domain:<22}{s['happy']:>8}{s['unhappy']:>10}{s['impossible']:>12}",
              file=sys.stderr)
    print("-" * 58, file=sys.stderr)
    print(f"{'TOTAL FILES':<22}{written:>8}", file=sys.stderr)
    if collisions:
        print(f"\n{len(collisions)} filename collision(s) renamed with use_case prefix",
              file=sys.stderr)
    if a.dry_run:
        print("\n[dry-run] nothing written", file=sys.stderr)


if __name__ == "__main__":
    main()