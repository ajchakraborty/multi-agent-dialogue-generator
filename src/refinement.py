#!/usr/bin/env python3
"""Correction loop: scan failed runs, map eval reasons to agent hints,
re-simulate the same scenario+persona with hints injected, re-eval,
and let simulate.py promote passing runs to valid_outputs.

Usage:
    python3 src/refinement.py data/outputs --dry-run          # inspect only
    python3 src/refinement.py data/outputs --max-attempts 2   # fix for real
    python3 src/refinement.py data/outputs --domain banking   # filter by scenario prefix
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REJECTED_LOG = REPO_ROOT / "data" / "rejected" / "still_failing.jsonl"
VALID_OUTPUTS_ROOT = REPO_ROOT / "data" / "valid_outputs"


# ---------------------------------------------------------------------------
# Failure reason -> agent hints
# ---------------------------------------------------------------------------

def hints_from_eval(ev: dict) -> dict:
    """Convert an eval.json into guidance for the system/user agents."""
    system_hints, user_hints = [], []

    # 1. Task success -> system agent
    succ = ev.get("success") or {}
    if isinstance(succ, dict) and not succ.get("success", True):
        reason = succ.get("reason", "")
        if "impossible" in reason.lower() or "decline" in reason.lower():
            system_hints.append(
                f"A previous attempt at this exact task FAILED for this reason: "
                f"\"{reason}\". This task CANNOT be completed. You must clearly and "
                f"gracefully DECLINE: verify via tools that the request is infeasible, "
                f"explicitly tell the user it cannot be done and why, and do NOT "
                f"promise workarounds, partial actions, or future follow-ups."
            )
        else:
            system_hints.append(
                f"A previous attempt at this exact task FAILED for this reason: "
                f"\"{reason}\". You must actually EXECUTE the required tool call(s) "
                f"using <action type=\"tool\" name=\"...\">{{...}}</action>. Never "
                f"merely describe, promise, or claim to have performed an action, "
                f"and never say you cannot do something that an available tool supports."
            )

    # 2. Faithfulness -> system agent
    faith = ev.get("faithfulness") or {}
    faith_summary = faith.get("summary") or {}
    if isinstance(faith_summary, dict) and not faith_summary.get("valid", True):
        turn_reasons = []
        for t in faith.get("error_turns", []) or []:
            if isinstance(t, dict) and t.get("reason"):
                turn_reasons.append(t["reason"])
        detail = f" Specifically: {'; '.join(turn_reasons)}." if turn_reasons else ""
        system_hints.append(
            "A previous attempt stated facts that were NOT grounded in tool "
            "results." + detail + " Only report values that literally appear in "
            "tool observations; if you have not called a tool, you do not know "
            "the answer."
        )

    # 3. Role confusion -> user agent
    rc = ev.get("role_confusion") or {}
    if isinstance(rc, dict) and rc.get("has_confusion", False):
        reason = rc.get("reason", "")
        user_hints.append(
            f"A previous attempt failed because the simulated user drifted into "
            f"acting like an assistant: \"{reason}\". You are the CUSTOMER. "
            f"Never offer help, never summarize options back, never confirm "
            f"details in an assistant-like tone. State your own needs, "
            f"preferences, and reactions only."
        )

    # 4. Syntax -> system agent (backstop; in-loop retry should catch most)
    syn_summary = (ev.get("syntax") or {}).get("summary") or {}
    structure_ok = (syn_summary.get("structure") or {}).get("valid", True)
    tool_ok = (syn_summary.get("tool") or {}).get("valid", True)
    if not (structure_ok and tool_ok):
        counts = (syn_summary.get("structure") or {}).get("failure_counts", {})
        detail = f" ({', '.join(sorted(counts))})" if counts else ""
        system_hints.append(
            "A previous attempt had response-format errors" + detail + ". Every "
            "response must be exactly: <think>...</think> then <plan>...</plan> "
            "then ONE <action> block, with no text outside these blocks."
        )

    return {"system_hints": system_hints, "user_hints": user_hints}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_run_dir(run_dir: Path):
    """'20260714_233553__ba_001__persona_005' -> ('ba_001', 'persona_005')."""
    parts = run_dir.name.split("__")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return None, None


def load_eval(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def latest_eval_for(outputs_root: Path, scenario_id: str, persona_id: str):
    """Newest eval.json for a scenario+persona pair (by run-dir mtime)."""
    candidates = sorted(
        outputs_root.glob(f"*__{scenario_id}__{persona_id}/eval.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def collect_already_valid() -> set:
    """(scenario_id, persona_id) pairs already covered by ANY valid_outputs version."""
    pair_re = re.compile(r"([a-z]{2}_[a-z0-9]+_\d+\w*)__(persona_\d+)")
    pairs = set()
    if not VALID_OUTPUTS_ROOT.exists():
        return pairs
    for f in VALID_OUTPUTS_ROOT.rglob("*.json"):
        m = pair_re.search(str(f))
        if m:
            pairs.add((m.group(1), m.group(2)))
    return pairs


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fix failed conversations via regenerate-with-feedback")
    ap.add_argument("outputs_root", type=Path, help="e.g. data/outputs")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be retried")
    ap.add_argument("--domain", help="Only process scenario IDs starting with this prefix (e.g. 'ba')")
    ap.add_argument("--limit", type=int, default=0, help="Max number of failed runs to process (0 = all)")
    ap.add_argument("--eval-model", default="gpt-5.1")
    args = ap.parse_args()

    outputs_root = args.outputs_root.resolve()
    REJECTED_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Collect failed runs, deduplicated by (scenario, persona) keeping the newest.
    # (If a pair was retried before, only its latest verdict matters.)
    latest: dict = {}
    for eval_path in outputs_root.rglob("eval.json"):
        scenario_id, persona_id = parse_run_dir(eval_path.parent)
        if not scenario_id:
            continue
        if args.domain and not scenario_id.startswith(args.domain):
            continue
        key = (scenario_id, persona_id)
        if key not in latest or eval_path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = eval_path

    failed = {k: p for k, p in latest.items()
              if load_eval(p).get("SUCCESS") is not True}

    # Skip pairs already covered by a valid conversation in ANY valid_outputs
    # version — fixing those adds nothing to dataset coverage.
    already_valid = collect_already_valid()
    before = len(failed)
    failed = {k: p for k, p in failed.items() if k not in already_valid}

    print(f"pairs seen: {len(latest)}  currently failing: {before}  "
          f"needing fix (not already valid): {len(failed)}")
    if args.limit:
        failed = dict(list(failed.items())[:args.limit])

    fixed = exhausted = skipped = 0

    for (scenario_id, persona_id), eval_path in sorted(failed.items()):
        ev = load_eval(eval_path)
        hints = hints_from_eval(ev)
        if not (hints["system_hints"] or hints["user_hints"]):
            skipped += 1
            continue

        run_dir = eval_path.parent
        hints_file = run_dir / "correction_hints.json"
        hints_file.write_text(json.dumps(hints, indent=2))
        print(f"\n[{scenario_id}__{persona_id}] "
              f"{len(hints['system_hints'])} system hint(s), "
              f"{len(hints['user_hints'])} user hint(s)")
        for h in hints["system_hints"] + hints["user_hints"]:
            print(f"    - {h[:120]}{'...' if len(h) > 120 else ''}")
        if args.dry_run:
            continue

        success = False
        for attempt in range(1, args.max_attempts + 1):
            print(f"  attempt {attempt}/{args.max_attempts} ...")
            cmd = [
                sys.executable, "src/simulate.py", scenario_id,
                "--persona-id", persona_id,
                "--correction-hints", str(hints_file),
                "--run-eval",
                "--eval-model", args.eval_model,
                "--valid-outputs-version", "v3",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

            new_eval_path = latest_eval_for(outputs_root, scenario_id, persona_id)
            new_ev = load_eval(new_eval_path) if new_eval_path else {}

            if new_ev.get("SUCCESS") is True:
                # simulate.py already copied the conversation into valid_outputs
                print(f"  ✓ fixed on attempt {attempt}")
                fixed += 1
                success = True
                break

            # Refresh hints from the NEW failure for the next attempt
            new_hints = hints_from_eval(new_ev)
            if new_hints["system_hints"] or new_hints["user_hints"]:
                hints_file.write_text(json.dumps(new_hints, indent=2))

        if not success:
            exhausted += 1
            with REJECTED_LOG.open("a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "scenario_id": scenario_id,
                    "persona_id": persona_id,
                    "last_eval": str(eval_path),
                    "hints": hints,
                }) + "\n")
            print(f"  ✗ still failing after {args.max_attempts} attempts -> {REJECTED_LOG}")

    print(f"\n=== summary ===\nfixed={fixed}  exhausted={exhausted}  skipped={skipped}")


if __name__ == "__main__":
    main()
