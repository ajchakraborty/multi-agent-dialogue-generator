#!/usr/bin/env python3
"""
Convert generated conversations to APIGen-MT-5k format, in two variants:

  1. non-reasoning : <think>/<plan> dropped (clean APIGen-style turns)
  2. reasoning     : <think>/<plan> preserved inline in the turn value

APIGen-MT-5k row schema (https://huggingface.co/datasets/Salesforce/APIGen-MT-5k):
{
  "conversations": [{"from": "human"|"gpt"|"function_call"|"observation", "value": str}, ...],
  "system": str,
  "tools": str   # JSON string of a list of tool schema objects
}

Usage:
  # JSON array output (byte-compatible with the HF dataset file)
  python scripts/convert_to_apigen.py \
      --source data/valid_outputs/v2 \
      --tools data/scenario/banking/tools.json \
      --out-dir data/apigen

  # JSONL output (recommended for analysis / HF datasets loader)
  python scripts/convert_to_apigen.py \
      --source data/valid_outputs/v2 \
      --tools data/scenario/banking/tools.json \
      --out-dir data/apigen \
      --include-meta --jsonl

Outputs:
  data/apigen/apigen_non_reasoning.json[l]
  data/apigen/apigen_reasoning.json[l]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)

DEFAULT_SYSTEM = (
    "You are a helpful customer-support assistant. Use the provided tools to help "
    "the user accomplish their task. Call a tool when you need information or need "
    "to perform an action; otherwise reply to the user directly."
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def load_apigen_tools(tools_path: Path) -> str:
    """Convert internal tools.json ({name: {description, parameters, ...}})
    into an APIGen-style JSON string: list of {name, description, parameters}."""
    with open(tools_path, "r") as f:
        raw = json.load(f)

    tool_defs = raw.get("tools", raw)
    apigen_tools = []
    for name, spec in tool_defs.items():
        if not isinstance(spec, dict):
            continue
        params = spec.get("parameters", {}) or {}
        properties = {}
        required = []
        for pname, pspec in params.items():
            pspec = dict(pspec or {})
            if pspec.pop("required", False):
                required.append(pname)
            properties[pname] = pspec
        apigen_tools.append({
            "name": name,
            "description": spec.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return json.dumps(apigen_tools, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Conversation discovery / loading
# ---------------------------------------------------------------------------

SKIP_FILES = {"eval.json", "agent_prompts.json", "tools.json", "catalog.json",
              "index.jsonl", "results.jsonl", "orchestration_summary.json"}


def find_conversation_files(source: Path) -> List[Path]:
    """Find conversation json files under source (handles v1 & v2 layouts)."""
    files = []
    for p in sorted(source.rglob("*.json")):
        if p.name in SKIP_FILES or p.name.startswith("orchestration_"):
            continue
        # v1 layout: conversation.json ; v2 layout: <run_dir_name>.json
        if p.name == "conversation.json" or p.name == f"{p.parent.name}.json":
            files.append(p)
    if not files:
        # Fallback: any remaining json (validated later by presence of "messages")
        for p in sorted(source.rglob("*.json")):
            if p.name in SKIP_FILES or p.name.startswith("orchestration_"):
                continue
            files.append(p)
    return files


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: cannot read {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Turn conversion
# ---------------------------------------------------------------------------

def extract_reasoning_prefix(output_raw: str) -> str:
    """Pull <think> and <plan> blocks out of raw output, re-wrapped cleanly."""
    parts = []
    m = THINK_RE.search(output_raw or "")
    if m and m.group(1).strip():
        parts.append(f"<think>\n{m.group(1).strip()}\n</think>")
    m = PLAN_RE.search(output_raw or "")
    if m and m.group(1).strip():
        parts.append(f"<plan>\n{m.group(1).strip()}\n</plan>")
    return "\n".join(parts)


def convert_messages(convo: Dict[str, Any], reasoning: bool) -> List[Dict[str, str]]:
    """Map internal messages/steps structure -> APIGen 'conversations' list."""
    out: List[Dict[str, str]] = []

    for msg in convo.get("messages", []):
        role = msg.get("role")

        if role == "user":
            text = (msg.get("output_raw") or msg.get("content") or "").strip()
            # Strip control tokens the user simulator may emit
            text = text.replace("[DONE_SUCCESS]", "").replace("[DONE_FAILURE]", "").strip()
            if text:
                out.append({"from": "human", "value": text})
            continue

        if role != "assistant":
            continue

        for step in msg.get("steps", []):
            action = step.get("action_structured") or {}
            a_type = action.get("type")
            prefix = extract_reasoning_prefix(step.get("output_raw", "")) if reasoning else ""

            if a_type == "tool_call":
                fc = json.dumps(
                    {"name": action.get("name"), "arguments": action.get("args") or {}},
                    ensure_ascii=False,
                )
                value = f"{prefix}\n{fc}".strip() if prefix else fc
                out.append({"from": "function_call", "value": value})

                obs = step.get("observation") or {}
                obs_raw = obs.get("raw")
                if obs_raw is None and obs.get("parsed") is not None:
                    obs_raw = json.dumps(obs["parsed"], ensure_ascii=False)
                out.append({"from": "observation", "value": obs_raw or ""})

            elif a_type == "say":
                text = (action.get("text") or "").strip()
                value = f"{prefix}\n{text}".strip() if prefix else text
                if value:
                    out.append({"from": "gpt", "value": value})

    return out


def build_entry(convo: Dict[str, Any], tools_str: str, reasoning: bool,
                source_file: Path, include_meta: bool) -> Optional[Dict[str, Any]]:
    conversations = convert_messages(convo, reasoning)
    if not conversations:
        return None

    entry: Dict[str, Any] = {
        "conversations": conversations,
        "system": DEFAULT_SYSTEM,
        "tools": tools_str,
    }
    if include_meta:
        meta = convo.get("meta", {}) or {}
        cfg = convo.get("config", {}) or {}
        outcome = convo.get("outcome", {}) or {}
        entry["_meta"] = {
            "source_file": str(source_file),
            "conversation_id": meta.get("conversation_id"),
            "domain": meta.get("domain_id"),
            "scenario": cfg.get("scenario_name"),
            "persona": (cfg.get("persona") or {}).get("id"),
            "model": meta.get("model"),
            "total_turns": outcome.get("total_turns"),
            "had_tool_calls": outcome.get("had_tool_calls"),
        }
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert conversations to APIGen-MT format (reasoning + non-reasoning variants)"
    )
    ap.add_argument("--source", default="data/valid_outputs/v2",
                    help="Directory containing validated conversation runs")
    ap.add_argument("--tools", default="data/scenario/banking/tools.json",
                    help="Path to internal tools.json (converted to APIGen tool list)")
    ap.add_argument("--out-dir", default="data/apigen",
                    help="Output directory for the two dataset files")
    ap.add_argument("--include-meta", action="store_true",
                    help="Attach _meta provenance (scenario/persona/domain) for statistical analysis")
    ap.add_argument("--jsonl", action="store_true",
                    help="Write JSONL (one entry per line) instead of a JSON array")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    tools_path = Path(args.tools)
    if not tools_path.exists():
        print(f"Error: tools file not found: {tools_path}", file=sys.stderr)
        sys.exit(1)

    tools_str = load_apigen_tools(tools_path)
    convo_files = find_conversation_files(source)
    print(f"Found {len(convo_files)} conversation file(s) under {source}", file=sys.stderr)

    non_reasoning: List[Dict[str, Any]] = []
    reasoning_ds: List[Dict[str, Any]] = []
    skipped = 0

    for cf in convo_files:
        convo = load_json(cf)
        if not convo or "messages" not in convo:
            skipped += 1
            continue
        e1 = build_entry(convo, tools_str, reasoning=False, source_file=cf,
                         include_meta=args.include_meta)
        e2 = build_entry(convo, tools_str, reasoning=True, source_file=cf,
                         include_meta=args.include_meta)
        if e1 and e2:
            non_reasoning.append(e1)
            reasoning_ds.append(e2)
        else:
            skipped += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = "jsonl" if args.jsonl else "json"
    nr_path = out_dir / f"apigen_non_reasoning.{ext}"
    r_path = out_dir / f"apigen_reasoning.{ext}"

    def write(path: Path, entries: List[Dict[str, Any]]) -> None:
        with open(path, "w") as f:
            if args.jsonl:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            else:
                json.dump(entries, f, indent=2, ensure_ascii=False)

    write(nr_path, non_reasoning)
    write(r_path, reasoning_ds)

    print(f"non-reasoning : {len(non_reasoning)} entries -> {nr_path}")
    print(f"reasoning     : {len(reasoning_ds)} entries -> {r_path}")
    print(f"skipped       : {skipped}")


if __name__ == "__main__":
    main()