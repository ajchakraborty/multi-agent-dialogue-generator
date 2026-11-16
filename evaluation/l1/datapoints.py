#!/usr/bin/env python3
"""
Step-level L-1 datapoint builder.

Unlike evaluation/apigen_L1.py::build_datapoints, which aggregates every tool call
*and* the final response between two human turns into one datapoint (collapsing a
tool-call-then-say turn into a single "say" label), this emits ONE datapoint per
assistant action -- every `function_call` and every `gpt` message in the APIGen
`conversations` list is its own step, conditioned on the real (teacher-forced)
history up to that point. Matches the paper's per-action L-1 definition: "For each
given request, the action prediction from the LLM is either a Final Answer or a
tool call."

Reuses evaluation/apigen_L1.py's prompt/tool plumbing (parse_function_call,
_REASONING_RE, sft_system_prompt, domain_tools, SYSTEM_INSTRUCTION_TEMPLATE) rather
than duplicating it.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "evaluation"))

from apigen_L1 import (  # noqa: E402
    SYSTEM_INSTRUCTION_TEMPLATE,
    _REASONING_RE,
    domain_tools,
    parse_function_call,
    sft_system_prompt,
)


def build_step_datapoints(example, fmt="native"):
    conv = example["conversations"]
    if isinstance(conv, str):
        conv = json.loads(conv)

    tools = example.get("tools", "[]")
    if not isinstance(tools, str):
        tools = json.dumps(tools, indent=2)

    meta = example.get("_meta") or {}

    if fmt == "native":
        instruction = sft_system_prompt()
        native_tools = domain_tools(meta.get("domain"))
    else:
        instruction = SYSTEM_INSTRUCTION_TEMPLATE.replace("{tools_json}", tools)
        native_tools = None

    datapoints = []
    history_blocks = []
    history_msgs = [{"role": "system", "content": instruction}]
    call_seq = 0
    step_index = 0
    pending_call_id = None

    for msg in conv:
        role, val = msg["from"], (msg["value"] or "").strip()

        if role == "human":
            history_blocks.append(f"<user> {val} </user>")
            history_msgs.append({"role": "user", "content": val})
            continue

        if role == "function_call":
            parsed = parse_function_call(val)
            if parsed is None:
                continue
            name, params = parsed
            datapoints.append({
                "input": "**Dialogue Records History**\n" + "\n\n".join(history_blocks),
                "instruction": instruction,
                "messages": list(history_msgs) if fmt == "native" else None,
                "tools": native_tools,
                "gold_type": "tool",
                "gold_tool_calls": [{"name": name, "params": params}],
                "gold_response": None,
                "meta": {**meta, "step_index": step_index},
            })
            step_index += 1

            history_blocks.append(
                f'<action type="tool" name="{name}">\n{json.dumps(params)}\n</action>'
            )
            call_seq += 1
            pending_call_id = f"call_{call_seq}"
            history_msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": pending_call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(params)},
                }],
            })
            continue

        if role == "observation":
            history_blocks.append(f"<obs> {val} </obs>")
            if pending_call_id is not None:
                history_msgs.append({
                    "role": "tool", "tool_call_id": pending_call_id, "content": val,
                })
                pending_call_id = None
            continue

        if role == "gpt":
            gold_response = _REASONING_RE.sub("", val).strip()
            datapoints.append({
                "input": "**Dialogue Records History**\n" + "\n\n".join(history_blocks),
                "instruction": instruction,
                "messages": list(history_msgs) if fmt == "native" else None,
                "tools": native_tools,
                "gold_type": "say",
                "gold_tool_calls": [],
                "gold_response": gold_response,
                "meta": {**meta, "step_index": step_index},
            })
            step_index += 1

            history_blocks.append(f'<action type="say">{gold_response}</action>')
            history_msgs.append({"role": "assistant", "content": gold_response})
            continue

    return datapoints
