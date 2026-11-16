# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm==0.6.6.post1",
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "transformers==4.47.1",
#     "datasets",
#     "sentence-transformers",
#     "numpy",
#     "tqdm",
# ]
# ///

"""
L-1 evaluation of instruct models on APIGen-format data using the v3 system prompt,
accelerated with vLLM (batched inference across N GPUs).

Usage:
  # public benchmark
  uv run evaluation/apigen_L1.py --model Qwen/Qwen2.5-7B-Instruct --n-samples 100 --tp 8

  # your generated data
  uv run evaluation/apigen_L1.py --model Qwen/Qwen2.5-7B-Instruct \
      --local-file data/apigen/banking/apigen_non_reasoning.jsonl \
      --n-samples 300 --tp 8 \
      --predictions preds_banking.jsonl --summary l1_banking.json

  # held-out abstention set, scored in the grammar the SFT files actually train
  # (--format native is REQUIRED for any model fine-tuned on data/sft/*.jsonl)
  uv run evaluation/apigen_L1.py --model <ft-checkpoint> --format native \\
      --local-file data/apigen/l1_impossible_test.jsonl --n-samples 765 --tp 8

  # re-score existing predictions (no GPU needed)
  uv run evaluation/apigen_L1.py --eval-only --predictions preds_banking.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Reuse the internal-tools.json -> OpenAI `tools` transform rather than
# reimplementing the per-parameter `required` hoist. build_sft_data is stdlib-only
# at import time, so this costs nothing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_sft_data import load_openai_tools  # noqa: E402

# ----------------------------------------------------------------------
# 1. SYSTEM PROMPTS + TOOLS
# ----------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent

V3_PROMPT_PATH = REPO / "prompts/system_agent/v3.txt"
V3_SYSTEM_PROMPT = V3_PROMPT_PATH.read_text(encoding="utf-8").strip()

SYSTEM_INSTRUCTION_TEMPLATE = (
    V3_SYSTEM_PROMPT
    + "\n\n=== AVAILABLE TOOLS ===\n{tools_json}\n\nYou now begin when the user speaks."
)

# --format native: the prompt the SFT files are built with. Same text as v3.txt minus
# the <think>/<plan>/<action> format contract, so the model is asked for a decision
# rather than for a grammar it was trained out of.
SFT_PROMPT_PATH = REPO / "prompts/sft/system_v3_sft.txt"

# Internal tools.json per domain, for the native-mode `tools=` argument.
TOOLS_ROOT = REPO / "data/scenario"
_TOOLS_CACHE = {}


def sft_system_prompt() -> str:
    return SFT_PROMPT_PATH.read_text(encoding="utf-8").strip()


def domain_tools(domain):
    """OpenAI tool schemas for a domain, or None when it cannot be resolved."""
    if domain not in _TOOLS_CACHE:
        path = TOOLS_ROOT / str(domain) / "tools.json"
        _TOOLS_CACHE[domain] = load_openai_tools(path) if path.exists() else None
    return _TOOLS_CACHE[domain]

# ----------------------------------------------------------------------
# 2. DATA PREPARATION (APIGen format -> per-turn L-1 datapoints)
# ----------------------------------------------------------------------

# Strips <think>/<plan> wrappers that the "reasoning" variant of
# data_conversion.py writes into function_call values.
_REASONING_RE = re.compile(r"<(think|plan)>.*?</\1>", re.DOTALL)


def parse_function_call(mval: str):
    """
    Parse a function_call value that may be prefixed with <think>/<plan> blocks.

    Returns (name, params) or None if unparseable.
    """
    s = _REASONING_RE.sub("", mval).strip()
    start = s.find("{")
    if start == -1:
        return None
    try:
        obj = json.loads(s[start:])
    except json.JSONDecodeError:
        # tolerate trailing prose after the JSON object
        depth, end = 0, None
        for idx in range(start, len(s)):
            if s[idx] == "{":
                depth += 1
            elif s[idx] == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end is None:
            return None
        try:
            obj = json.loads(s[start:end])
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    params = obj.get("arguments") or obj.get("parameters") or {}
    if not name:
        return None
    return name, params


def build_datapoints(example, fmt="action"):
    conv = example["conversations"]
    if isinstance(conv, str):
        conv = json.loads(conv)

    tools = example.get("tools", "[]")
    if not isinstance(tools, str):
        tools = json.dumps(tools, indent=2)

    # carried through to predictions so results can be sliced by cell
    meta = example.get("_meta") or {}

    if fmt == "native":
        # System prompt carries no tool text and no format contract: the schemas go
        # through the chat template's own `tools=` slot, exactly as in training.
        instruction = sft_system_prompt()
        native_tools = domain_tools(meta.get("domain"))
    else:
        # NOTE: use .replace, NOT .format — the v3 prompt contains literal JSON braces
        instruction = SYSTEM_INSTRUCTION_TEMPLATE.replace("{tools_json}", tools)
        native_tools = None

    datapoints = []
    history_blocks = []
    # Native mode mirrors the SFT record shape: assistant messages carry `tool_calls`
    # with sequential call_N ids and empty content, each followed by a role:"tool"
    # message keyed on tool_call_id. The flattened <user>/<obs>/<action> text is an
    # input format the fine-tune never saw, and its inline <action> tags would prime
    # the model back into the grammar we deliberately trained out.
    history_msgs = [{"role": "system", "content": instruction}]
    call_seq = 0

    i = 0
    while i < len(conv):
        msg = conv[i]
        role, val = msg["from"], (msg["value"] or "").strip()

        if role == "human":
            history_blocks.append(f"<user> {val} </user>")
            history_msgs.append({"role": "user", "content": val})

            gold_tool_calls, gold_response = [], None
            j = i + 1
            turn_blocks = []
            turn_msgs = []
            pending_call_id = None
            while j < len(conv) and conv[j]["from"] != "human":
                m = conv[j]
                mval = (m["value"] or "").strip()
                if m["from"] == "function_call":
                    parsed = parse_function_call(mval)
                    if parsed is not None:
                        name, params = parsed
                        gold_tool_calls.append({"name": name, "params": params})
                        turn_blocks.append(
                            f'<action type="tool" name="{name}">\n{json.dumps(params)}\n</action>'
                        )
                        call_seq += 1
                        pending_call_id = f"call_{call_seq}"
                        turn_msgs.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": pending_call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(params)},
                            }],
                        })
                elif m["from"] == "observation":
                    turn_blocks.append(f"<obs> {mval} </obs>")
                    if pending_call_id is not None:
                        turn_msgs.append({"role": "tool",
                                          "tool_call_id": pending_call_id,
                                          "content": mval})
                        pending_call_id = None
                elif m["from"] == "gpt":
                    gold_response = _REASONING_RE.sub("", mval).strip()
                    turn_blocks.append(f'<action type="say">{gold_response}</action>')
                    turn_msgs.append({"role": "assistant", "content": gold_response})
                j += 1

            if gold_tool_calls or gold_response:
                datapoints.append(
                    {
                        "instruction": instruction,
                        "input": "**Dialogue Records History**\n" + "\n\n".join(history_blocks),
                        "messages": list(history_msgs) if fmt == "native" else None,
                        "tools": native_tools,
                        "gold_tool_calls": gold_tool_calls,
                        "gold_response": gold_response,
                        "meta": meta,
                    }
                )

            history_blocks.extend(turn_blocks)
            history_msgs.extend(turn_msgs)
            i = j
        else:
            i += 1

    return datapoints


# ----------------------------------------------------------------------
# 3. vLLM INFERENCE (batched, multi-GPU)
# ----------------------------------------------------------------------

def _patch_broken_nvml():
    """Work around a host driver/library version mismatch (libnvidia-ml.so 580.178
    vs kernel driver 580.173.02) that makes real NVML calls raise, even though
    torch.cuda works fine (it talks to the CUDA driver directly, not NVML). Left
    untouched, vLLM's platform autodetection (vllm/platforms/__init__.py, which
    gates on `pynvml.nvmlInit()` succeeding) falls through to UnspecifiedPlatform
    and "Failed to infer device type".

    vLLM already has a torch.cuda-only fallback for exactly this situation --
    NonNvmlCudaPlatform (vllm/platforms/cuda.py) -- but it's only selected if the
    *first* nvmlInit() (the top-level is_cuda gate) succeeds and the *second*
    nvmlInit() (cuda.py's own nvml_available probe) fails. So: no-op if real NVML
    already works here; otherwise install a stub that succeeds once, then raises,
    to steer vLLM onto its own intended fallback rather than reimplementing it.
    """
    import sys
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return  # real NVML is fine in this environment; don't touch anything
    except Exception:
        pass

    import types
    import torch

    calls = {"init": 0}

    def _nvml_init():
        calls["init"] += 1
        if calls["init"] > 1:
            raise RuntimeError("stubbed NVML: forcing vLLM's NonNvmlCudaPlatform")

    stub = types.ModuleType("pynvml")
    stub.__file__ = "<pynvml_nvml_mismatch_stub>"
    stub.nvmlInit = _nvml_init
    stub.nvmlShutdown = lambda: None
    stub.nvmlDeviceGetCount = torch.cuda.device_count
    sys.modules["pynvml"] = stub


def run_inference(model_name, datapoints, max_new_tokens, tp_size, max_model_len,
                  fmt="action", cpu_offload_gb=0, gpu_memory_utilization=0.85):
    _patch_broken_nvml()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    llm = LLM(
        model=model_name,
        device="cuda",
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        cpu_offload_gb=cpu_offload_gb,
        enforce_eager=True,
        dtype="float16",
        trust_remote_code=True,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    prompts = []
    for dp in datapoints:
        if fmt == "native":
            # Real message history + the template's own tools slot. No trailing
            # instruction line: training never had one, and the generation prompt
            # already marks the assistant's turn.
            kwargs = {"tools": dp["tools"]} if dp.get("tools") else {}
            prompts.append(
                tokenizer.apply_chat_template(
                    dp["messages"], tokenize=False, add_generation_prompt=True, **kwargs
                )
            )
            continue
        messages = [
            {"role": "system", "content": dp["instruction"]},
            {
                "role": "user",
                "content": dp["input"]
                + "\n\n### Produce the assistant turn for the last user message, following ALL rules.",
            },
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    # vLLM batches everything internally with continuous batching
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outputs]


# ----------------------------------------------------------------------
# 4. L-1 EVALUATION
# ----------------------------------------------------------------------

ACTION_RE = re.compile(
    r"<action\s+type=\"(?P<type>[a-zA-Z]+)\"(?:\s+name=\"(?P<name>[^\"]+)\")?>\s*(?P<body>.*?)</action>",
    re.DOTALL,
)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>", re.DOTALL)


def normalize_params(p):
    # Coerce JSON-string params into dicts
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except json.JSONDecodeError:
            return {}
    if not isinstance(p, dict):
        return {}
    return {
        k: (str(v).strip().lower() if not isinstance(v, (dict, list)) else v)
        for k, v in p.items()
    }


def extract_pred(text, fmt="action"):
    if fmt == "native":
        return extract_pred_native(text)

    calls = []
    response = None

    for m in ACTION_RE.finditer(text):
        a_type = m.group("type")
        body = m.group("body").strip()
        if a_type == "tool":
            try:
                params = json.loads(body)
            except json.JSONDecodeError:
                params = {}
            calls.append({"name": m.group("name"), "params": params})
        else:
            response = body

    if not calls:
        for m in TOOL_CALL_RE.finditer(text):
            try:
                obj = json.loads(m.group("body"))
                calls.append(
                    {"name": obj.get("name"), "params": obj.get("arguments") or obj.get("parameters") or {}}
                )
            except json.JSONDecodeError:
                continue

    return calls, response


def extract_pred_native(text):
    """Parse a natively tool-calling reply: `<tool_call>{...}</tool_call>` blocks
    (Qwen's template emits these) plus whatever prose sits outside them.

    When the model calls a tool, the turn's ACTION is that call and any prose is a
    preamble, so `response` is left None -- mirroring the training records, where a
    tool-call assistant message has empty content. That keeps `action_recall` a
    say-vs-tool decision rather than rewarding a hedge that does both.
    """
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group("body"))
        except json.JSONDecodeError:
            continue
        calls.append({"name": obj.get("name"),
                      "params": obj.get("arguments") or obj.get("parameters") or {}})

    # Tolerate a bare {"name": ..., "arguments": ...} object with no wrapper.
    if not calls:
        parsed = parse_function_call(text)
        if parsed is not None:
            name, params = parsed
            calls.append({"name": name, "params": params})

    if calls:
        return calls, None

    prose = _REASONING_RE.sub("", TOOL_CALL_RE.sub("", text)).strip()
    return calls, (prose or None)


def token_f1(a, b):
    a_set, b_set = set(a.lower().split()), set(b.lower().split())
    overlap = len(a_set & b_set)
    if overlap == 0:
        return 0.0
    p, r = overlap / len(a_set), overlap / len(b_set)
    return 2 * p * r / (p + r)


METRIC_KEYS = [
    "action_recall",
    "tool_name_acc",
    "params_match_full",
    "params_match_partial",
    "tool_recall",
    "final_answer_f1",
    "final_answer_sim",
]


def evaluate(predictions, use_semantic=True, fmt="action"):
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
    for item in tqdm(predictions, desc="Evaluating", ncols=90):
        gold_calls = item["gold_tool_calls"]
        gold_resp = item["gold_response"]
        pred_calls, pred_resp = extract_pred(item["prediction"], item.get("format", fmt))

        L = min(len(gold_calls), len(pred_calls))
        name_m, full_m, part_m = [], [], []
        for k in range(L):
            g, p = gold_calls[k], pred_calls[k]
            name_ok = g["name"] == p["name"]
            gp, pp = normalize_params(g["params"]), normalize_params(p["params"])
            name_m.append(1.0 if name_ok else 0.0)
            full_m.append(1.0 if (name_ok and gp == pp) else 0.0)
            if name_ok and gp:
                part_m.append(sum(1 for kk, vv in gp.items() if pp.get(kk) == vv) / len(gp))
            else:
                part_m.append(0.0)

        gold_type = "say" if gold_resp else "tool"
        pred_type = "say" if pred_resp else ("tool" if pred_calls else None)
        action_recall = 1.0 if gold_type == pred_type else 0.0

        fa_f1 = fa_sim = None
        if gold_resp:
            if pred_resp:
                fa_f1 = token_f1(pred_resp, gold_resp)
                fa_sim = semantic_similarity(pred_resp, gold_resp)
            else:
                fa_f1, fa_sim = 0.0, 0.0

        results.append(
            {
                "action_recall": action_recall,
                "tool_name_acc": np.mean(name_m) if name_m else None,
                "params_match_full": np.mean(full_m) if full_m else None,
                "params_match_partial": np.mean(part_m) if part_m else None,
                "tool_recall": (len(pred_calls) / len(gold_calls)) if gold_calls else None,
                "final_answer_f1": fa_f1,
                "final_answer_sim": fa_sim,
                "_meta": item.get("meta") or {},
            }
        )

    def avg_over(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {k: avg_over(results, k) for k in METRIC_KEYS}
    summary["n_datapoints"] = len(results)

    # per-cell breakdown (happy / unhappy / impossible), when _meta is present
    buckets = defaultdict(list)
    for r in results:
        cell = (r["_meta"] or {}).get("path_type")
        if cell:
            buckets[cell].append(r)
    if buckets:
        summary["by_path_type"] = {
            cell: {**{k: avg_over(rows, k) for k in METRIC_KEYS}, "n_datapoints": len(rows)}
            for cell, rows in sorted(buckets.items())
        }

    return summary, results


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def load_local(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    data = json.load(open(path, encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="Salesforce/APIGen-MT-5k")
    ap.add_argument("--local-file", default=None,
                    help="Path to a local APIGen-format .jsonl/.json (skips the HF download)")
    ap.add_argument("--n-samples", type=int, default=1000, help="Number of conversations")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=8, help="tensor_parallel_size (number of GPUs)")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predictions", default="predictions-apigen.jsonl")
    ap.add_argument("--summary", default="l1-summary-apigen.json")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--format", choices=["action", "native"], default="action",
                    help="Output grammar expected of the model. 'action' (default) "
                         "prompts with prompts/system_agent/v3.txt and parses "
                         "<action type=...> blocks -- keeps older L-1 runs comparable. "
                         "'native' prompts with prompts/sft/system_v3_sft.txt, passes "
                         "real tool schemas through the chat template's tools= slot, "
                         "rebuilds the history as real messages, and parses native "
                         "tool_calls with bare prose as the say action. Use 'native' "
                         "for any model fine-tuned on data/sft/*.jsonl: under 'action' "
                         "it emits native calls and prose, matches no <action> tag, and "
                         "scores 0 action_recall on every say turn (61%% of the "
                         "impossible test set) for purely formatting reasons.")
    args = ap.parse_args()

    if not args.eval_only:
        if args.local_file:
            p = Path(args.local_file)
            ds = load_local(p)
            random.Random(args.seed).shuffle(ds)
            ds = ds[: args.n_samples]
            print(f"Loaded {len(ds)} conversations from {p}")
        else:
            from datasets import load_dataset

            print(f"Loading {args.dataset} ...")
            n = args.n_samples
            ds = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
            ds = ds.select(range(min(n, len(ds))))
            print(f"Loaded {len(ds)} conversations from {args.dataset}")

        datapoints = []
        for ex in ds:
            datapoints.extend(build_datapoints(ex, args.format))
        print(f"Built {len(datapoints)} per-turn datapoints from {len(ds)} conversations "
              f"(format={args.format})")

        n_tool_turns = sum(1 for d in datapoints if d["gold_tool_calls"])
        n_say_turns = sum(1 for d in datapoints if d["gold_response"] and not d["gold_tool_calls"])
        print(f"  gold turns -> tool: {n_tool_turns}   say-only: {n_say_turns}")
        if n_tool_turns == 0:
            print("  WARNING: no gold tool calls parsed — check the function_call format.")
        if args.format == "native":
            missing = sum(1 for d in datapoints if not d.get("tools"))
            if missing:
                print(f"  WARNING: {missing} datapoint(s) have no resolved tool schema — "
                      f"check _meta.domain against {TOOLS_ROOT}/<domain>/tools.json")

        print(f"Loading vLLM engine: {args.model} (tp={args.tp}) ...")
        preds = run_inference(
            args.model, datapoints, args.max_new_tokens, args.tp, args.max_model_len,
            fmt=args.format
        )

        with open(args.predictions, "w", encoding="utf-8") as f:
            for dp, pred in zip(datapoints, preds):
                rec = {
                    "input": dp["input"],
                    "gold_tool_calls": dp["gold_tool_calls"],
                    "gold_response": dp["gold_response"],
                    "prediction": pred,
                    "format": args.format,
                    "meta": dp.get("meta") or {},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Predictions saved -> {args.predictions}")

    predictions = [json.loads(line) for line in open(args.predictions, encoding="utf-8") if line.strip()]
    summary, _ = evaluate(predictions, use_semantic=not args.no_semantic, fmt=args.format)

    print("\n===== L-1 SUMMARY =====")
    print(json.dumps(summary, indent=2))
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved -> {args.summary}")


if __name__ == "__main__":
    main()