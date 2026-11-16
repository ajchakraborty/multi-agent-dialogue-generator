#!/usr/bin/env python3
"""
OpenAI-compatible inference client against Fireworks AI (api.fireworks.ai). Serves
both the pre-fine-tune base model and a Fireworks-deployed fine-tuned checkpoint --
only --model (and --format, wired in run.py) differ between the two runs.

Nothing like this existed in the repo before: scripts/firework.py only builds
Fireworks *fine-tuning* data, it does not call an inference endpoint. There is also
no FIREWORKS_API_KEY or deployed model/deployment ID anywhere in this repo -- both
must be supplied by the caller (env var + --model).

A returned `tool_calls` response is re-encoded as `<tool_call>{...}</tool_call>` text
so evaluation/l1/metrics.py can parse it through the same extract_pred() (native
mode) that already parses local vLLM generations -- one parser, not two.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def make_client():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FIREWORKS_API_KEY is not set. Export it before running evaluation/l1/run.py -- "
            "there is no Fireworks key or deployment configured anywhere in this repo."
        )
    return OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)


def build_request(dp, fmt):
    """-> (messages, extra_kwargs) for one datapoint."""
    if fmt == "native":
        kwargs = {"tools": dp["tools"]} if dp.get("tools") else {}
        return dp["messages"], kwargs
    messages = [
        {"role": "system", "content": dp["instruction"]},
        {"role": "user", "content": dp["input"]
         + "\n\n### Produce the next assistant action, following ALL rules."},
    ]
    return messages, {}


def _encode_tool_calls(tool_calls):
    parts = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}
        parts.append(json.dumps({"name": tc.function.name, "arguments": args}))
    return "".join(f"<tool_call>{p}</tool_call>" for p in parts)


def _call_one(client, model, dp, fmt, max_tokens, max_retries):
    messages, kwargs = build_request(dp, fmt)
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0,
                max_tokens=max_tokens, **kwargs,
            )
            choice = resp.choices[0].message
            text = choice.content or ""
            if getattr(choice, "tool_calls", None):
                text += _encode_tool_calls(choice.tool_calls)
            return text
        except openai.APIError as e:
            status = getattr(e, "status_code", None)
            retryable = status is None or status >= 500 or status == 429
            if attempt == max_retries or not retryable:
                print(f"  ERROR (giving up after {attempt + 1} attempt(s)): {e}",
                      file=sys.stderr)
                return ""
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return ""


def run_inference(datapoints, model, fmt="native", concurrency=8, max_tokens=512,
                   max_retries=5):
    client = make_client()
    results = [None] * len(datapoints)

    def worker(i, dp):
        return i, _call_one(client, model, dp, fmt, max_tokens, max_retries)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i, dp) for i, dp in enumerate(datapoints)]
        done = 0
        for fut in as_completed(futures):
            i, text = fut.result()
            results[i] = text
            done += 1
            if done % 200 == 0 or done == len(datapoints):
                print(f"  {done}/{len(datapoints)} done", file=sys.stderr)

    return results
