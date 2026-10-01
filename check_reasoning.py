# -*- coding: utf-8 -*-
"""
Probe how each chosen model reasons on OpenRouter.

Sends the SAME short prompt to every model under three settings and reports the
reasoning_tokens actually spent, so you can see the provider default (enabled,
no effort) vs off vs each effort level. Comparable reasoning => stable token
counts across the effort sweep; the 'default' column is what you'd get today by
sending {"enabled": True} without an effort.

Run:  python check_reasoning.py
Needs: OPENROUTER_API_KEY in env. Reuses client/policies from utils.py.
"""
import time

from utils import client, clean_policy, NO_REASONING_PARAM, log_usage

MODELS = [
    "qwen/qwen3.7-plus",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
]

# A prompt that invites (but doesn't force) reasoning, so a reasoning model
# will actually spend tokens if it defaults to on.
PROBE = [
    {"role": "user",
     "content": "A bat and a ball cost $1.10. The bat costs $1 more than the "
                "ball. How much is the ball? Answer with a number only."},
]

# settings to test per model: label -> reasoning dict (None = omit the field)
SETTINGS = [
    ("off",           {"enabled": False}),
    ("default",       {"enabled": True}),      # provider default intensity
    ("effort=low",    {"effort": "low"}),
    ("effort=medium", {"effort": "medium"}),
    ("effort=high",   {"effort": "high"}),
]


def call(model, reasoning):
    body = {"provider": clean_policy(model)}
    if model not in NO_REASONING_PARAM and reasoning is not None:
        body["reasoning"] = reasoning
    resp = client.chat.completions.create(
        model=model,
        messages=PROBE,
        temperature=0,
        max_tokens=2000,           # headroom so reasoning isn't truncated
        extra_body=body,
    )
    u = log_usage(resp)
    content = (resp.choices[0].message.content or "").strip().replace("\n", " ")
    return u, content[:40]


def main():
    header = f"{'model':32} {'setting':14} {'provider':12} {'reason_tok':>10} {'compl_tok':>9}  answer"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        no_param = model in NO_REASONING_PARAM
        for label, reasoning in SETTINGS:
            if no_param and label != "off":
                print(f"{model:32} {label:14} {'-':12} {'n/a (NO_REASONING_PARAM)':>10}")
                continue
            try:
                u, ans = call(model, reasoning)
                print(f"{model:32} {label:14} {str(u['provider']):12} "
                      f"{u['reasoning_tokens']:>10} {str(u['completion_tokens']):>9}  {ans}")
            except Exception as e:
                print(f"{model:32} {label:14} {'ERR':12} {str(e)[:50]}")
            time.sleep(1)   # gentle on rate limits
        print()


if __name__ == "__main__":
    main()
