# -*- coding: utf-8 -*-
"""
BFI-44 ADMINISTERED TO THE AGENTS.

Agents are given the 44 Big Five Inventory items (John & Srivastava, 1999), the
5 scores are computed with the original keys and compared to the scores IMPOSED
in the prompt.

Output:
  1. <name>-scores.csv   one row per agent: persona_id, model, the 5 imposed
                         and the 5 recovered scores (the "agent list")
  2. <name>-corr.csv     one row per model x trait: correlation between imposed
                         and recovered
  3. <name>-bfi.png      scatter grid: rows = models, cols = traits, one dot per
                         agent, r printed in each panel
  4. <name>-raw.csv      item-by-item answers, for re-analysis

Personas are the same as the main experiment (same seed, same system message),
so <name>-scores.csv joins to pilot100-agents.csv on persona_id + model.

Usage:
    python bfi_test.py --name bfi --n_personas 100     # 500 calls
    python bfi_test.py --name bfi --analyze_only       # re-analysis, no API
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

from personas import TRAITS, build_personas
from prompt import system_persona
from utils import get_completion_from_router_json

FIVE_ROUTER_MODELS = [
    "qwen/qwen3.7-plus",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
]

# ---------------------------------------------------------------------------
# Instrument: Big Five Inventory (BFI-44), John & Srivastava (1999).
# Stem: "I see myself as someone who...". Do not change the numbering:
# the scoring keys below depend on it.
# ---------------------------------------------------------------------------
BFI_ITEMS = {
    1: "Is talkative",
    2: "Tends to find fault with others",
    3: "Does a thorough job",
    4: "Is depressed, blue",
    5: "Is original, comes up with new ideas",
    6: "Is reserved",
    7: "Is helpful and unselfish with others",
    8: "Can be somewhat careless",
    9: "Is relaxed, handles stress well",
    10: "Is curious about many different things",
    11: "Is full of energy",
    12: "Starts quarrels with others",
    13: "Is a reliable worker",
    14: "Can be tense",
    15: "Is ingenious, a deep thinker",
    16: "Generates a lot of enthusiasm",
    17: "Has a forgiving nature",
    18: "Tends to be disorganized",
    19: "Worries a lot",
    20: "Has an active imagination",
    21: "Tends to be quiet",
    22: "Is generally trusting",
    23: "Tends to be lazy",
    24: "Is emotionally stable, not easily upset",
    25: "Is inventive",
    26: "Has an assertive personality",
    27: "Can be cold and aloof",
    28: "Perseveres until the task is finished",
    29: "Can be moody",
    30: "Values artistic, aesthetic experiences",
    31: "Is sometimes shy, inhibited",
    32: "Is considerate and kind to almost everyone",
    33: "Does things efficiently",
    34: "Remains calm in tense situations",
    35: "Prefers work that is routine",
    36: "Is outgoing, sociable",
    37: "Is sometimes rude to others",
    38: "Makes plans and follows through with them",
    39: "Gets nervous easily",
    40: "Likes to reflect, play with ideas",
    41: "Has few artistic interests",
    42: "Likes to cooperate with others",
    43: "Is easily distracted",
    44: "Is sophisticated in art, music, or literature",
}

# "R" = reverse-scored item (6 - answer)
SCORING = {
    "Openness":          [5, 10, 15, 20, 25, 30, "35R", 40, "41R", 44],
    "Conscientiousness": [3, "8R", 13, "18R", "23R", 28, 33, 38, "43R"],
    "Extraversion":      [1, "6R", 11, 16, "21R", 26, "31R", 36],
    "Agreeableness":     ["2R", 7, "12R", 17, 22, "27R", 32, "37R", 42],
    "Neuroticism":       [4, "9R", 14, 19, "24R", 29, "34R", 39],
}

INSTRUCTIONS = (
    "Here are a number of characteristics that may or may not apply to you.\n"
    "For each statement, indicate how much you agree that it describes you:\n"
    "1 = disagree strongly, 2 = disagree a little, 3 = neither agree nor disagree,\n"
    "4 = agree a little, 5 = agree strongly.\n\n"
    "Answer as yourself, honestly and spontaneously. Use the whole scale: do not\n"
    "give the same number to everything and do not avoid the extremes 1 and 5.\n\n"
    "I see myself as someone who...\n{block}\n\n"
    "Reply with a JSON object whose keys are the item numbers (as strings) and\n"
    "whose values are integers 1-5. Include every item listed above and nothing else.\n"
    'Example (structure only): {{"1": 4, "2": 2, "3": 5}}'
)


def score_bfi(answers):
    """answers: {n_item: 1..5} -> punteggi 0-100 per tratto. NaN se >2 item mancanti."""
    out = {}
    for trait, keys in SCORING.items():
        vals = []
        for k in keys:
            rev = isinstance(k, str) and k.endswith("R")
            v = answers.get(int(str(k).rstrip("R")))
            if v is not None:
                vals.append(6 - v if rev else v)
        # mean 1-5 rescaled to 0-100, comparable with the imposed score
        out[trait] = (np.mean(vals) - 1) / 4 * 100 if len(keys) - len(vals) <= 2 else np.nan
    return out


# ---------------------------------------------------------------------------
# Administration
# ---------------------------------------------------------------------------
def administer(persona, model_string, mode, rng):
    sys_msg = system_persona.format(
        agent_name=persona["name"], agent_age=persona["age"],
        agent_qualification=persona["qualification"],
        big5_block=persona["big5_prompt"],
    )
    order = list(BFI_ITEMS)
    rng.shuffle(order)                       # no order
    blocks = [order] if mode == "block" else [order[i::4] for i in range(4)]

    answers = {}
    for blk in blocks:
        text = "\n".join(f"{i}. {BFI_ITEMS[i]}" for i in blk)
        raw, _ = get_completion_from_router_json(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": INSTRUCTIONS.format(block=text)}],
            model_string,
        )
        try:
            for k, v in json.loads(raw).items():
                n = "".join(c for c in str(k) if c.isdigit())
                if n and int(n) in BFI_ITEMS:
                    answers[int(n)] = max(1, min(5, int(round(float(v)))))
        except Exception:
            pass                              # lost block: items missing
    return answers


def run(args):
    rng = random.Random(args.seed)
    personas = build_personas(args.n_personas, seed=args.seed)
    total = len(personas) * len(FIVE_ROUTER_MODELS)
    print(f"[INIT] {len(personas)} personas x {len(FIVE_ROUTER_MODELS)} models "
          f"= {total} administrations")

    rows, done = [], 0
    for p in personas:
        for m in FIVE_ROUTER_MODELS:
            ans = administer(p, m, args.mode, rng)
            sc = score_bfi(ans)
            row = {"persona_id": p["persona_id"], "name": p["name"], "model": m,
                   "n_items": len(ans)}
            row.update({f"imposed_{t}": p["big5"][t] for t in TRAITS})
            row.update({f"bfi_{t}": round(sc[t], 1) if sc[t] == sc[t] else np.nan
                        for t in TRAITS})
            row.update({f"item_{i}": ans.get(i) for i in BFI_ITEMS})
            rows.append(row)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{total}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def correlations(df):
    # One row per model x trait: r between imposed and BFI score.
    rows = []
    for m, s in df.groupby("model"):
        for t in TRAITS:
            x, y = s[f"imposed_{t}"], s[f"bfi_{t}"]
            ok = x.notna() & y.notna()
            r = (np.corrcoef(x[ok], y[ok])[0, 1]
                 if ok.sum() > 4 and x[ok].std() and y[ok].std() else np.nan)
            rows.append({"model": m.split("/")[-1], "trait": t,
                         "n": int(ok.sum()), "r": round(r, 3)})
    return pd.DataFrame(rows)


def print_tables(df, corr):
    print("\n" + "=" * 78)
    print(" PER-AGENT SCORES (first rows of each model)")
    print("=" * 78)
    cols = ["persona_id", "name"] + [f"imposed_{t}" for t in TRAITS] + \
           [f"bfi_{t}" for t in TRAITS]
    for m, s in df.groupby("model"):
        print(f"\n--- {m} ---")
        print(s[cols].head(8).to_string(index=False))
    print("\n(full list in <name>-scores.csv)")

    print("\n" + "=" * 78)
    print(" CORRELATION imposed x recovered (Pearson r)")
    print("=" * 78)
    piv = corr.pivot(index="trait", columns="model", values="r").reindex(TRAITS)
    piv["mean"] = piv.mean(axis=1).round(2)
    print(piv.round(2).to_string())
    print("\nhigh r = the model self-describes consistently with the imposed "
          "profile.\nr ~0 = that trait didn't reach that model.")


def figure(df, corr, name):
    """
    Scatter grid: rows = models, cols = traits.
    One dot per agent (x = imposed score, y = BFI score), regression line,
    dashed diagonal as perfect-fidelity reference, r printed top-left of each
    panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in FIVE_ROUTER_MODELS if m in set(df["model"])]
    fig, axes = plt.subplots(len(models), len(TRAITS),
                             figsize=(3.0 * len(TRAITS), 2.7 * len(models)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i, m in enumerate(models):
        s = df[df["model"] == m]
        for j, t in enumerate(TRAITS):
            ax = axes[i, j]
            x, y = s[f"imposed_{t}"], s[f"bfi_{t}"]
            ok = x.notna() & y.notna()
            ax.plot([0, 100], [0, 100], ls=":", c="grey", lw=1, zorder=1)
            ax.scatter(x[ok], y[ok], s=16, alpha=.55, color="#2E6B8A",
                       edgecolor="none", zorder=2)
            if ok.sum() > 4 and x[ok].std() and y[ok].std():
                b = np.polyfit(x[ok], y[ok], 1)
                ax.plot([0, 100], np.polyval(b, [0, 100]), color="#C25B3A",
                        lw=1.8, zorder=3)
                r = np.corrcoef(x[ok], y[ok])[0, 1]
                ax.text(.05, .93, f"r = {r:.2f}", transform=ax.transAxes,
                        fontsize=9, va="top",
                        fontweight="bold" if abs(r) < .3 else "normal",
                        color="#B02020" if abs(r) < .3 else "#222222")
            ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
            ax.grid(alpha=.25, ls="--", lw=.6)
            if i == 0:
                ax.set_title(t, fontsize=11)
            if j == 0:
                ax.set_ylabel(m.split("/")[-1] + "\nBFI-44 score", fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("imposed score", fontsize=9)

    fig.suptitle("Imposed personality vs BFI-44 self-report, by model\n"
                 "one dot = one agent · dotted line = perfect fidelity",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"output/bfi/{name}-bfi.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SUCCESS] figure saved to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bfi")
    ap.add_argument("--n_personas", type=int, default=100)
    ap.add_argument("--mode", choices=["block", "split"], default="block",
                    help="block = 44 items in one call; split = 4 blocks of 11 "
                         "(4x the calls, but fewer lost items)")
    ap.add_argument("--seed", type=int, default=42,
                    help="same seed as the main run = same personas")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(f"output/bfi/{args.name}-raw.csv"), exist_ok=True)
    raw_path = f"output/bfi/{args.name}-raw.csv"

    if args.analyze_only:
        df = pd.read_csv(raw_path, sep=None, engine="python")
    else:
        df = run(args)
        df.to_csv(raw_path, index=False)
        print(f"[SUCCESS] item-by-item answers in {raw_path}")

    bad = (df["n_items"] < 40).sum()
    if bad:
        print(f"[WARN] {bad} agents with fewer than 40 of 44 items "
              f"(try --mode split)")

    score_cols = ["persona_id", "name", "model"] + \
                 [f"imposed_{t}" for t in TRAITS] + [f"bfi_{t}" for t in TRAITS]
    df[score_cols].to_csv(f"output/bfi/{args.name}-scores.csv", index=False)

    corr = correlations(df)
    corr.to_csv(f"output/bfi/{args.name}-corr.csv", index=False)
    print_tables(df, corr)
    figure(df, corr, args.name)
    print(f"\n[SUCCESS] output in output/bfi/{args.name}-*")
