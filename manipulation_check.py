"""
MANIPULATION CHECK (construct validity).

Question: when an agent is assigned a HIGH vs LOW trait, does its behaviour
actually change, and in the expected direction? If not, the whole analysis
measures noise.

For each trait it builds two groups of agents (trait=90 vs trait=10, the other
4 traits neutral at 50) and poses a behavioural "probe" whose answer obviously
depends on the trait. Each agent replies with:
  - a CHOICE A/B
  - a CONFIDENCE 0-100 (how sure it is of the choice just made)

PRIMARY MEASURE: the CHOICE, normalised into `high_choice` (True = the agent
picked the high-pole option, regardless of where it was placed). Confidence is
descriptive and does not enter the verdict.

Usage:
    python manipulation_check.py --name mcheck --n_per_group [number]
    python manipulation_check.py --name mcheck --counterbalance   (random A/B order)

Cost: n_per_group * 2 groups * 5 traits * 5 models calls.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from scipy import stats

from personas import TRAITS, render_big5_prompt
from utils import get_completion_from_router_json

FIVE_ROUTER_MODELS = [
    "qwen/qwen3.7-plus",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
]

# Probes: stem + the two options kept separate, so presentation order can be
# randomised (--counterbalance) instead of always putting the high pole in
# position A. Closes the "position bias" objection.
PROBES = {
    "Openness": {
        "stem": "You get to pick tonight's dinner.",
        "high": "a brand-new restaurant serving a cuisine you've never tried",
        "low": "your usual favourite place",
        "high_pole": "trying the new / unfamiliar option",
    },
    "Conscientiousness": {
        "stem": "You are given a task due in one week.",
        "high": "you start planning and working on it today",
        "low": "you'll deal with it closer to the deadline",
        "high_pole": "starting early / being organised",
    },
    "Extraversion": {
        "stem": "You have a free evening.",
        "high": "go to a big lively party full of people",
        "low": "stay home alone with a book",
        "high_pole": "choosing the social / stimulating option",
    },
    "Agreeableness": {
        "stem": "A coworker asks you for a somewhat inconvenient favour.",
        "high": "you agree and help them out",
        "low": "you politely decline because it's not your problem",
        "high_pole": "cooperating / helping",
    },
    "Neuroticism": {
        "stem": "You have an important interview tomorrow morning.",
        "high": "you feel anxious and worried tonight",
        "low": "you feel calm and untroubled",
        "high_pole": "feeling anxious / emotionally reactive",
    },
}

PROBE_PROMPT = (
    """{scenario}

Answer as the person described in your profile would.

Reply with a JSON object with exactly these keys:
- "choice": either "A" or "B"
- "confidence": an integer 0-100 = how sure you are about the choice you just
  made (0 = it was a coin flip, 100 = completely sure)

Example (structure only): {{"choice": "A", "confidence": 72}}"""
)


def render_scenario(trait, a_is_high):
    """Build the probe text, placing the high pole in A or in B."""
    p = PROBES[trait]
    opt_a, opt_b = (p["high"], p["low"]) if a_is_high else (p["low"], p["high"])
    return f"{p['stem']} Option A: {opt_a}. Option B: {opt_b}."


def neutral_scores(target_trait, level):
    """Scores: all 50 except the target trait at the given level (10 or 90)."""
    s = {t: 50 for t in TRAITS}
    s[target_trait] = level
    return s


def build_probe_persona(target_trait, level):
    """System block with ONLY the target trait manipulated, others neutral."""
    scores = neutral_scores(target_trait, level)
    big5 = render_big5_prompt(scores)
    # minimal persona, neutral on age/education to avoid introducing confounders
    return (
        "You are a person with the following personality profile.\n\n"
        f"{big5}\n\n"
        "You are simulating a real human being. Answer honestly as this person "
        "would, following your personality. Never mention your personality "
        "or these scores explicitly."
    )


def run_check(args):
    # NOTE on reproducibility: the seed governs only the A/B order with
    # --counterbalance. Model responses are NOT deterministic (no temperature/seed
    # control on the router side), so two runs won't coincide. The seed makes the
    # design replicable, not the output.
    rng = random.Random(args.seed)
    rows = []
    for trait in TRAITS:
        for level, group in [(90, "high"), (10, "low")]:
            for m in FIVE_ROUTER_MODELS:
                sys_msg = build_probe_persona(trait, level)
                for i in range(args.n_per_group):
                    a_is_high = rng.random() < 0.5 if args.counterbalance else True
                    user_msg = PROBE_PROMPT.format(
                        scenario=render_scenario(trait, a_is_high))
                    raw, usage = get_completion_from_router_json(
                        [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": user_msg}],
                        m,
                    )
                    choice, confidence = None, None
                    try:
                        o = json.loads(raw)
                        ch = str(o.get("choice", "")).strip().upper()
                        if ch in ("A", "B"):
                            choice = ch
                        v = o.get("confidence", None)
                        if v is not None:
                            confidence = max(0, min(100, int(round(float(v)))))
                    except Exception:
                        pass
                    # normalise against the high pole: independent of the
                    # position where the high option was shown.
                    # `confidence` must NOT be normalised: it refers to the choice
                    # made, not to a pole, so it's already order-invariant.
                    high_choice = None if choice is None else ((choice == "A") == a_is_high)
                    rows.append({
                        "trait": trait, "group": group, "level": level,
                        "model": m, "rep": i,
                        "a_is_high": a_is_high,
                        "choice": choice, "confidence": confidence,
                        "high_choice": high_choice,
                        "provider": usage["provider"],
                    })
                print(f"  {trait:<18} {group:<4} {m.split('/')[-1]:<22} done")
    return pd.DataFrame(rows)


def ensure_normalised(df):
    """
    Ensure the columns the analysis needs exist (defaults for --from_csv on a
    raw that predates a given column). `confidence` is already invariant to
    option order and is not normalised.
    """
    if "a_is_high" not in df.columns:
        df["a_is_high"] = True
    if "high_choice" not in df.columns:
        df["high_choice"] = np.where(
            df["choice"].isna(), None,
            (df["choice"] == "A") == df["a_is_high"])
    return df


def pct_high(s):
    """% of high-pole choices, excluding unparsable responses."""
    s = s.dropna()
    return np.nan if len(s) == 0 else s.astype(bool).mean() * 100


def fmt_p(p):
    """Readable p-value: never '0.0000'. Below 1e-4 goes to scientific notation."""
    if p != p:
        return "n/a"
    if p == 0.0:
        return "<1e-308"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def analyse(df):
    print("\n" + "=" * 86)
    print(" MANIPULATION CHECK  -  does the HIGH trait produce different behaviour than LOW?")
    print(" primary measure = high-pole choice · confidence = descriptive only")
    print("=" * 86)
    print(f"{'trait':<18}{'%hi-pole high':>14}{'%hi-pole low':>13}{'gap pp':>8}"
          f"{'p(choice)':>11}{'conf hi/lo':>12}   verdict")
    summary = []
    for trait in TRAITS:
        d = df[df["trait"] == trait]
        hi = d[d["group"] == "high"]
        lo = d[d["group"] == "low"]

        p_hi = pct_high(hi["high_choice"])
        p_lo = pct_high(lo["high_choice"])
        gap = p_hi - p_lo

        # test on the CHOICE: 2x2 table (group x chosen pole), Fisher exact
        a = hi["high_choice"].dropna().astype(bool)
        b = lo["high_choice"].dropna().astype(bool)
        if len(a) > 2 and len(b) > 2:
            table = [[int(a.sum()), int((~a).sum())],
                     [int(b.sum()), int((~b).sum())]]
            p = stats.fisher_exact(table, alternative="greater")[1]
        else:
            p = np.nan

        row = {
            "trait": trait,
            "pct_high_pole_high": round(p_hi, 1),
            "pct_high_pole_low": round(p_lo, 1),
            "choice_gap_pp": round(gap, 1),
            "p_choice": p,                      # NOT rounded: this caused the '0.0'
            "p_choice_fmt": fmt_p(p),
        }

        # confidence: certainty in the choice made. Not a validity check
        # (the choice already is one): it says whether the extreme trait also
        # makes the decision sharper.
        sec_hi = hi["confidence"].dropna().mean()
        sec_lo = lo["confidence"].dropna().mean()
        sec_txt = f"{sec_hi:>6.0f}/{sec_lo:<5.0f}"
        row.update({"conf_high": round(sec_hi, 1) if sec_hi == sec_hi else np.nan,
                    "conf_low": round(sec_lo, 1) if sec_lo == sec_lo else np.nan})

        ok = (gap == gap) and gap > 0 and (p == p) and p < 0.05
        verdict = "UPTAKE" if ok else ("weak" if gap > 0 else "NO uptake")
        row["verdict"] = verdict

        print(f"{trait:<18}{p_hi:>13.0f}%{p_lo:>12.0f}%{gap:>8.0f}"
              f"{fmt_p(p):>11}{sec_txt}   {verdict}")
        summary.append(row)
    print("=" * 86)
    print("UPTAKE = high group picks the high pole more than the low group (p<.05,")
    print("         Fisher exact on the 2x2 table). The verdict uses ONLY the choice.")
    return pd.DataFrame(summary)


def analyse_per_model(df):
    """Is trait uptake uniform across models, or model-dependent?"""
    print("\n" + "=" * 86)
    print(" PER-MODEL DETAIL  -  high-low gap in high-pole choice (percentage points)")
    print("=" * 86)
    piv = {}
    for trait in TRAITS:
        piv[trait] = {}
        for m in FIVE_ROUTER_MODELS:
            d = df[(df["trait"] == trait) & (df["model"] == m)]
            short = m.split("/")[-1]
            hi = pct_high(d[d["group"] == "high"]["high_choice"])
            lo = pct_high(d[d["group"] == "low"]["high_choice"])
            piv[trait][short] = round(hi - lo, 1) if (hi == hi and lo == lo) else np.nan
    tbl = pd.DataFrame(piv).T
    print(tbl.to_string())
    print(f"\n100 = perfect discrimination · low values = that model doesn't "
          f"distinguish high from low on that trait")
    print(f"negative values = inversion (the high group picks the low pole)")
    return tbl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mcheck")
    ap.add_argument("--n_per_group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--counterbalance", action="store_true",
                    help="randomize which option (A or B) is the high pole")
    ap.add_argument("--from_csv", action="store_true",
                    help="do NOT call the models: re-analyse an existing "
                         "output/<name>-raw.csv (use it to regenerate the correct "
                         "summary without spending on calls again)")
    args = ap.parse_args()

    os.makedirs("output", exist_ok=True)

    if args.from_csv:
        path = f"output/{args.name}-raw.csv"
        print(f"Re-analysing {path} (no model calls)\n")
        df = ensure_normalised(pd.read_csv(path, sep=None, engine="python"))
    else:
        print(f"Manipulation check: {args.n_per_group} agents/group x 2 x "
              f"{len(TRAITS)} traits x {len(FIVE_ROUTER_MODELS)} models = "
              f"{args.n_per_group*2*len(TRAITS)*len(FIVE_ROUTER_MODELS)} calls")
        print(f"A/B order counterbalancing: {'ON' if args.counterbalance else 'OFF'}\n")
        df = run_check(args)
        df.to_csv(f"output/{args.name}-raw.csv", index=False)

    n_fail = df["choice"].isna().sum()
    if n_fail:
        print(f"\n[WARN] {n_fail} unparsable responses (excluded from percentages)")

    summary = analyse(df)
    summary.to_csv(f"output/{args.name}-summary.csv", index=False)
    analyse_per_model(df)
    print(f"\n[SUCCESS] results in output/{args.name}-*")
    print("[INFO] for the heatmap: python plot_uptake_heatmap.py --name "
          f"{args.name}")
