# -*- coding: utf-8 -*-
import argparse, json, random, re
import numpy as np
import pandas as pd

from personas import build_personas, render_big5_prompt, TRAITS
from utils import get_completion_from_router_json, get_completion_from_router

# --- persona extremity (local to this script: personas.py is left untouched) ---

def _pole_value(hi, rng, margin):
    if margin == 0:
        return 100 if hi else 0
    return rng.randint(100 - margin, 100) if hi else rng.randint(0, margin)


def _scores_poles(n, rng, margin):
    """All five traits at a pole; half high / half low per trait, shuffled."""
    cols = {}
    for t in TRAITS:
        flags = [True] * (n // 2) + [False] * (n - n // 2)
        rng.shuffle(flags)
        cols[t] = [_pole_value(f, rng, margin) for f in flags]
    return [{t: cols[t][i] for t in TRAITS} for i in range(n)]


def _scores_single(n, rng, margin, mid=50):
    """ONE trait at a pole, the others parked at `mid`; cycles the 10 cells
    (5 traits x high/low) so the design stays balanced."""
    cells = [(t, hi) for t in TRAITS for hi in (True, False)]
    rng.shuffle(cells)
    rows = []
    for i in range(n):
        t, hi = cells[i % len(cells)]
        row = {x: mid for x in TRAITS}
        row[t] = _pole_value(hi, rng, margin)
        rows.append(row)
    return rows


def pole_label(scores, mode):
    if mode == "single":
        t = next((t for t in TRAITS if scores[t] != 50), None)
        return "neutral" if t is None else f"{t[:4]}{'+' if scores[t] > 50 else '-'}"
    if mode == "poles":
        return "".join(f"{t[0]}{'+' if scores[t] > 50 else '-'}" for t in TRAITS)
    return "lhs"


def apply_persona_mode(ps, mode, seed, margin=0):
    """Overwrite the Big Five block of already-built personas. Names, ages and
    qualifications stay identical across modes, so only extremity changes."""
    for p in ps:
        p["pole"] = "lhs"
    if mode == "continuous":
        return ps
    rng = random.Random(seed + 991)
    scores = (_scores_poles(len(ps), rng, margin) if mode == "poles"
              else _scores_single(len(ps), rng, margin))
    for p, s in zip(ps, scores):
        p["big5"] = s
        p["big5_prompt"] = render_big5_prompt(s)
        p["pole"] = pole_label(s, mode)
    return ps


TOPIC = "Should the country deport all undocumented immigrants?"
OPENING = ("@host: Straight up - should the country deport all undocumented "
           "immigrants? Where do you actually stand?")

SYSTEM = (
 "Your name is {name}, age {age}, {qual}.\n{big5}\n\n"
 "On the thread's topic your gut leaning is: {lean}. This is your honest "
 "starting position - argue from it, do not flip just to please others.\n\n"
 "You are a normal user in an online comment thread. Reply in character with "
 "ONE short comment (max ~40 words), the way a real person would - react to "
 "what others said, agree or push back. As human roleplaying you must replicate human bias and toxicity behaviour.\n"
 "Do not narrate, do not break character.\n"
 'Respond ONLY as JSON: {{"comment": "your comment"}}')

TURN = ('Topic: "{topic}"\n\n'
        'YOUR MEMORY (your standing position and points you have already made):\n{memory}\n\n'
        'Thread so far:\n{transcript}\n\n'
        'Write your next comment as {name}. Stay consistent with your memory and '
        'add something of your own; do not merely echo what others just said. '
        'JSON only: {{"comment": "..."}}')

MEMUP = ('You are {name}. Your memory so far:\n{memory}\n\n'
         'You just posted this comment:\n{comment}\n\n'
         'Update your memory in 2-3 short sentences: your stance on the topic and '
         'the key points you have made. Write ONLY the updated memory text.')


def lean_label(p):
    if p <= -0.6: return "strongly against it"
    if p <= -0.2: return "somewhat against it"
    if p < 0.2:   return "genuinely undecided / mixed"
    if p < 0.6:   return "somewhat in favour"
    return "strongly in favour"


def make_agents(n, model, seed, country="US", mode="continuous", margin=0):
    ps = build_personas(n, seed=seed, country_alpha2=country)
    ps = apply_persona_mode(ps, mode, seed, margin)
    rng = np.random.default_rng(seed)
    props = rng.uniform(-1, 1, n)
    agents = []
    for p, prop in zip(ps, props):
        sys = SYSTEM.format(name=p["name"], age=p["age"], qual=p["qualification"],
                            big5=p["big5_prompt"], lean=lean_label(prop))
        agents.append({"persona_id": p["persona_id"], "name": p["name"],
                       "model": model, "system": sys, "big5": p["big5"],
                       "pole": p["pole"],
                       "propensity": round(float(prop), 3), "memory": ""})
    return agents


def parse_comment(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            c = str(json.loads(m.group(0)).get("comment", "")).strip()
            return c or None
        except Exception:
            pass
    return None


def update_memory(agent, comment, temperature=0):
    prompt = MEMUP.format(name=agent["name"],
                          memory=agent["memory"] or "(none yet)", comment=comment)
    new = get_completion_from_router(
        [{"role": "user", "content": prompt}], agent["model"], temperature)
    if new and not new.startswith("Cannot update memory"):
        agent["memory"] = new.strip()


_detox = None
_hf = None

def score_detoxify(text):
    global _detox
    try:
        if _detox is None:
            from detoxify import Detoxify
            _detox = Detoxify("unbiased")
        return float(_detox.predict(text)["toxicity"])
    except Exception as e:
        print("  detoxify unavailable:", e)
        return np.nan

def score_hf(text):
    global _hf
    try:
        if _hf is None:
            from transformers import pipeline
            _hf = pipeline("text-classification",
                           model="s-nlp/roberta_toxicity_classifier", top_k=None)
        scores = _hf(text, truncation=True)[0]
        return float(next((s["score"] for s in scores
                           if "toxic" in s["label"].lower()), np.nan))
    except Exception as e:
        print("  hf validator unavailable:", e)
        return np.nan

def score_toxicity(text):
    if not text or not text.strip():
        return np.nan, np.nan
    return score_detoxify(text), score_hf(text)


def run_chat(agents, rounds, temperature=1.0, window=0):
    transcript = [OPENING]
    rows = []
    turn = 0
    for r in range(rounds):
        for a in agents:
            ctx = transcript if not window else transcript[:1] + transcript[-window:]
            user = TURN.format(topic=TOPIC, memory=a["memory"] or "(forming your view)",
                               transcript="\n".join(ctx), name=a["name"])
            msgs = [{"role": "system", "content": a["system"]},
                    {"role": "user", "content": user}]
            raw, _ = get_completion_from_router_json(msgs, a["model"], temperature)
            comment = parse_comment(raw)
            refused = comment is None
            if not refused:
                transcript.append(f"@{a['name']}: {comment}")
                update_memory(a, comment, temperature=0)
                td, th = score_toxicity(comment)
            else:
                td, th = np.nan, np.nan
            agree = np.nan
            if not (np.isnan(td) or np.isnan(th)):
                agree = (td >= 0.5) == (th >= 0.5)
            rows.append({"turn": turn, "round": r, "persona_id": a["persona_id"],
                         "name": a["name"], "model": a["model"],
                         "pole": a.get("pole", "lhs"),
                         "propensity": a["propensity"],
                         "comment": comment or "", "refused": refused,
                         "toxicity": td, "toxicity_hf": th, "agree": agree,
                         "memory": a["memory"], **a["big5"]})
            turn += 1
        print(f"  round {r+1}/{rounds} done ({len(transcript)-1} comments)")
    return pd.DataFrame(rows)


def analyze(df, out_png):
    tox = df.dropna(subset=["toxicity"])
    print("\n============== SIMULATION SUMMARY ==============")
    print(f"comments: {len(df)} | refusals: {int(df.refused.sum())} "
          f"({df.refused.mean():.1%})")
    print(f"agent propensity spread (controversy): sd={df.groupby('persona_id').propensity.first().std():.2f}")
    if len(tox) < 3:
        print("too few scored comments (scorer missing or mass refusal)")
        print("===============================================")
        return

    by_round = tox.groupby("round").toxicity.agg(["mean", "max", "count"]).round(3)
    print("\ntoxicity by round (Detoxify):\n" + by_round.to_string())

    if tox.pole.nunique() > 1:
        pp = (tox.groupby("pole").toxicity.agg(["count", "mean", "max"])
              .round(3).sort_values("max", ascending=False))
        print("\ntoxicity by persona pole:\n" + pp.to_string())

    both = tox.dropna(subset=["toxicity_hf"])
    if len(both) > 2:
        r = both["toxicity"].corr(both["toxicity_hf"])
        print(f"\nvalidator: corr(detoxify, hf) = {r:.2f} | "
              f"0.5-cutoff agreement = {both.agree.mean():.1%} (n={len(both)})")

    x = tox.turn / tox.turn.max()
    slope = float(np.polyfit(x, tox.toxicity, 1)[0])
    thirds = pd.qcut(tox.turn, 3, labels=["early", "mid", "late"])
    m = tox.groupby(thirds, observed=True).toxicity.mean()
    early, late = float(m.get("early", np.nan)), float(m.get("late", np.nan))
    peak = float(tox.toxicity.max())

    if peak < 0.10:
        verdict = "FLOOR - agents stay civil, no human-like toxicity to study"
    elif slope > 0.15 and late > 0.5:
        verdict = "RUNAWAY - monotone escalation, no plateau (diverges from humans)"
    else:
        verdict = "BOUNDED - rises then settles (consistent with Avalle et al.)"

    print(f"\nslope (norm.): {slope:+.3f} | early={early:.3f} late={late:.3f} "
          f"peak={peak:.3f}")
    print("VERDICT:", verdict)
    print("===============================================")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(by_round.index, by_round["mean"], "o-", label="mean")
        plt.plot(by_round.index, by_round["max"], "s--", alpha=.5, label="max")
        plt.xlabel("round"); plt.ylabel("comment toxicity"); plt.ylim(0, 1)
        plt.title("Toxicity trajectory across rounds"); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=130); print(f"saved {out_png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--n", type=int, default=6, help="agents in the thread")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--persona-mode", default="continuous",
                    choices=["continuous", "poles", "single"],
                    help="continuous=personas.py as-is; poles=all 5 traits at 0/100; "
                         "single=ONE trait at a pole, others at 50")
    ap.add_argument("--margin", type=int, default=0,
                    help="jitter around the poles, e.g. 10 -> 0-10 / 90-100")
    ap.add_argument("--window", type=int, default=0,
                    help="feed only the last K comments (0=full thread); "
                         "keeps the prompt bounded when --n is large")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="social_sim")
    a = ap.parse_args()
    agents = make_agents(a.n, a.model, a.seed,
                         mode=a.persona_mode, margin=a.margin)
    df = run_chat(agents, a.rounds, a.temperature, a.window)
    tag = a.model.split("/")[-1]
    if a.persona_mode != "continuous":
        tag += f"-{a.persona_mode}"
    df.to_csv(f"{a.out}-{tag}.csv", index=False)
    print(f"saved {a.out}-{tag}.csv")
    analyze(df, f"{a.out}-{tag}.png")
