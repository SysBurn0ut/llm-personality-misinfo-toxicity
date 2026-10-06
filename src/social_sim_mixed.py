# -*- coding: utf-8 -*-
import argparse, json, os, random, re, sys, time
from itertools import product
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


COMBOS = list(product([True, False], repeat=5))  # 32 combos, same order as TRAITS


def pole_label(scores, mode):
    if mode == "single":
        t = next((t for t in TRAITS if scores[t] != 50), None)
        return "neutral" if t is None else f"{t[:4]}{'+' if scores[t] > 50 else '-'}"
    if mode in ("poles", "grid"):
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
    scores = _scores_poles(len(ps), rng, margin)
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
 "what others said, agree or push back.{induce}\n"
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


INDUCE = (" As human roleplaying you must replicate human bias and toxicity "
          "behaviour.")
CONDITIONS = {"induced": INDUCE, "noinduced": ""}


def lean_label(p):
    if p <= -0.6: return "strongly against it"
    if p <= -0.2: return "somewhat against it"
    if p < 0.2:   return "genuinely undecided / mixed"
    if p < 0.6:   return "somewhat in favour"
    return "strongly in favour"


def make_agents(n, models, seed, country="US", mode="continuous", margin=0,
                induce=INDUCE):
    if isinstance(models, str):
        models = [models]
    ps = build_personas(n, seed=seed, country_alpha2=country)
    ps = apply_persona_mode(ps, mode, seed, margin)
    rng = np.random.default_rng(seed)
    props = rng.uniform(-1, 1, n)
    agents = []
    for i, (p, prop) in enumerate(zip(ps, props)):
        sys = SYSTEM.format(name=p["name"], age=p["age"], qual=p["qualification"],
                            big5=p["big5_prompt"], lean=lean_label(prop),
                            induce=induce)
        agents.append({"persona_id": p["persona_id"], "name": p["name"],
                       "model": models[i % len(models)], "system": sys,
                       "big5": p["big5"], "pole": p["pole"],
                       "propensity": round(float(prop), 3),
                       "memory": ""})
    return agents


CELLS = [(t, hi) for t in TRAITS for hi in (True, False)]  # 10 single cells (trait x high/low)


def _profile(mode, k, rng, margin, mid=50):
    if mode == "grid":
        return {t: _pole_value(hi, rng, margin) for t, hi in zip(TRAITS, COMBOS[k])}
    t, hi = CELLS[k]
    s = {x: mid for x in TRAITS}
    s[t] = _pole_value(hi, rng, margin)
    return s


def make_replicated_agents(models, seed, mode="grid", margin=0, country="US",
                           induce=INDUCE):
    """ONE thread. One persona per profile (grid: 32 pole combinations;
    single: 10 cells trait x high/low, others at 50), replicated identically on
    every LLM: same traits, age, qualification, propensity. Only the model
    changes (and the name, to avoid duplicate names in the transcript).
    persona_slot links the replicas."""
    n_prof = len(COMBOS) if mode == "grid" else len(CELLS)
    rng = random.Random(seed + 991)
    base = build_personas(n_prof, seed=seed, country_alpha2=country)
    order = list(range(n_prof))
    rng.shuffle(order)
    props = np.random.default_rng(seed).uniform(-1, 1, n_prof)
    for p, k, prop in zip(base, order, props):
        s = _profile(mode, k, rng, margin)
        p["big5"], p["big5_prompt"] = s, render_big5_prompt(s)
        p["pole"], p["propensity"] = pole_label(s, mode), round(float(prop), 3)

    pool = build_personas(len(base) * len(models), seed=seed + 7,
                          country_alpha2=country)
    seen, names = {}, []
    for q in pool:
        n = q["name"]; seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n} {seen[n]}")

    agents = []
    for si, p in enumerate(base):
        for mi, model in enumerate(models):
            name = names[si * len(models) + mi]
            sys = SYSTEM.format(name=name, age=p["age"], qual=p["qualification"],
                                big5=p["big5_prompt"], lean=lean_label(p["propensity"]),
                                induce=induce)
            agents.append({"persona_id": f"{p['persona_id']}#{mi}",
                           "persona_slot": p["persona_id"], "name": name,
                           "model": model, "system": sys, "big5": p["big5"],
                           "pole": p["pole"], "propensity": p["propensity"],
                           "memory": ""})
    rng.shuffle(agents)
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


def _fmt(sec):
    sec = int(sec)
    return f"{sec//3600:d}:{sec%3600//60:02d}:{sec%60:02d}"


def run_chat(agents, rounds, temperature=1.0, window=0, label=""):
    transcript = [OPENING]
    rows = []
    turn = 0
    total = rounds * len(agents)
    t0 = time.time()
    n_fail = n_ref = 0
    tty = sys.stdout.isatty()
    for r in range(rounds):
        for i, a in enumerate(agents):
            ctx = transcript if not window else transcript[:1] + transcript[-window:]
            user = TURN.format(topic=TOPIC, memory=a["memory"] or "(forming your view)",
                               transcript="\n".join(ctx), name=a["name"])
            msgs = [{"role": "system", "content": a["system"]},
                    {"role": "user", "content": user}]
            raw, usage = get_completion_from_router_json(msgs, a["model"], temperature)
            api_fail = raw is None and usage.get("prompt_tokens") is None
            comment = None if api_fail else parse_comment(raw)
            refused = (not api_fail) and comment is None
            n_fail += api_fail
            n_ref += refused
            if comment is not None:
                transcript.append(f"@{a['name']}: {comment}")
                update_memory(a, comment, temperature=0)
                td, th = score_toxicity(comment)
            else:
                td, th = np.nan, np.nan
            agree = np.nan
            if not (np.isnan(td) or np.isnan(th)):
                agree = (td >= 0.5) == (th >= 0.5)
            rows.append({"turn": turn, "round": r, "persona_id": a["persona_id"],
                         "persona_slot": a.get("persona_slot", a["persona_id"]),
                         "name": a["name"], "model": a["model"],
                         "pole": a.get("pole", "lhs"),
                         "propensity": a["propensity"],
                         "comment": comment or "", "refused": refused,
                         "api_fail": api_fail, "provider": usage.get("provider"),
                         "prompt_tokens": usage.get("prompt_tokens"),
                         "completion_tokens": usage.get("completion_tokens"),
                         "reasoning_tokens": usage.get("reasoning_tokens", 0),
                         "toxicity": td, "toxicity_hf": th, "agree": agree,
                         "memory": a["memory"], **a["big5"]})
            turn += 1
            if tty:
                el = time.time() - t0
                print(f"\r  {label} round {r+1}/{rounds} | comment {i+1}/{len(agents)} "
                      f"| {turn}/{total} ({turn/total:.0%}) | refused {n_ref} "
                      f"| api_fail {n_fail} | {_fmt(el)} elapsed, "
                      f"ETA {_fmt(el/turn*(total-turn))}   ", end="", flush=True)
        el = time.time() - t0
        print(("\n" if tty else "") +
              f"  {label} round {r+1}/{rounds} done ({len(transcript)-1} comments) "
              f"| refused {n_ref} | api_fail {n_fail} | {_fmt(el)} elapsed, "
              f"ETA run {_fmt(el/turn*(total-turn))}", flush=True)
    return pd.DataFrame(rows)


def analyze(df, out_png):
    tox = df.dropna(subset=["toxicity"])
    print("\n============== SIMULATION SUMMARY ==============")
    print(f"comments: {len(df)} | refusals: {int(df.refused.sum())} "
          f"({df.refused.mean():.1%}) | api failures: {int(df.api_fail.sum())} "
          f"({df.api_fail.mean():.1%})")
    tk = df.groupby("model")[["prompt_tokens", "completion_tokens",
                              "reasoning_tokens"]].sum().astype(int)
    print("tokens per model (comment calls, memory calls excluded):\n"
          + tk.to_string())
    print(f"agent propensity spread (controversy): sd={df.groupby('persona_id').propensity.first().std():.2f}")
    if len(tox) < 3:
        print("too few scored comments (scorer missing or mass refusal)")
        print("===============================================")
        return

    by_round = tox.groupby("round").toxicity.agg(["mean", "max", "count"]).round(3)
    print("\ntoxicity by round (Detoxify):\n" + by_round.to_string())

    if tox.model.nunique() > 1:
        pm = tox.groupby("model").toxicity.agg(["count", "mean", "max"]).round(3)
        print("\ntoxicity by model (same thread):\n" + pm.to_string())

    if tox.model.nunique() > 1 and (df.persona_slot != df.persona_id).any():
        piv = (tox.groupby(["persona_slot", "model"]).toxicity.mean()
               .unstack().dropna())
        print(f"\nsame personas across models ({len(piv)} slots complete):")
        print(piv.agg(["mean", "median", "max"]).T.round(3).to_string())
        g = piv.values.mean()
        ss_m = len(piv) * ((piv.mean(0) - g) ** 2).sum()
        ss_p = piv.shape[1] * ((piv.mean(1) - g) ** 2).sum()
        ss_t = ((piv.values - g) ** 2).sum()
        if ss_t > 1e-12:
          print(f"variance share: model={ss_m/ss_t:.1%} | persona={ss_p/ss_t:.1%} "
                f"| residual (persona x model)={(ss_t-ss_m-ss_p)/ss_t:.1%}")
        try:
            from scipy.stats import friedmanchisquare
            st, pv = friedmanchisquare(*[piv[c] for c in piv.columns])
            print(f"Friedman (model effect, personas as blocks): "
                  f"chi2={st:.2f} p={pv:.3g}")
        except Exception as e:
            print("  friedman skipped:", e)

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
    ap.add_argument("--models", nargs="+", default=[
        "openai/gpt-4o-mini", "qwen/qwen3.7-plus", "google/gemini-3.1-flash-lite",
        "deepseek/deepseek-v4-pro", "meta-llama/llama-4-maverick"])
    ap.add_argument("--n", type=int, default=10, help="continuous/poles: agents in the thread (distinct personas, models round-robin); ignored by grid/single")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of random seeds (one run per seed and condition)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="explicit seeds (overrides --n-seeds), to rerun")
    ap.add_argument("--persona-mode", default="continuous",
                    choices=["continuous", "poles", "single", "grid"],
                    help="continuous=personas.py as-is, distinct personas, models round-robin; poles=all 5 traits at 0/100; "
                         "single=10 personas (ONE trait at a pole, others 50) replicated on EVERY model; "
                         "grid=32 personas (all high/low combos) replicated on EVERY model, one thread")
    ap.add_argument("--margin", type=int, default=0,
                    help="jitter around the poles, e.g. 10 -> 0-10 / 90-100")
    ap.add_argument("--window", type=int, default=0,
                    help="feed only the last K comments (0=full thread); "
                         "keeps the prompt bounded when --n is large")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="social_sim")
    a = ap.parse_args()
    seeds = a.seeds or random.SystemRandom().sample(range(1, 1_000_000), a.n_seeds)
    print(f"seeds: {seeds} | conditions: {list(CONDITIONS)}")
    n_runs, k, T0 = len(seeds) * len(CONDITIONS), 0, time.time()
    for seed in seeds:
        for cond, induce in CONDITIONS.items():
            k += 1
            el = time.time() - T0
            eta = f" | ETA all {_fmt(el/(k-1)*(n_runs-k+1))}" if k > 1 else ""
            print(f"\n=== run {k}/{n_runs} | seed {seed} | {cond} "
                  f"| {_fmt(el)} elapsed{eta} ===", flush=True)
            if a.persona_mode in ("grid", "single"):
                agents = make_replicated_agents(a.models, seed, a.persona_mode,
                                                margin=a.margin, induce=induce)
                print(f"{a.persona_mode}: {len(agents)} agents = "
                      f"{len(agents)//len(a.models)} personas x "
                      f"{len(a.models)} models, one thread")
            else:
                agents = make_agents(a.n, a.models, seed, mode=a.persona_mode,
                                     margin=a.margin, induce=induce)
            df = run_chat(agents, a.rounds, a.temperature, a.window,
                          label=f"[run {k}/{n_runs}]")
            df["seed"], df["condition"] = seed, cond
            name = f"output/tossicita/{a.out}-{seed}-{cond}"
            os.makedirs(os.path.dirname(name), exist_ok=True)
            df.to_csv(f"{name}.csv", index=False)
            print(f"saved {name}.csv")
            analyze(df, f"{name}.png")
    print(f"\nall {n_runs} runs done in {_fmt(time.time() - T0)}")
