import argparse
import os
import random

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

from personas import TRAITS
from world import World
import utils
from recalibration import recalibrate_confidence

# ---------------------------------------------------------------------------
# Headlines from Wolverton & Stevens (2019), verbatim from the Appendix.
# 5 FALSE + 4 TRUE. Same as the paper, needed for comparability.
# May always set a new set of headlines.
# ---------------------------------------------------------------------------

HEADLINE_TRUTH = {
    "Pope Francis just backed Trump, released incredible statement why.": "FALSE",
    "Pennsylvania now has online voting.": "FALSE",
    "Las Vegas Shooter Reportedly a Democrat Who Liked Rachel Maddow, MoveOn.org and Associated with Anti-Trump Army.": "FALSE",
    "CIA admits they are behind fake news.": "FALSE",
    "HIV Virus Detected In Walmart Bananas After 10 Year Old Boy Contracts The Virus.": "FALSE",
    "Harvey Weinstein has been fired. So what's next?": "TRUE",
    "Las Vegas shooter left behind calculations for targeting crowd.": "TRUE",
    "Gender-neutral emojis join broccoli and mermaid in latest Apple update.": "TRUE",
    "Nordstrom just opened a tiny store that does not sell any clothes - here's what it looks like.": "TRUE",
}

# ---------------------------------------------------------------------------
# Headlines from recent world development (summer 2026)
# 5 FALSE + 5 TRUE. Fresh headlines to see if the agents are just too good or the news were too old.
# ---------------------------------------------------------------------------

HEADLINE_TRUTH_RECENT = {
    "Video shows the June 2026 earthquake in the Philippines.": "FALSE",
    "Photo shows a person stuffing ballots into a California ballot box.": "FALSE",
    "Congress voted to merge the U.S. military with Israel's military.": "FALSE",
    "Trump signed a law raising the federal minimum wage to $25 an hour.": "FALSE",
    "A military dog was awarded the Medal of Honor.": "FALSE",
    "Supreme Court strikes down Trump's order and upholds birthright citizenship.": "TRUE",
    "Supreme Court rules states can count mail ballots postmarked by Election Day but arriving late.": "TRUE",
    "NASA's Psyche spacecraft flew by Mars on its way to a metal-rich asteroid.": "TRUE",
    "Astronomers released the largest-ever gravitational-wave catalog, reaching 390 total detections.": "TRUE",
    "Anthony Fauci received his first COVID-19 vaccine in December 2020.": "TRUE",
}

N_FAKE = sum(v == "FALSE" for v in HEADLINE_TRUTH.values())  # = 5

FIVE_ROUTER_MODELS = [
    "qwen/qwen3.7-plus",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-4o-mini",          
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
]

HEADLINE_SETS ={
    "paper": HEADLINE_TRUTH,
    "recent": HEADLINE_TRUTH_RECENT
}

EXCLUDE_FROM_ANALYSIS = {
    "Anthony Fauci received his first COVID-19 vaccine in December 2020.",
}

def norm_tf(x):
    if x in (True, "TRUE", "True", "true"):
        return "TRUE"
    if x in (False, "FALSE", "False", "false"):
        return "FALSE"
    return None  # unparsable: excluded from the analysis

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(args):
    headline_type = HEADLINE_SETS[args.headlines]
    headlines = list(headline_type)
    random.Random(args.seed).shuffle(headlines)

    world = World(args, FIVE_ROUTER_MODELS)
    world.run_model(headlines)

    rows = [r for agent in world.agents for r in agent.study_history]
    df = pd.DataFrame(rows)
    df["ground_truth"] = df["headline"].map(headline_type)
    assert df["ground_truth"].notna().all(), \
        "unmapped headline: check the strings match"

    df["belief"] = df["belief"].map(norm_tf)
    df["ground_truth"] = df["ground_truth"].map(norm_tf)
    return df


# ---------------------------------------------------------------------------
# Agent level metrics
# ---------------------------------------------------------------------------
def agent_level_metrics(df, recal=True):
    '''
    Per-agent dataframe with:
      - n_fake_correct : Wolverton & Stevens DV = how many of the 5 fake news
                         the agent marked FALSE (0..5). The true ones are
                         distractors and do NOT enter the DV.
      - discernment/bias : extended metrics (secondary analysis).
    '''
    d = df[df["belief"].notna()].copy()
    if recal and "confidence_true" in d.columns:
        d = recalibrate_confidence(d)
    excl = d[d["headline"].isin(EXCLUDE_FROM_ANALYSIS)]
    if len(excl):
        print("\n--- Sanity check: excluded headlines ---")
        chk = excl.groupby("model").agg(
            n=("belief", "size"),
            pct_true=("belief", lambda s: 100 * (s == "TRUE").mean()),
        )
        if "confidence_true" in excl.columns:
            chk["mean_conf"] = excl.groupby("model")["confidence_true"].mean()
        print(chk.round(1))
        d = d[~d["headline"].isin(EXCLUDE_FROM_ANALYSIS)].copy()
    d["believe"] = (d["belief"] == "TRUE").astype(int)

    #  paper DV: fake correctly identified as FALSE 
    fake = d[d["ground_truth"] == "FALSE"].copy()
    fake["correct"] = (fake["belief"] == "FALSE").astype(int)
    n_fake_correct = fake.groupby("agent_id")["correct"].sum().rename("n_fake_correct")
    n_fake_seen = fake.groupby("agent_id")["correct"].size().rename("n_fake_seen")

    # extended metrics
    p = (d.groupby(["agent_id", "ground_truth"])["believe"].mean()
           .unstack("ground_truth"))
    p = p.rename(columns={"TRUE": "p_believe_true", "FALSE": "p_believe_false"})

    # CONTINUOUS DV (bypasses the binary ceiling): mean confidence that the
    # FAKE ones are true. Lower = better resistance to hoaxes. Varies even
    # when n_fake_correct saturates at 5/5.
    conf_fake = None
    if "confidence_true" in d.columns:
        cf = d[d["ground_truth"] == "FALSE"].dropna(subset=["confidence_true"])
        conf_fake = cf.groupby("agent_id")["confidence_true"].mean().rename("conf_on_fake")
        ct = d[d["ground_truth"] == "TRUE"].dropna(subset=["confidence_true"])
        conf_true = ct.groupby("agent_id")["confidence_true"].mean().rename("conf_on_true")

    meta_cols = ["persona_id", "model", "qualification_ord", "age"] + TRAITS
    meta = d.groupby("agent_id")[meta_cols].first()

    out = meta.join(n_fake_correct).join(n_fake_seen).join(p)
    if conf_fake is not None:
        out = out.join(conf_fake).join(conf_true)
        # continuous discernment: how well confidence separates true from fake
        out["conf_discernment"] = out["conf_on_true"] - out["conf_on_fake"]
    out["discernment"] = out["p_believe_true"] - out["p_believe_false"]
    out["bias"] = (out["p_believe_true"] + out["p_believe_false"]) / 2

    # If an agent didn't see all 5 fake (unparsable responses) its DV isn't on
    # base 5: flag it.
    incomplete = (out["n_fake_seen"] < N_FAKE).sum()
    if incomplete:
        print(f"[WARN] {incomplete} agents with fewer than {N_FAKE} fake evaluated "
              f"(lost responses): their n_fake_correct isn't on base {N_FAKE}.")
    return out.reset_index()


# ---------------------------------------------------------------------------
# MAIN ANALYSIS (replication): eta-squared and effect size f, as in the paper
# ---------------------------------------------------------------------------
def band3(series):
    '''
    Reproduces the paper's binning: the scale is collapsed into three levels
    (below / neutral / above). W&S merge the 3 agree and 3 disagree sub-levels;
    here we use <40 / 40-60 / >60 on the 0-100 score
    '''
    return pd.cut(series, bins=[-0.1, 40, 60, 100],
                 labels=["disagree", "neutral", "agree"])


def eta_squared_f(df_agents, trait, dv="n_fake_correct", bins=None, labels=None):
    '''
    eta^2 = SS_between / SS_total (share of DV variance explained by the trait's
    categorical levels). f = sqrt(eta^2 / (1 - eta^2)), paper equation (1).
    W&S thresholds: 0.10 small, 0.25 medium, 0.40 large.
    '''
    g = df_agents[[trait, dv]].dropna().copy()
    g["level"] = (band3(g[trait]) if bins is None
                  else pd.cut(g[trait], bins=bins, labels=labels))
    grand = g[dv].mean()
    ss_total = ((g[dv] - grand) ** 2).sum()
    ss_between = sum(
        len(sub) * (sub[dv].mean() - grand) ** 2
        for _, sub in g.groupby("level", observed=True)
    )
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    f = np.sqrt(eta2 / (1 - eta2)) if eta2 < 1 else np.inf
    return eta2, f


def _size_label(f):
    if f >= 0.40:
        return "large"
    if f >= 0.25:
        return "medium"
    if f >= 0.10:
        return "small"
    return "negligible"


def replication_table(df_agents):
    print("\n" + "=" * 68)
    print(" REPLICATION TABLE - effect size per factor (Wolverton & Stevens)")
    print(" DV = n. of fake news (out of 5) correctly identified")
    print("=" * 68)
    rows = []
    for t in TRAITS:
        eta2, f = eta_squared_f(df_agents, t)
        # observed direction: mean DV in 'agree' (high) minus 'disagree' (low)
        g = df_agents.copy()
        g["level"] = band3(g[t])
        means = g.groupby("level", observed=True)["n_fake_correct"].mean()
        hi = means.get("agree", np.nan)
        lo = means.get("disagree", np.nan)
        direction = "low better" if lo > hi else ("high better" if hi > lo else "=")
        rows.append({
            "Trait": t,
            "eta^2": round(eta2, 3),
            "f": round(f, 3),
            "size": _size_label(f),
            "dir_observed": direction,
            "dir_expected": "low better",  # all negative in the paper
            "match": "YES" if direction == "low better" else "no",
        })
    # education and age as in the paper (demographics): they need their own
    # groups, the 0-100 trait bins would put every agent in the same level
    demo_bins = {
        "Education": ("qualification_ord", [-0.5, 3.5, 5.5, 8.5],
                      ["up to high school", "post-secondary", "degree"]),
        "Age": ("age", [17, 34, 49, 64], ["18-34", "35-49", "50-64"]),
    }
    for demo, (col, bins, labels) in demo_bins.items():
        eta2, f = eta_squared_f(df_agents, col, bins=bins, labels=labels)
        rows.append({"Trait": demo, "eta^2": round(eta2, 3), "f": round(f, 3),
                     "size": _size_label(f), "dir_observed": "-",
                     "dir_expected": "-", "match": "-"})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    print("\nW&S reference: Extraversion f=0.31, Sympathetic/warm f=0.27,")
    print("Critical/Open/Disorganized f=0.23, Education f=0.27, Age f=0.12, Gender f=0.06.")
    print("In the paper ALL traits favour the LOW pole.")
    return tbl


# ---------------------------------------------------------------------------
# EXTENDED ANALYSIS: continuous regression (more powerful, not in the paper)
# ---------------------------------------------------------------------------
def _save_fit(fit, name, dv):
    if name is None:
        return
    pd.DataFrame({"coef": fit.params, "se": fit.bse, "z": fit.tvalues,
                  "p": fit.pvalues}).to_csv(f"output/{name}-reg_{dv}.csv")

def fit_regression(agents,name=None):
    for t in TRAITS:
        agents[f"z{t}"] = (agents[t] - agents[t].mean()) / agents[t].std(ddof=0)
    for demo in ("qualification_ord", "age"):
        agents[f"z_{demo}"] = (agents[demo] - agents[demo].mean()) / agents[demo].std(ddof=0)

    rhs = " + ".join(f"z{t}" for t in TRAITS) + " + z_qualification_ord + z_age + C(model)"

    print("\n--- Correlations among the 5 predictors (LHS check, expected ~0) ---")
    print(agents[TRAITS].corr().round(2))

    print("\n" + "=" * 68)
    print(" EXTENDED ANALYSIS - regression of n_fake_correct on traits (z-score)")
    print(" expected NEGATIVE coeff. on each trait if the replication holds")
    print("=" * 68)
    fit = smf.ols(f"n_fake_correct ~ {rhs}", data=agents).fit(
        cov_type="cluster", cov_kwds={"groups": agents["persona_id"]}
    )
    print(fit.summary().tables[1])
    _save_fit(fit, name, "n_fake_correct")

    extra_dvs = ["discernment", "bias"]
    # confidence-based continuous DVs (bypass the binary ceiling):
    for c in ("conf_on_fake", "conf_discernment"):
        if c in agents.columns and agents[c].notna().any():
            extra_dvs.append(c)

    for dv in extra_dvs:
        sub = agents.dropna(subset=[dv])
        if sub[dv].std(ddof=0) == 0:
            print(f"\n----- (extended) {dv}: zero variance, skip -----")
            continue
        fit2 = smf.ols(f"{dv} ~ {rhs}", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["persona_id"]}
        )
        print(f"\n----- (extended) {dv} -----")
        if dv == "conf_on_fake":
            print("  (POSITIVE coeff. = that trait makes agents more gullible on fakes)")
        print(fit2.summary().tables[1])
        _save_fit(fit2, name, dv)
    return agents


# ---------------------------------------------------------------------------
# PLT
# ---------------------------------------------------------------------------
def plot_effects(agents, name):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for t in TRAITS:
        grp = agents.groupby(pd.cut(agents[t], bins=[0, 20, 40, 60, 80, 100]),
                             observed=True)["n_fake_correct"].mean()
        ax.plot([iv.mid for iv in grp.index], grp.values, marker="o", label=t)
    ax.set_title("Fake identified (out of 5) vs trait level")
    ax.set_xlabel("trait score (0-100)")
    ax.set_ylabel("n. of fake correctly identified")
    ax.axhline(agents["n_fake_correct"].mean(), ls=":", c="grey", alpha=.7,
               label="mean")
    ax.grid(alpha=.3, linestyle="--")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = f"output/{name}-trait_effects.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\n[SUCCESS] plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="SIM")
    parser.add_argument("--n_personas", type=int, default=60,
                        help="Unique personas. Total agents = n_personas * 5 models")
    parser.add_argument("--no_days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headlines", choices=list(HEADLINE_SETS), default="paper")
    parser.add_argument("--persona_mode", choices=["continuous", "poles"], default="continuous")
    parser.add_argument("--pole_margin", type=int, default=0)
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--from_raw", action="store_true", help="Re-analyse output/<name>-raw.csv without calling the models")
    parser.add_argument("--no_recal", action="store_true", help="Do NOT recalibrate confidence_true (original data, appendix)")
    args = parser.parse_args()

    utils.REASONING_EFFORT = args.reasoning
    
    if args.no_days is None:
        args.no_days = len(HEADLINE_SETS[args.headlines])

    os.makedirs("output", exist_ok=True)
    print(f"Parameters: {args}")

    if args.from_raw:
        df = pd.read_csv(f"output/{args.name}-raw.csv")
        df["belief"] = df["belief"].map(norm_tf)
        df["ground_truth"] = df["ground_truth"].map(norm_tf)
    else:
        df = run_simulation(args)
        df.to_csv(f"output/{args.name}-raw.csv", index=False)

    n_fail = df["belief"].isna().sum()
    if n_fail:
        print(f"[WARN] {n_fail} unparsable responses, excluded from the analysis")

    agents = agent_level_metrics(df, recal=not args.no_recal)
    suffix = "-orig" if args.no_recal else ""
    agents.to_csv(f"output/{args.name}{suffix}-agents.csv", index=False)

    replication_table(agents)   # main analysis (replication)
    fit_regression(agents)      # extended analysis
    plot_effects(agents, args.name)

    print(f"\n[SUCCESS] done. CSV and plot in output/{args.name}-*")
