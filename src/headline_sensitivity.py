"""
Headline-level sensitivity analysis for conf_on_fake (study 1, "paper" headlines).

For each seed (42, 12, 73) it:
  1. builds a long dataset: one row = one agent x one FAKE headline;
  2. re-fits the agent-level regression of the thesis (conf_on_fake,
     cluster-robust SE by persona_id) as a reproduction check;
  3. fits the same specification on the single confidences, adding the
     headline as fixed effect, SE clustered by persona_id;
  4. reports the expected confidence per model (no reference model);
  5. (default run only) adds ONLY the model x headline interaction to the
     headline-level model and tests it with a joint Wald test, on both the
     recalibrated and the original data, plus a model x headline heatmap
     (the --no_recal run saves the same heatmap on the original data).

Same preprocessing as main.agent_level_metrics: unparsable belief dropped,
confidence_true recalibrated (recalibration.py), z-scores computed on the
agent-level table (one row per agent), as in main.fit_regression.

Usage:  python headline_sensitivity.py [--no_recal] [--out DIR]
        --no_recal : original (non-recalibrated) confidence_true, as main.py --no_recal;
                     default output folder becomes output/replica/HEADLINE-SENS-orig.
                     The interaction (step 5) already covers both data versions,
                     so it is computed only in the default run.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from personas import TRAITS
from recalibration import recalibrate_confidence

SEEDS = {12: "100A-S12-NR", 42: "100A-S42-NR", 73: "100A-S73-NR"}

# The 5 fake headlines of Wolverton & Stevens (same strings as main.HEADLINE_TRUTH)
FAKE_HEADLINES = {
    "Pope Francis just backed Trump, released incredible statement why.": "H1_pope",
    "Pennsylvania now has online voting.": "H2_pennsylvania",
    "Las Vegas Shooter Reportedly a Democrat Who Liked Rachel Maddow, MoveOn.org and Associated with Anti-Trump Army.": "H3_vegas_democrat",
    "CIA admits they are behind fake news.": "H4_cia",
    "HIV Virus Detected In Walmart Bananas After 10 Year Old Boy Contracts The Virus.": "H5_hiv_bananas",
}
DEMO = ["qualification_ord", "age"]
LABELS = {
    "C(model)[T.google/gemini-3.1-flash-lite]": "Gemini",
    "C(model)[T.meta-llama/llama-4-maverick]": "Llama",
    "C(model)[T.openai/gpt-4o-mini]": "GPT-4o-mini",
    "C(model)[T.qwen/qwen3.7-plus]": "Qwen",
    "zOpenness": "Openness", "zConscientiousness": "Conscientiousness",
    "zExtraversion": "Extraversion", "zAgreeableness": "Agreeableness",
    "zNeuroticism": "Neuroticism", "z_qualification_ord": "Titolo di studio",
    "z_age": "Eta",
}
MODELS = {
    "deepseek/deepseek-v4-pro": "DeepSeek",
    "google/gemini-3.1-flash-lite": "Gemini",
    "meta-llama/llama-4-maverick": "Llama",
    "openai/gpt-4o-mini": "GPT",
    "qwen/qwen3.7-plus": "Qwen",
}
HEADLINE_LABELS = {
    "H1_pope": "Pope Francis",
    "H2_pennsylvania": "Pennsylvania",
    "H3_vegas_democrat": "Las Vegas",
    "H4_cia": "CIA",
    "H5_hiv_bananas": "HIV bananas",
}
RHS = " + ".join(f"z{t}" for t in TRAITS) + " + z_qualification_ord + z_age + C(model)"


def norm_tf(x):
    s = str(x).strip().upper()
    return s if s in ("TRUE", "FALSE") else None


def build_long(seed, run, recal=True):
    raw = pd.read_csv(f"output/replica/{run}/{run}-raw.csv")
    raw["belief"] = raw["belief"].map(norm_tf)
    raw["ground_truth"] = raw["ground_truth"].map(norm_tf)
    d = raw[raw["belief"].notna()].copy()
    if recal:
        d = recalibrate_confidence(d, verbose=False)
    else:
        d["recalibrated"] = False

    # z-scores on the agent-level table, as in main.fit_regression
    meta = d.groupby("agent_id")[["persona_id", "model"] + DEMO + TRAITS].first()
    for c in TRAITS:
        meta[f"z{c}"] = (meta[c] - meta[c].mean()) / meta[c].std(ddof=0)
    for c in DEMO:
        meta[f"z_{c}"] = (meta[c] - meta[c].mean()) / meta[c].std(ddof=0)

    fake = d[d["ground_truth"] == "FALSE"].dropna(subset=["confidence_true"])
    assert set(fake["headline"]) <= set(FAKE_HEADLINES), "unmapped fake headline"
    long = fake[["agent_id", "headline", "confidence_true", "recalibrated"]].copy()
    long["headline_id"] = long["headline"].map(FAKE_HEADLINES)
    long = long.join(meta, on="agent_id")
    long.insert(0, "seed", seed)
    cols = (["seed", "agent_id", "persona_id", "model", "headline_id", "headline",
             "confidence_true", "recalibrated"] + TRAITS + DEMO
            + [f"z{t}" for t in TRAITS] + [f"z_{c}" for c in DEMO])
    return long[cols].rename(columns={"confidence_true": "confidence"}), meta


def tidy(fit, level, seed):
    t = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues})
    t.index.name = "term"
    t = t.reset_index()
    t.insert(0, "level", level)
    t.insert(0, "seed", seed)
    return t


def model_wald(fit):
    terms = [p for p in fit.params.index if p.startswith("C(model)")]
    # F version of the cluster-robust Wald test, as in analyze_per_model.py
    w = fit.wald_test(", ".join(f"{t} = 0" for t in terms), use_f=True, scalar=True)
    return float(w.statistic), float(w.pvalue)


def interaction_test(long):
    """Item-level model + ONLY model x headline; joint Wald on the interaction."""
    cl = {"groups": long["persona_id"]}
    base = smf.ols(f"confidence ~ {RHS} + C(headline_id)", data=long).fit(
        cov_type="cluster", cov_kwds=cl)
    full = smf.ols(f"confidence ~ {RHS} + C(headline_id) + C(model):C(headline_id)",
                   data=long).fit(cov_type="cluster", cov_kwds=cl)
    terms = [p for p in full.params.index if ":" in p]
    w = full.wald_test(", ".join(f"{t} = 0" for t in terms), use_f=True, scalar=True)
    return {"F": float(w.statistic), "df_num": int(w.df_num),
            "df_denom": int(w.df_denom), "p": float(w.pvalue),
            "R2_base": base.rsquared, "R2_int": full.rsquared,
            "delta_R2": full.rsquared - base.rsquared}


def heatmap(longs, path, title=None):
    """Mean confidence model x headline, one panel per seed, shared color scale."""
    fig, axes = plt.subplots(1, len(longs), figsize=(4.3 * len(longs), 3.9),
                             sharey=True, constrained_layout=True)
    tabs = {s: (l.pivot_table(index="model", columns="headline_id",
                              values="confidence", aggfunc="mean")
                 .reindex(index=list(MODELS), columns=list(HEADLINE_LABELS)))
            for s, l in longs.items()}
    vmax = max(t.values.max() for t in tabs.values())
    for ax, (seed, t) in zip(np.atleast_1d(axes), tabs.items()):
        im = ax.imshow(t.values, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=12,
                        color="white" if v > 0.6 * vmax else "black")
        ax.set_xticks(range(t.shape[1]), [HEADLINE_LABELS[h] for h in t.columns],
                      rotation=35, ha="right", fontsize=11)
        ax.set_yticks(range(t.shape[0]), [MODELS[m] for m in t.index], fontsize=11)
        ax.set_title(f"Seed {seed}", fontsize=12)
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("Veridicità attribuita (0-100)", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=13)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_interaction(out):
    """Model x headline interaction on recalibrated and original data."""
    rows, longs_recal = [], {}
    for rec, label in ((True, "ricalibrati"), (False, "non ricalibrati")):
        for seed, run in SEEDS.items():
            long, _ = build_long(seed, run, rec)
            if rec:
                longs_recal[seed] = long
            rows.append({"seed": seed, "dati": label, **interaction_test(long)})
    res = pd.DataFrame(rows)
    res["dati"] = pd.Categorical(res["dati"], ["ricalibrati", "non ricalibrati"])
    res["seed"] = pd.Categorical(res["seed"], list(SEEDS))
    res = res.sort_values(["seed", "dati"]).reset_index(drop=True)
    res.to_csv(f"{out}/headline_interaction.csv", index=False)

    tab = res[["seed", "dati", "F", "df_num", "df_denom", "p", "delta_R2"]].copy()
    tab["F"] = tab["F"].map(lambda x: f"{x:.1f}")
    tab["p"] = tab["p"].map(lambda x: "<.001" if x < .001 else f"{x:.3f}")
    tab["delta_R2"] = tab["delta_R2"].map(lambda x: f"{x:.3f}")
    tab.columns = ["Seed", "Dati", "F", "gdl num", "gdl den", "p", "Delta R2"]
    txt = ("Interazione modello x headline (Wald congiunto in forma F, SE cluster per persona)\n"
           + tab.to_string(index=False) + "\n")
    print("\n" + txt)
    with open(f"{out}/headline_interaction.txt", "w") as f:
        f.write(txt)
    heatmap(longs_recal, f"{out}/heatmap_model_headline.png", "Dati ricalibrati")


def main(out, recal=True):
    os.makedirs(out, exist_ok=True)
    longs, coefs, summ, means = [], [], [], []
    for seed, run in SEEDS.items():
        long, meta = build_long(seed, run, recal)
        longs.append(long)

        # (A) agent-level: conf_on_fake = mean of the agent's fake confidences
        ag = long.groupby("agent_id")["confidence"].mean().rename("conf_on_fake")
        ag = meta.join(ag).dropna(subset=["conf_on_fake"])
        fa = smf.ols(f"conf_on_fake ~ {RHS}", data=ag).fit(
            cov_type="cluster", cov_kwds={"groups": ag["persona_id"]})

        # reproduction check against the thesis outputs
        suffix = "" if recal else "-orig"
        ref = pd.read_csv(f"output/replica/{run}/{run}{suffix}-agents.csv").set_index("agent_id")
        dconf = (ag["conf_on_fake"] - ref["conf_on_fake"]).abs().max()
        msg = f"seed {seed}: max |conf_on_fake diff| vs {run}{suffix}-agents.csv = {dconf:.2e}"
        if recal:
            orig = pd.read_csv(f"output/replica/{run}/{run}-reg_conf_on_fake.csv", index_col=0)
            msg += f"; max |coef diff| vs reg CSV = {(fa.params - orig['coef']).abs().max():.2e}"
        print(msg)

        # (B) headline-level: single confidences + headline fixed effect
        fb = smf.ols(f"confidence ~ {RHS} + C(headline_id)", data=long).fit(
            cov_type="cluster", cov_kwds={"groups": long["persona_id"]})

        coefs += [tidy(fa, "agent", seed), tidy(fb, "headline", seed)]

        # same fits without reference model: expected confidence per model for an
        # agent with mean traits/demographics, averaged over the 5 headlines
        # (headline in deviation coding). Trait coefficients are unchanged.
        X = RHS.replace(" + C(model)", "")
        fa0 = smf.ols(f"conf_on_fake ~ 0 + C(model) + {X}", data=ag).fit(
            cov_type="cluster", cov_kwds={"groups": ag["persona_id"]})
        fb0 = smf.ols(f"confidence ~ 0 + C(model) + C(headline_id, Sum) + {X}",
                      data=long).fit(cov_type="cluster",
                                     cov_kwds={"groups": long["persona_id"]})
        for lvl, f0 in (("agent", fa0), ("headline", fb0)):
            m = [k for k in f0.params.index if k.startswith("C(model)")]
            means.append(pd.DataFrame({"seed": seed, "level": lvl,
                                       "model": [k[9:-1] for k in m],
                                       "mean": f0.params[m].values,
                                       "se": f0.bse[m].values}))
        wa, wb = model_wald(fa), model_wald(fb)
        hl = [p for p in fb.params.index if p.startswith("C(headline_id)")]
        wh = fb.wald_test(", ".join(f"{t} = 0" for t in hl), use_f=True, scalar=True)
        summ.append({"seed": seed, "n_agent": int(fa.nobs), "n_long": int(fb.nobs),
                     "clusters": long["persona_id"].nunique(),
                     "R2_agent": fa.rsquared, "R2_long": fb.rsquared,
                     "W_model_agent": wa[0], "W_model_long": wb[0],
                     "p_model_agent": wa[1], "p_model_long": wb[1],
                     "W_headline_long": float(wh.statistic),
                     "p_headline_long": float(wh.pvalue)})

    pd.concat(longs).to_csv(f"{out}/headline_long.csv", index=False)
    pd.concat(means).to_csv(f"{out}/headline_sens_model_means.csv", index=False)
    coefs = pd.concat(coefs)
    coefs.to_csv(f"{out}/headline_sens_coefs.csv", index=False)
    pd.DataFrame(summ).to_csv(f"{out}/headline_sens_summary.csv", index=False)

    # readable table: coef (SE), * p<.05 -- agent level vs headline level
    keep = ~coefs["term"].str.startswith(("Intercept", "C(headline_id)"))
    c = coefs[keep].copy()
    c["term"] = c["term"].map(lambda t: LABELS.get(t, t))
    c["cell"] = c.apply(lambda r: f"{r.coef:6.2f} ({r.se:.2f})"
                        + ("*" if r.p < .05 else " "), axis=1)
    c["col"] = c.apply(lambda r: f"S{r.seed} {'aggr.' if r.level == 'agent' else 'headline'}",
                       axis=1)
    order = [f"S{s} {l}" for s in SEEDS for l in ("aggr.", "headline")]
    tab = c.pivot_table(index="term", columns="col", values="cell",
                        aggfunc="first", sort=False)[order]
    tab = tab.reindex([v for v in LABELS.values() if v in tab.index])
    tab.to_csv(f"{out}/headline_sens_table.csv")

    s = pd.DataFrame(summ).set_index("seed")
    s = s.rename(columns={"n_agent": "N aggr.", "n_long": "N headline",
                          "clusters": "Cluster", "R2_agent": "R2 aggr.",
                          "R2_long": "R2 headline", "W_model_agent": "F Wald modello aggr.",
                          "W_model_long": "F Wald modello headline",
                          "W_headline_long": "F Wald FE headline",
                          "p_headline_long": "p FE headline",
                          "p_model_agent": "p modello aggr.",
                          "p_model_long": "p modello headline"})
    s.round(3).to_csv(f"{out}/headline_sens_summary.csv")

    pd.set_option("display.width", 200)
    print("\nCoef (SE cluster per persona), * p<.05 -- riferimento modello: DeepSeek -- " + ("dati ricalibrati" if recal else "dati NON ricalibrati"))
    print(tab.to_string())
    print()
    print(s.round(3).to_string())
    with open(f"{out}/headline_sens_table.txt", "w") as f:
        f.write("Coef (SE cluster per persona), * p<.05 -- riferimento modello: DeepSeek -- " + ("dati ricalibrati" if recal else "dati NON ricalibrati") + "\n")
        f.write(tab.to_string() + "\n\n" + s.round(3).to_string() + "\n")

    if recal:
        run_interaction(out)
    else:
        heatmap({l["seed"].iloc[0]: l for l in longs},
                f"{out}/heatmap_model_headline.png", "Dati non ricalibrati")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_recal", action="store_true",
                    help="Do NOT recalibrate confidence_true (original data, appendix)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or ("output/replica/HEADLINE-SENS-orig" if a.no_recal else "output/replica/HEADLINE-SENS")
    main(out, recal=not a.no_recal)
