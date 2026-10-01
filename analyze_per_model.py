"""
POST-HOC per-model analysis on the CSV already produced by main.py.
Does NOT run simulations: reads only output/<name>-agents.csv.

Question: is the effect of traits on fake-news gullibility (conf_on_fake)
consistent across the 5 models, or is it driven by some of them? The aggregate
analysis with C(model) controls each model's mean LEVEL but not whether traits
act DIFFERENTLY within each. Here we run 5 separate regressions + the formal
trait x model interaction test.

Usage:
    python analyze_per_model.py --name pilot60
    python analyze_per_model.py --name pilot60 --figs
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DV = "conf_on_fake"  # fixed dependent variable
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def zscore(df, cols):
    """z-score WITHIN the passed subset (so coeffs are comparable)."""
    out = df.copy()
    for c in cols:
        sd = out[c].std(ddof=0)
        out[f"z{c}"] = (out[c] - out[c].mean()) / sd if sd > 0 else 0.0
    return out


def per_model_regressions(agents, dv):
    print("\n" + "=" * 78)
    print(f" SEPARATE REGRESSIONS PER MODEL  -  DV = {dv}")
    if dv == "conf_on_fake":
        print(" POSITIVE coeff. = that trait makes the agent MORE gullible on fakes")
    print("=" * 78)

    # summary table: one row per (model x trait)
    summary = []
    for model, sub in agents.groupby("model"):
        sub = sub.dropna(subset=[dv]).copy()
        if len(sub) < 15 or sub[dv].std(ddof=0) == 0:
            print(f"\n### {model}: insufficient data or zero variance, skip")
            continue
        sub = zscore(sub, TRAITS + ["qualification_ord", "age"])
        rhs = " + ".join(f"z{t}" for t in TRAITS) + " + zqualification_ord + zage"
        # no cluster here: within a single model each persona appears once
        fit = smf.ols(f"{dv} ~ {rhs}", data=sub).fit()

        print(f"\n### {model}   (n={len(sub)}, mean {dv}={sub[dv].mean():.1f})")
        print(f"{'trait':<20}{'coef':>9}{'p':>9}   sig")
        for t in TRAITS:
            c = fit.params[f"z{t}"]
            p = fit.pvalues[f"z{t}"]
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            print(f"{t:<20}{c:>9.3f}{p:>9.3f}   {star}")
            summary.append({"model": model, "trait": t, "coef": c, "p": p})
    return pd.DataFrame(summary)


def interaction_test(agents, dv, name):
    """
    Formal test: do traits interact with the model? Compares the additive-only
    model (traits + model) against the one with interactions (traits*model).
    The verdict uses a Wald test with cluster-robust SE by persona: each persona
    appears once per model, so its 5 rows are not independent and the plain
    F-test (kept for reference) is too optimistic.
    """
    d = agents.dropna(subset=[dv]).copy()
    d = zscore(d, TRAITS + ["qualification_ord", "age"])
    ztraits = [f"z{t}" for t in TRAITS]
    f_add = f"{dv} ~ {' + '.join(ztraits)} + C(model) + zqualification_ord + zage"
    f_int = f"{dv} ~ ({' + '.join(ztraits)})*C(model) + zqualification_ord + zage"

    additive = smf.ols(f_add, data=d).fit()
    inter = smf.ols(f_int, data=d).fit()
    ftest = inter.compare_f_test(additive)

    inter_c = smf.ols(f_int, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["persona_id"]})
    names = list(inter_c.params.index)
    inter_terms = [n for n in names if ":" in n]
    R = np.zeros((len(inter_terms), len(names)))
    for i, n in enumerate(inter_terms):
        R[i, names.index(n)] = 1
    w = inter_c.wald_test(R, use_f=True, scalar=True)

    print("\n" + "=" * 78)
    print(f" INTERACTION TEST trait x model  -  DV = {dv}")
    print("=" * 78)
    print(f"R^2 additive model       : {additive.rsquared:.3f}")
    print(f"R^2 model w/ interactions: {inter.rsquared:.3f}")
    print(f"\nPlain F-test (reference): F={ftest[0]:.2f}, p={ftest[1]:.4g}")
    print(f"Wald test, cluster-robust by persona: F({int(w.df_num)}, {int(w.df_denom)})"
          f" = {float(w.statistic):.2f}, p={float(w.pvalue):.4g}")
    if float(w.pvalue) < 0.05:
        print(">> The effect of traits DEPENDS on the model: NOT generalisable to 'LLMs'.")
        print("   Report per-model, not aggregated.")
    else:
        print(">> No significant interaction: the effect of traits is consistent")
        print("   across models, the aggregate analysis is adequate.")

    res = pd.DataFrame([{
        "dv": dv, "n": len(d),
        "r2_additive": additive.rsquared, "r2_interaction": inter.rsquared,
        "plain_F": ftest[0], "plain_p": ftest[1], "plain_df": f"{int(ftest[2])},{int(inter.df_resid)}",
        "wald_F": float(w.statistic), "wald_p": float(w.pvalue),
        "wald_df": f"{int(w.df_num)},{int(w.df_denom)}",
    }])
    out = f"output/{name}-interaction_{dv}.csv"
    res.to_csv(out, index=False)
    print(f"[SUCCESS] interaction test saved to {out}")
    return res


def plot_human_vs_llm(agents, name, dv="conf_on_fake"):
    """
    The core non-replication figure: for each trait it compares the DIRECTION of
    the effect found in humans (Wolverton & Stevens: all traits favour the LOW
    pole) with that of the LLMs (median of per-model coefs on conf_on_fake).

    conf_on_fake convention: NEGATIVE coef = high trait makes LESS gullible =
    HIGH pole detects better. POSITIVE coef = LOW pole detects better (like humans).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # median + range of per-model coefs, for each trait
    per_model = {t: [] for t in TRAITS}
    for m, sub in agents.groupby("model"):
        sub = sub.dropna(subset=[dv])
        if len(sub) < 15 or sub[dv].std(ddof=0) == 0:
            continue
        sub = zscore(sub, TRAITS + ["qualification_ord", "age"])
        rhs = " + ".join(f"z{t}" for t in TRAITS) + " + zqualification_ord + zage"
        fit = smf.ols(f"{dv} ~ {rhs}", data=sub).fit()
        for t in TRAITS:
            per_model[t].append(fit.params[f"z{t}"])

    #med = {t: np.median(per_model[t]) for t in TRAITS}

    # Verdict per trait. Need to tell apart "inverts", "replicates", "disagree".
    # We do NOT use the raw count of models: it can mislead when few models have
    # huge effects and many are ~0 (e.g. Agreeableness, where two very strong
    # negative models outweigh three near-zero ones).
    # We use: (a) the magnitude-weighted median for the dominant sign,
    #         (b) "disagree" if strong effects coexist on both sides.
    def _verdict_pos(t):
        vals = np.array(per_model[t])
        strong = vals[np.abs(vals) >= 0.5]        # non-trivial effects
        if len(strong) == 0:
            return 0.0                             # no relevant effect
        pos_strong = strong[strong > 0]
        neg_strong = strong[strong < 0]
        # disagree: STRONG effects on both sides with comparable magnitude
        if len(pos_strong) and len(neg_strong):
            if min(abs(pos_strong).max(), abs(neg_strong).max()) >= 1.0:
                return 0.0                         # strongly split -> disagree
        # dominant sign = the one with the largest magnitude
        dom = strong[np.argmax(np.abs(strong))]
        # coef on conf_on_fake: NEG = HIGH pole detects better (left side)
        return -1.0 if dom < 0 else 1.0

    def llm_pos(t):
        return _verdict_pos(t)

    def is_disagree(t):
        vals = np.array(per_model[t])
        return _verdict_pos(t) == 0.0 and np.any(np.abs(vals) >= 0.5)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(TRAITS))[::-1]

    for yi, t in zip(y, TRAITS):
        hx = 1.0
        lx = llm_pos(t)
        # humans on the right (+1 = LOW pole detects better).
        # LLM on the right (+1) = same side as humans = replicates; on the left (-1) = inverts;
        # at the centre (0) = no effect or models disagree.
        if lx == 1.0:
            lcol, verdict = "#2E8B57", "replicates humans"
        elif lx == -1.0:
            lcol, verdict = "#C0392B", "inverts"
        else:
            lcol, verdict = "#8A8A8A", ("models disagree" if is_disagree(t) else "no effect")

        # faint connecting line between the two points
        ax.plot([lx, hx], [yi, yi], color="#CFCEC8", lw=1, zorder=1)
        # human point (green, always right)
        ax.plot(hx, yi, "o", ms=13, color="#008300", zorder=3, mec="white", mew=1)
        # LLM point
        ax.plot(lx, yi, "o", ms=13, color=lcol, zorder=3, mec="white", mew=1)
        # verdict label next to the LLM point
        ax.text(lx, yi + 0.32, verdict, fontsize=8.5, color=lcol,
                ha="center", va="bottom", style="italic")

    ax.axvline(0, color="#D0CFC9", lw=1, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(TRAITS, fontsize=11)
    ax.set_ylim(-0.6, len(TRAITS) - 0.4)
    ax.set_xlim(-1.6, 1.6)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["HIGH pole\ndetects better", "no effect /\ninconsistent",
                        "LOW pole\ndetects better"], fontsize=9)
    ax.set_title("Direction of each trait's effect on fake-news detection:\nhumans vs LLMs",
                 fontsize=13, pad=12)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#008300",
               markersize=11, label="humans (Wolverton & Stevens 2019)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8A8A8A",
               markersize=11, label="LLMs (median of 5 models)"),
    ], fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
        frameon=False)
    ax.grid(axis="x", ls="--", alpha=0.25)
    fig.tight_layout()
    path = f"output/{name}-human_vs_llm.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[SUCCESS] human-vs-LLM comparison saved to {path}")


def plot_ceiling(agents, name):
    """
    Histogram of W&S's binary DV (n_fake_correct 0-5): shows the saturation. If
    the vast majority score 5/5, the original task no longer discriminates ->
    justifies the switch to confidence.
    """
    import matplotlib.pyplot as plt

    counts = agents["n_fake_correct"].value_counts().reindex(range(6), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#378ADD", width=0.7)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("number of fakes (out of 5) correctly identified")
    ax.set_ylabel("number of agents")
    ax.set_title("Distribution of the binary DV (ceiling effect)")
    ax.set_xticks(range(6))
    ax.grid(axis="y", ls="--", alpha=0.3)
    n_max = int(counts.get(5, 0)); tot = int(counts.sum())
    ax.text(0.02, 0.95, f"{n_max}/{tot} agents at ceiling (5/5)",
            transform=ax.transAxes, fontsize=10, va="top", color="#185FA5")
    fig.tight_layout()
    path = f"output/{name}-ceiling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[SUCCESS] ceiling histogram saved to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot60")
    ap.add_argument("--figs", action="store_true",
                    help="generate thesis figures (human-vs-llm + ceiling)")
    ap.add_argument("--pooled", action="store_true",
                    help="also re-run main.py's pooled analysis (replication table + regressions)")
    args = ap.parse_args()

    path = f"output/{args.name}-agents.csv"
    # sep=None + engine="python" auto-detects the separator (comma or ';').
    # On Italian Windows/Excel CSVs often use ';': this works in both cases.
    agents = pd.read_csv(path, sep=None, engine="python")
    print(f"Read {path}: {len(agents)} agents, {agents['model'].nunique()} models")

    tbl = per_model_regressions(agents, DV)
    interaction_test(agents, DV, args.name)
    if args.figs:
        plot_human_vs_llm(agents, args.name, DV)
        plot_ceiling(agents, args.name)

    # save the summary coef table per (model x trait)
    if not tbl.empty:
        pivot = tbl.pivot(index="trait", columns="model", values="coef").round(3)
        print("\n" + "=" * 78)
        print(" coef MATRIX per (trait x model)  -  quick look at consistency")
        print("=" * 78)
        print(pivot.to_string())
        out = f"output/{args.name}-per_model_{DV}.csv"
        pivot.to_csv(out)
        print(f"\n[SUCCESS] matrix saved to {out}")

    # pooled analysis of main.py (all models together, C(model), cluster SE),
    # recomputed from the agents CSV: printed only, as in main.py
    if args.pooled:
        from main import replication_table, fit_regression
        rep = replication_table(agents.copy())
        rep.to_csv(f"output/{args.name}-replication_table.csv", index=False)
        fit_regression(agents.copy(), args.name)