# -*- coding: utf-8 -*-
"""
BFI DIAGNOSTICS: are the high correlations personality or arithmetic?

The Big Five score is written verbatim in the system message ("Openness: 90/100").
A model can therefore hit r=0.95 without "being" that person: it just needs to
translate the number into Likert answers. These checks separate the two cases.

  1. WITHIN-SCALE VARIABILITY   within-scale item SD ~0 = arithmetic
  2. FLAT RESPONSES             % identical answer to all items of a scale
  3. SLOPE AND SATURATION       slope > 1 and pile-up at 0/100 = amplification
  4. ADDED VALUE                joint Wald test of BFI terms over imposed scores
                                (only with a behaviour run on the SAME personas)
  5. INTERNAL CONSISTENCY       Cronbach's alpha and mean inter-item r
  6. SCALE USE                  % answers at 1/5 and at 3
  + DISCRIMINANT VALIDITY       imposed x recovered 5x5 matrix

Usage:
    python bfi_diagnostics.py --name bfiS42 --behavior pilot100
    python bfi_diagnostics.py --name bfiS12 bfiS42 bfiS73 (pooled, no check 4)
"""
import argparse
import os

import numpy as np
import pandas as pd

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

SCORING = {
    "Openness":          [5, 10, 15, 20, 25, 30, "35R", 40, "41R", 44],
    "Conscientiousness": [3, "8R", 13, "18R", "23R", 28, 33, 38, "43R"],
    "Extraversion":      [1, "6R", 11, 16, "21R", 26, "31R", 36],
    "Agreeableness":     ["2R", 7, "12R", 17, 22, "27R", 32, "37R", 42],
    "Neuroticism":       [4, "9R", 14, 19, "24R", 29, "34R", 39],
}

# BFI-44 human alpha: pooled estimate and 95% prediction interval,
# reliability generalization meta-analysis (Husain et al., 2025, BMC Psychol 13:20)
HUMAN_ALPHA = {"Openness":          (0.77, 0.65, 0.90),
               "Conscientiousness": (0.80, 0.71, 0.90),
               "Extraversion":      (0.80, 0.69, 0.90),
               "Agreeableness":     (0.73, 0.61, 0.82),
               "Neuroticism":       (0.80, 0.70, 0.93)}


def recoded_items(df, trait):
    #Scale items with reverse-scoring already applied.
    cols = {}
    for k in SCORING[trait]:
        rev = isinstance(k, str) and k.endswith("R")
        n = int(str(k).rstrip("R"))
        c = f"item_{n}"
        if c in df.columns:
            cols[f"i{n}"] = 6 - df[c] if rev else df[c]
    return pd.DataFrame(cols)

def cronbach_alpha(items):
    """alpha = k/(k-1) * (1 - sum(var_j) / var(total)), complete cases only."""
    it = items.dropna()
    k = it.shape[1]
    if k < 2 or len(it) < 3:
        return np.nan
    var_tot = it.sum(axis=1).var(ddof=1)
    return np.nan if var_tot == 0 else k / (k - 1) * (1 - it.var(ddof=1).sum() / var_tot)


def mean_interitem_r(items):
    """Mean off-diagonal inter-item correlation, complete cases."""
    c = items.dropna().corr().values
    return np.nan if c.shape[0] < 2 else c[~np.eye(c.shape[0], dtype=bool)].mean()


def _pivot(rows, val):
    return pd.DataFrame(rows).pivot(index="trait", columns="model",
                                    values=val).reindex(TRAITS)


def diagnostics(raw, scores):
    rows = []
    for m in sorted(raw["model"].unique()):
        g, gs = raw[raw["model"] == m], scores[scores["model"] == m]
        for t in TRAITS:
            it = recoded_items(g, t)
            x, y = gs[f"imposed_{t}"], gs[f"bfi_{t}"]
            ok = x.notna() & y.notna()
            rows.append({
                "model": m.split("/")[-1], "trait": t,
                "r": np.corrcoef(x[ok], y[ok])[0, 1],
                "slope": np.polyfit(x[ok], y[ok], 1)[0],
                "sd_within": it.std(axis=1).mean(),
                "flat_%": (it.nunique(axis=1) == 1).mean() * 100,
                "sat_%": ((y[ok] == 0) | (y[ok] == 100)).mean() * 100,
                "alpha": cronbach_alpha(it),
                "r_ii": mean_interitem_r(it),
            })
    d = pd.DataFrame(rows)

    print("\n" + "=" * 74)
    print(" 1. MEAN ITEM SD WITHIN THE SAME SCALE (after reverse-scoring)")
    print("    low values = all items of the scale get the same answer")
    print("=" * 74)
    print(_pivot(rows, "sd_within").round(2).to_string())

    print("\n" + "=" * 74)
    print(" 2. % OF AGENTS WITH IDENTICAL ANSWER TO ALL ITEMS OF THE SCALE")
    print("=" * 74)
    print(_pivot(rows, "flat_%").round(1).to_string())

    print("\n" + "=" * 74)
    print(" 3. SLOPE (1 = faithful, >1 = amplifies) AND SATURATION AT 0/100")
    print("=" * 74)
    print("slope:"); print(_pivot(rows, "slope").round(2).to_string())
    print("\n% saturated scores:"); print(_pivot(rows, "sat_%").round(1).to_string())

    print("\n" + "=" * 74)
    print(" 5. INTERNAL CONSISTENCY (human BFI-44 pooled alpha: O .77, C .80, E .80, A .73, N .80)")
    print("=" * 74)
    print("Cronbach's alpha:"); print(_pivot(rows, "alpha").round(3).to_string())
    print("\nmean inter-item r:"); print(_pivot(rows, "r_ii").round(3).to_string())

    rr = np.corrcoef(d["sd_within"], d["r"])[0, 1]
    print(f"\nCorrelation between within-scale variability and recovery r, "
          f"over the {len(d)} cells: {rr:+.2f}")
    print("Negative = the models that answer more mechanically get the highest "
          "recovery:\nthe correlation rewards arithmetic, not embodiment of the "
          "profile.")
    return d


def discriminant(scores):
    print("\n" + "=" * 74)
    print(" DISCRIMINANT VALIDITY: imposed (rows) x recovered (cols)")
    print("=" * 74)
    s = scores.dropna(subset=[f"bfi_{t}" for t in TRAITS])
    M = pd.DataFrame(index=TRAITS, columns=TRAITS, dtype=float)
    for a in TRAITS:
        for b in TRAITS:
            M.loc[a, b] = np.corrcoef(s[f"imposed_{a}"], s[f"bfi_{b}"])[0, 1]
    print(M.round(2).to_string())
    off = M.values[~np.eye(5, dtype=bool)]
    print(f"\nmean diagonal = {np.mean(np.diag(M.values)):.3f} · "
          f"mean |off-diagonal| = {np.abs(off).mean():.3f}")
    print("Low off-diagonal = the 5 traits stay separated, LHS holds.")
    return M

def scale_use(raw):
    items = raw[[f"item_{i}" for i in range(1, 45)]]
    out = []
    for m, g in raw.groupby("model"):
        v = items.loc[g.index].values.ravel()
        v = v[~np.isnan(v)]
        out.append({"model": m.split("/")[-1],
                    "extreme_%": np.isin(v, [1, 5]).mean() * 100,
                    "mid_%": (v == 3).mean() * 100})
    u = pd.DataFrame(out)
    print("\n" + "=" * 74)
    print(" 6. SCALE USE: % answers at 1/5 and at 3 (raw, all 44 items)")
    print("=" * 74)
    print(u.round(1).to_string(index=False))
    return u

def added_value(scores, behavior_name, recal=True):
    # Does the BFI explain behaviour better than the imposed score?
    import statsmodels.formula.api as smf
    path = f"output/{behavior_name}-raw.csv"
    if not os.path.exists(path):
        print(f"\n[skip] {path} not found: skipping check 4")
        return None
    raw = pd.read_csv(path, sep=None, engine="python")
    if recal:
        from recalibration import recalibrate_confidence
        raw = recalibrate_confidence(raw)
    raw = raw[raw["ground_truth"].isin([False, "FALSE"])].dropna(subset=["confidence_true"])
    beh = (raw.groupby(["persona_id", "model"])["confidence_true"]
              .mean().rename("conf_on_fake").reset_index())
    d = beh.merge(scores.dropna(subset=[f"bfi_{t}" for t in TRAITS]),
                  on=["persona_id", "model"])

    # same persona_id must mean same persona: guard against seed mismatch
    ag_path = f"output/{behavior_name}-agents.csv"
    if not os.path.exists(ag_path):
        print(f"\n[skip] {ag_path} not found: cannot verify personas match, skipping check 4")
        return None
    ag = pd.read_csv(ag_path, sep=None, engine="python")[["persona_id", "model"] + TRAITS]
    chk = d.merge(ag, on=["persona_id", "model"])
    if len(chk) != len(d) or not all(np.allclose(chk[f"imposed_{t}"], chk[t]) for t in TRAITS):
        print(f"\n[skip] personas in {behavior_name} differ from the BFI ones "
              f"(different seed): skipping check 4")
        return None

    print("\n" + "=" * 74)
    print(f" 4. ADDED VALUE — conf_on_fake ~ traits + C(model), n={len(d)}")
    print("=" * 74)
    for pre in ("imposed", "bfi"):
        for t in TRAITS:
            d[f"z{pre}_{t}"] = ((d[f"{pre}_{t}"] - d[f"{pre}_{t}"].mean())
                                / d[f"{pre}_{t}"].std(ddof=0))
    res, fits = {}, {}
    for label, terms in [
        ("model only", []),
        ("imposed only", [f"zimposed_{t}" for t in TRAITS]),
        ("bfi only", [f"zbfi_{t}" for t in TRAITS]),
        ("both", [f"zimposed_{t}" for t in TRAITS] + [f"zbfi_{t}" for t in TRAITS]),
    ]:
        f = smf.ols(f"conf_on_fake ~ {' + '.join(terms + ['C(model)'])}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["persona_id"]})
        fits[label] = f
        res[label] = {"R2": f.rsquared, "R2_adj": f.rsquared_adj}
        print(f"  {label:<14} R2 = {f.rsquared:.4f}   R2_adj = {f.rsquared_adj:.4f}")

    # "both" nests "imposed only": R2 cannot decrease, so test the 5 BFI terms jointly
    w = fits["both"].wald_test(", ".join(f"zbfi_{t} = 0" for t in TRAITS),
                               use_f=True, scalar=True)
    res["wald_bfi"] = {"F": float(w.statistic), "p": float(w.pvalue),
                       "df_num": int(w.df_num), "df_denom": int(w.df_denom)}
    print(f"\n  Wald test H0: all BFI terms = 0 in 'both' (cluster-robust)"
          f"\n  F({int(w.df_num)}, {int(w.df_denom)}) = {float(w.statistic):.3f}"
          f"   p = {float(w.pvalue):.4g}")
    # robustness: BFI beyond a model-specific and quadratic use of the imposed score
    zi = [f"zimposed_{t}" for t in TRAITS]
    rhs = " + ".join([f"{z}*C(model)" for z in zi] + [f"I({z}**2)" for z in zi]
                     + [f"zbfi_{t}" for t in TRAITS])
    fr = smf.ols(f"conf_on_fake ~ {rhs}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["persona_id"]})
    w2 = fr.wald_test(", ".join(f"zbfi_{t} = 0" for t in TRAITS), use_f=True, scalar=True)
    res["wald_bfi_robust"] = {"F": float(w2.statistic), "p": float(w2.pvalue),
                              "df_num": int(w2.df_num), "df_denom": int(w2.df_denom)}
    print(f"\n  Robust Wald (imposed x model + imposed^2 controlled)"
          f"\n  F({int(w2.df_num)}, {int(w2.df_denom)}) = {float(w2.statistic):.3f}"
          f"   p = {float(w2.pvalue):.4g}")
    print("\nNB: R2 is dominated by the model fixed effects. 'both' nests 'imposed only',"
          "\nso its R2 is never lower: the evidence is the Wald test. p >= .05 = the BFI"
          "\nis a re-encoding of the imposed number, no added information.")
    return res


MODEL_COLORS = {"deepseek-v4-pro": "#2a78d6", "gemini-3.1-flash-lite": "#008300",
                "llama-4-maverick": "#e87ba4", "gpt-4o-mini": "#eda100",
                "qwen3.7-plus": "#4a3aa7"}
BINS = [0, 20, 40, 60, 80, 100.1]


def figures(d, raw, scores, M, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, stem):
        p = f"output/{name}-{stem}"
        fig.savefig(p + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(p + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[SUCCESS] {p}.png/.pdf")

    models = list(dict.fromkeys(d["model"]))
    col = lambda m: MODEL_COLORS.get(m, "#555555")
    x = np.arange(len(TRAITS))
    w = 0.8 / len(models)
    off = lambda k: (k - (len(models) - 1) / 2) * w
    short = raw["model"].str.split("/").str[-1]

    # A) flat responses + within-scale variability vs recovery
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for k, m in enumerate(models):
        s = d[d["model"] == m].set_index("trait").reindex(TRAITS)
        a1.bar(x + off(k), s["flat_%"], width=w, color=col(m), label=m)
        a2.scatter(s["sd_within"], s["r"], s=45, color=col(m), label=m)
    a1.set_xticks(x, TRAITS, fontsize=9, rotation=15)
    a1.set_ylabel("% of agents")
    a1.set_title("Identical answer to every item of the scale")
    a1.grid(axis="y", alpha=.3, ls="--"); a1.legend(fontsize=7, frameon=False)
    b = np.polyfit(d["sd_within"], d["r"], 1)
    xs = np.linspace(d["sd_within"].min(), d["sd_within"].max(), 10)
    a2.plot(xs, np.polyval(b, xs), color="#555555", ls="--", lw=1.4)
    rr = np.corrcoef(d["sd_within"], d["r"])[0, 1]
    a2.set_xlabel("within-scale item SD (low = mechanical)")
    a2.set_ylabel("imposed–recovered r")
    a2.set_title(f"Within-scale variability vs recovery (r = {rr:+.2f}, {len(d)} cells)")
    a2.grid(alpha=.3, ls="--")
    fig.tight_layout(); save(fig, "diag_flat_recovery")

    # B) Cronbach's alpha vs human benchmark
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, t in enumerate(TRAITS):
        est, lo, hi = HUMAN_ALPHA[t]
        ax.fill_between([i - .45, i + .45], lo, hi, color="#d9d9d9", alpha=.6, lw=0,
                        label="human 95% PI (BFI-44 meta-analysis)" if i == 0 else None)
        ax.hlines(est, i - .45, i + .45, color="#7a7a7a", lw=1.5,
                  label="human pooled α" if i == 0 else None)
    for k, m in enumerate(models):
        s = d[d["model"] == m].set_index("trait").reindex(TRAITS)
        ax.plot(x + off(k), s["alpha"], "o", ms=8, color=col(m), label=m)
    ax.set_xticks(x, TRAITS, fontsize=10, rotation=20, ha="right")
    ax.set_ylim(min(0.55, np.nanmin(d["alpha"]) - .05), 1.01)
    ax.set_ylabel("Cronbach's α")
    ax.set_title("Internal consistency by model and trait")
    ax.grid(axis="y", alpha=.3, ls="--")
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0))
    fig.tight_layout(); save(fig, "diag_alpha")

    # C) amplification: mean recovered score per imposed bin, traits pooled
    long = pd.concat([pd.DataFrame({"model": scores["model"].str.split("/").str[-1],
                                    "imp": scores[f"imposed_{t}"],
                                    "bfi": scores[f"bfi_{t}"]}) for t in TRAITS]).dropna()
    long["bin"] = pd.cut(long["imp"], BINS, right=False)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 100], [0, 100], ls=":", c="grey", lw=1, label="perfect fidelity")
    for m in models:
        g = long[long["model"] == m].groupby("bin", observed=False)[["imp", "bfi"]].mean()
        ax.plot(g["imp"], g["bfi"], "o-", color=col(m), lw=1.8, label=m)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("imposed score (bin mean)"); ax.set_ylabel("mean BFI-44 score")
    ax.set_title("Recovered vs imposed score, traits pooled")
    ax.grid(alpha=.3, ls="--"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); save(fig, "diag_amplification")

    # D) discriminant validity matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    V = M.values.astype(float)
    im = ax.imshow(V, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(V[i, j]) > .6 else "black")
    ax.set_xticks(range(5), TRAITS, rotation=25, ha="right")
    ax.set_yticks(range(5), TRAITS)
    ax.set_xlabel("recovered (BFI-44)"); ax.set_ylabel("imposed")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("Pearson r")
    ax.set_title("Imposed × recovered correlations")
    fig.tight_layout(); save(fig, "diag_discriminant")

    # E) raw answer distribution 1-5
    items = raw[[f"item_{i}" for i in range(1, 45)]]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for k, m in enumerate(models):
        v = items[short == m].values.ravel()
        v = v[~np.isnan(v)]
        ax.bar(np.arange(1, 6) + off(k), [np.mean(v == c) * 100 for c in range(1, 6)],
               width=w, color=col(m), label=m)
    ax.set_xticks(range(1, 6), ["1\ndisagree\nstrongly", "2", "3\nneutral", "4",
                                "5\nagree\nstrongly"])
    ax.set_ylabel("% of answers")
    ax.set_title("Distribution of raw item answers (all 44 items)")
    ax.grid(axis="y", alpha=.3, ls="--"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); save(fig, "diag_response_dist")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", nargs="+", default=["bfi"],
                    help="one or more BFI runs; several = pooled, check 4 skipped")
    ap.add_argument("--behavior", default="pilot100",
                    help="main run prefix, for check 4 (single --name only)")
    ap.add_argument("--no_recal", action="store_true",
                    help="do NOT recalibrate confidence_true in check 4 (original data)")
    args = ap.parse_args()

    load = lambda n, kind: pd.read_csv(f"output/{n}-{kind}.csv", sep=None, engine="python")
    raw = pd.concat([load(n, "raw") for n in args.name], ignore_index=True)
    scores = pd.concat([load(n, "scores") for n in args.name], ignore_index=True)
    tag = args.name[0] if len(args.name) == 1 else "bfi_pooled"

    d = diagnostics(raw, scores)
    scale_use(raw).to_csv(f"output/{tag}-scale_use.csv", index=False)
    M = discriminant(scores)
    if len(args.name) == 1:
        res = added_value(scores, args.behavior, recal=not args.no_recal)
        if res:
            w, w2 = res.pop("wald_bfi"), res.pop("wald_bfi_robust")
            av = pd.DataFrame(res).T.rename_axis("spec").reset_index()
            av["wald_F"], av["wald_p"] = w["F"], w["p"]
            av["wald_df"] = f"{w['df_num']},{w['df_denom']}"
            av["wald_robust_F"], av["wald_robust_p"] = w2["F"], w2["p"]
            av.to_csv(f"output/{tag}-added_value{'-orig' if args.no_recal else ''}.csv", index=False)
    else:
        print("\n[skip] check 4 is per seed: run it with a single --name")
    d.to_csv(f"output/{tag}-diagnostics.csv", index=False)
    figures(d, raw, scores, M, tag)
