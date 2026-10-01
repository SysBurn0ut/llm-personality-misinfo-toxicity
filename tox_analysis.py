# -*- coding: utf-8 -*-
"""
TOXICITY SIMULATION - aggregate analysis over all judged runs.

Reads every tox_<config>-<seed>-<condition>-judged.csv and produces:
  1. trajectory   share of toxic comments in 10 equal intervals of each thread,
                  Mann-Kendall test + linear slope per thread (as Avalle et al.),
                  figure per configuration (mean over seeds) + one per thread
  2. scorers      Detoxify vs LLM judge vs RoBERTa: Spearman, agreement at
                  the threshold, McNemar exact test on the Detoxify-judge
                  disagreements; the disagreeing comments are exported to
                  disagreements.csv for manual inspection
  3. traits       grid: agent toxicity ~ 5 traits (high/low) + model + seed FE,
                  cluster SE on the profile, pooled and per model;
                  single: agents with the trait at 100 vs 0, others at 50
  +  summary      share of toxic comments by configuration x condition x model

No model calls. Toxicity of a comment: score >= threshold (default 0.5 for all
scorers; --thr-detox 0.6 repeats everything with the threshold of Avalle et al.). Toxicity of an agent or
thread: share of its toxic comments (Avalle et al. definition).

Usage:
    python tox_analysis.py --glob "output/<folder>/tox_*-judged.csv" --out output/tox_analysis
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
CONFIGS = ["cont", "grid", "single"]
COND = {"induced": "indotta", "noinduced": "naturale"}
COND_EN = {"induced": "induced", "noinduced": "natural"}  # figure labels
SCORERS = {"toxicity": "Detoxify", "toxicity_judge": "giudice", "toxicity_hf": "RoBERTa"}
THR_DETOX = 0.5   # Detoxify and RoBERTa (lexical classifiers); --thr-detox
THR_JUDGE = 0.5   # LLM judge (0-1 after rescaling); --thr-judge


def thr(col):
    """Threshold for a score column: the judge has its own, the classifiers share one."""
    return THR_JUDGE if col == "toxicity_judge" else THR_DETOX
N_BINS = 10


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"tox_(cont|grid|single)-(\d+)-(induced|noinduced)-judged\.csv$",
                      path.replace("\\", "/"))
        if not m:
            continue
        d = pd.read_csv(path)
        d["config"], d["seed"], d["condition"] = m.group(1), int(m.group(2)), m.group(3)
        d["model"] = d["model"].str.split("/").str[-1]
        rows.append(d)
    if not rows:
        raise SystemExit(f"no file matches {pattern}")
    d = pd.concat(rows, ignore_index=True)
    ok = ~d["refused"].astype(bool) & d["toxicity"].notna()
    if "api_fail" in d:
        ok &= ~d["api_fail"].astype(bool)
    print(f"read {d[['config', 'seed', 'condition']].drop_duplicates().shape[0]} runs, "
          f"{len(d)} rows, {int((~ok).sum())} excluded (refused / failed)")
    return d[ok].copy()


# ------------------------------------------------------------------ 1. trajectory
def trajectories(d):
    d = d.sort_values("turn").copy()
    d["pos"] = d.groupby(["config", "seed", "condition"]).cumcount()
    d["n"] = d.groupby(["config", "seed", "condition"])["pos"].transform("size")
    d["bin"] = np.minimum((d["pos"] / d["n"] * N_BINS).astype(int), N_BINS - 1)
    curves, tests = [], []
    for (cfg, seed, cond), g in d.groupby(["config", "seed", "condition"]):
        for col, lab in [("toxicity", "Detoxify"), ("toxicity_judge", "giudice")]:
            s = (g.dropna(subset=[col]).groupby("bin")[col]
                 .apply(lambda v, c=col: (v >= thr(c)).mean()).reindex(range(N_BINS)))
            for b, v in s.items():
                curves.append({"config": cfg, "seed": seed, "condition": cond,
                               "scorer": lab, "bin": b, "share_toxic": v})
            y = s.dropna()
            if len(y) > 2:
                lr = stats.linregress((y.index + 0.5) / N_BINS, y.values)
                res = mk.original_test(y.values)
                mk_s, mk_p = res.s, res.p
                tests.append({"config": cfg, "seed": seed, "condition": cond,
                              "scorer": lab, "mean_share": y.mean(),
                              "slope": lr.slope, "slope_p": lr.pvalue,
                              "mk_S": mk_s, "mk_p": mk_p})
    return pd.DataFrame(curves), pd.DataFrame(tests)


def plot_trajectories(curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    style = {"induced": ("#C25B3A", "-"), "noinduced": ("#2E6B8A", "-")}
    c = curves[curves.scorer == "Detoxify"]
    cfgs = [k for k in CONFIGS if k in set(c.config)]

    # main figure: mean over seeds, band = min-max over seeds
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.2 * len(cfgs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    ymax = max(0.05, c.share_toxic.max() * 1.15)
    for ax, cfg in zip(axes, cfgs):
        for cond, (col, ls) in style.items():
            g = c[(c.config == cfg) & (c.condition == cond)]
            if g.empty:
                continue
            p = g.pivot_table(index="bin", columns="seed", values="share_toxic").reindex(range(N_BINS))
            ax.fill_between(x, p.min(axis=1), p.max(axis=1), color=col, alpha=.15, lw=0)
            ax.plot(x, p.mean(axis=1), ls, color=col, lw=2, marker="o", ms=3.5,
                    label=COND_EN[cond])
        ax.set_title({"cont": "continuous"}.get(cfg, cfg), fontsize=11)
        ax.set_xlabel("normalised position in the thread")
        ax.set_ylim(0, ymax); ax.set_xlim(0, 1)
        ax.grid(alpha=.3, ls="--")
    axes[0].set_ylabel("share of toxic comments")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/trajectory.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # appendix: one small panel per thread, Detoxify vs judge
    runs = curves[["config", "seed", "condition"]].drop_duplicates().sort_values(
        ["config", "condition", "seed"]).values.tolist()
    ncol = 6
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    ymax = max(0.05, curves.share_toxic.max() * 1.1)
    for ax, (cfg, seed, cond) in zip(axes, runs):
        for lab, col, lab_en in [("Detoxify", "#2E6B8A", "Detoxify"),
                                 ("giudice", "#C25B3A", "LLM judge")]:
            g = curves[(curves.config == cfg) & (curves.seed == seed) &
                       (curves.condition == cond) & (curves.scorer == lab)]
            ax.plot(x, g.sort_values("bin").share_toxic, color=col, lw=1.4, label=lab_en)
        ax.set_title(f"{ {'cont': 'continuous'}.get(cfg, cfg)} · {COND_EN[cond]} · {seed}", fontsize=8)
        ax.set_ylim(0, ymax); ax.grid(alpha=.3, ls="--")
    for ax in axes[len(runs):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.supxlabel("normalised position in the thread", fontsize=9)
    fig.supylabel("share of toxic comments", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/trajectory_per_run.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 2. scorers
def scorers(d):
    rows = []
    groups = [("all", "all", d)] + [(c, k, g) for (c, k), g in d.groupby(["config", "condition"])]
    for cfg, cond, g in groups:
        g = g.dropna(subset=["toxicity", "toxicity_judge"])
        dx, jd = g.toxicity >= THR_DETOX, g.toxicity_judge >= THR_JUDGE
        judge_only, detox_only = int((jd & ~dx).sum()), int((dx & ~jd).sum())
        disc = judge_only + detox_only
        p = (mcnemar([[0, judge_only], [detox_only, 0]], exact=True).pvalue
             if disc else np.nan)
        row = {"config": cfg, "condition": cond, "n": len(g),
               "rho_detox_judge": stats.spearmanr(g.toxicity, g.toxicity_judge)[0],
               "agree_at_thr": (dx == jd).mean(),
               "toxic_detox": dx.mean(), "toxic_judge": jd.mean(),
               "judge_only": judge_only, "detox_only": detox_only, "mcnemar_p": p}
        h = g.dropna(subset=["toxicity_hf"])
        if len(h) > 2:
            hf = h.toxicity_hf >= THR_DETOX
            row.update({
                "toxic_hf": hf.mean(),
                "rho_detox_hf": stats.spearmanr(h.toxicity, h.toxicity_hf)[0],
                "rho_hf_judge": stats.spearmanr(h.toxicity_hf, h.toxicity_judge)[0],
                "agree_detox_hf": ((h.toxicity >= THR_DETOX) == hf).mean(),
                "agree_hf_judge": (hf == (h.toxicity_judge >= THR_JUDGE)).mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def disagreements(d):
    """Comments that Detoxify and the judge classify in opposite ways, for
    manual inspection and examples. Sorted by the size of the gap."""
    g = d.dropna(subset=["toxicity", "toxicity_judge"]).copy()
    dx, jd = g.toxicity >= THR_DETOX, g.toxicity_judge >= THR_JUDGE
    g = g[dx != jd].copy()
    g["type"] = np.where(g.toxicity_judge >= THR_JUDGE, "judge_only", "detox_only")
    g["gap"] = (g.toxicity_judge - g.toxicity).abs()
    cols = ["type", "config", "seed", "condition", "model", "turn", "toxicity",
            "toxicity_hf", "toxicity_judge", "gap", "comment", "judge_reason"]
    return g.sort_values(["type", "gap"], ascending=[False, False])[cols]


# ------------------------------------------------------------------ 3. traits
def agent_table(d, cfg):
    g = d[d.config == cfg]
    a = (g.groupby(["seed", "condition", "persona_id"])
         .agg(model=("model", "first"), slot=("persona_slot", "first"),
              tox=("toxicity", lambda v: (v >= THR_DETOX).mean()),
              tox_judge=("toxicity_judge", lambda v: (v.dropna() >= THR_JUDGE).mean()),
              **{t: (t, "first") for t in TRAITS})
         .reset_index())
    a["profile"] = a["seed"].astype(str) + "_" + a["slot"].astype(str)
    return a


def traits_grid(d):
    a = agent_table(d, "grid")
    if a.empty:
        return pd.DataFrame()
    for t in TRAITS:
        a[f"hi_{t}"] = (a[t] > 50).astype(int)
    rhs = " + ".join(f"hi_{t}" for t in TRAITS)
    rows = []
    for cond, g in a.groupby("condition"):
        for dv in ("tox", "tox_judge"):
            fits = [("all", g, f"{dv} ~ {rhs} + C(model) + C(seed)")]
            fits += [(m, gm, f"{dv} ~ {rhs} + C(seed)") for m, gm in g.groupby("model")]
            for model, gg, f in fits:
                gg = gg.dropna(subset=[dv])
                if gg[dv].std() == 0:
                    for t in TRAITS:
                        rows.append({"condition": cond, "scorer": SCORERS[{"tox": "toxicity", "tox_judge": "toxicity_judge"}[dv]],
                                     "model": model, "trait": t, "coef": 0.0, "p": np.nan, "n": len(gg)})
                    continue
                fit = smf.ols(f, data=gg).fit(cov_type="cluster",
                                              cov_kwds={"groups": pd.factorize(gg["profile"])[0]})
                for t in TRAITS:
                    rows.append({"condition": cond,
                                 "scorer": SCORERS[{"tox": "toxicity", "tox_judge": "toxicity_judge"}[dv]],
                                 "model": model, "trait": t, "coef": fit.params[f"hi_{t}"],
                                 "p": fit.pvalues[f"hi_{t}"], "n": len(gg)})
    return pd.DataFrame(rows)


def traits_single(d):
    a = agent_table(d, "single")
    if a.empty:
        return pd.DataFrame()
    rows = []
    for cond, g in a.groupby("condition"):
        for t in TRAITS:
            hi, lo = g[g[t] == 100], g[g[t] == 0]
            for dv, lab in [("tox", "Detoxify"), ("tox_judge", "giudice")]:
                h, l = hi[dv].dropna(), lo[dv].dropna()
                p = (stats.mannwhitneyu(h, l).pvalue
                     if len(h) and len(l) and (h.std() > 0 or l.std() > 0 or h.mean() != l.mean())
                     else np.nan)
                rows.append({"condition": cond, "scorer": lab, "trait": t,
                             "share_high": h.mean(), "share_low": l.mean(),
                             "diff": h.mean() - l.mean(), "mw_p": p,
                             "n_high": len(h), "n_low": len(l)})
    return pd.DataFrame(rows)


def by_model(d):
    return (d.groupby(["config", "condition", "model"])
            .agg(n=("toxicity", "size"),
                 toxic_detox=("toxicity", lambda v: (v >= THR_DETOX).mean()),
                 toxic_judge=("toxicity_judge", lambda v: (v.dropna() >= THR_JUDGE).mean()),
                 mean_detox=("toxicity", "mean"),
                 mean_judge=("toxicity_judge", "mean"))
            .reset_index())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="output/tox_*-judged.csv")
    ap.add_argument("--out", default="output/tox_analysis")
    ap.add_argument("--thr-detox", type=float, default=0.5,
                    help="toxicity threshold for Detoxify and RoBERTa (default 0.5; "
                         "0.6 = Avalle et al., robustness check)")
    ap.add_argument("--thr-judge", type=float, default=0.5,
                    help="toxicity threshold for the LLM judge (default 0.5)")
    a = ap.parse_args()
    THR_DETOX, THR_JUDGE = a.thr_detox, a.thr_judge
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 160)

    d = load(a.glob)

    print("\n=== SHARE OF TOXIC COMMENTS BY MODEL (thresholds: Detoxify/RoBERTa %.2f, judge %.2f) ===" % (THR_DETOX, THR_JUDGE))
    bm = by_model(d)
    print(bm.round(3).to_string(index=False))
    bm.to_csv(f"{a.out}/by_model.csv", index=False)

    print("\n=== 1. TRAJECTORY (Detoxify) - per thread ===")
    curves, tests = trajectories(d)
    print(tests[tests.scorer == "Detoxify"].round(3).to_string(index=False))
    curves.to_csv(f"{a.out}/trajectory_curves.csv", index=False)
    tests.to_csv(f"{a.out}/trajectory_tests.csv", index=False)
    plot_trajectories(curves, a.out)

    print("\n=== 2. DETOXIFY vs JUDGE ===")
    sc = scorers(d)
    print(sc.round(3).to_string(index=False))
    sc.to_csv(f"{a.out}/scorers.csv", index=False)
    dis = disagreements(d)
    dis.to_csv(f"{a.out}/disagreements.csv", index=False)
    print(f"\n{len(dis)} comments classified in opposite ways "
          f"({(dis.type == 'judge_only').sum()} toxic only for the judge, "
          f"{(dis.type == 'detox_only').sum()} only for Detoxify) -> disagreements.csv")
    for typ in ("judge_only", "detox_only"):
        top = dis[dis.type == typ].head(3)
        for _, r in top.iterrows():
            print(f"  [{typ}] detox={r.toxicity:.2f} hf={r.toxicity_hf:.2f} "
                  f"judge={r.toxicity_judge:.2f} | {str(r.comment)[:100]}")

    print("\n=== 3a. TRAITS - grid (coef = change in share of toxic comments, low -> high) ===")
    tg = traits_grid(d)
    if not tg.empty:
        print(tg[tg.model == "all"].round(3).to_string(index=False))
        tg.to_csv(f"{a.out}/traits_grid.csv", index=False)

    print("\n=== 3b. TRAITS - single (trait at 100 vs 0, others at 50) ===")
    ts = traits_single(d)
    if not ts.empty:
        print(ts.round(3).to_string(index=False))
        ts.to_csv(f"{a.out}/traits_single.csv", index=False)

    print(f"\n[SUCCESS] tables and figures in {a.out}/")
