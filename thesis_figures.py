"""
Thesis figures (study 1): 25-panel grid, heatmap, replication grid.
Standalone: reads only output/<name>-agents.csv.
    python thesis_figures.py --name pilot100
    python thesis_figures.py --name pilot100 --singles   # also the 25 panels as single files + zip
    python thesis_figures.py --agents output/runA/pilot100-agents.csv output/runB/x-agents.csv --singles
"""
import argparse
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import FancyArrow
import statsmodels.formula.api as smf

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
MODELS = ["deepseek-v4-pro", "gemini-3.1-flash-lite", "llama-4-maverick", "gpt-4o-mini", "qwen3.7-plus"]
COLORS = {"deepseek-v4-pro": "#2a78d6", "gemini-3.1-flash-lite": "#008300",
          "llama-4-maverick": "#e87ba4", "gpt-4o-mini": "#eda100", "qwen3.7-plus": "#4a3aa7"}
# W&S 2019: higher trait -> believes fakes MORE
HUMAN_SIGN = {t: +1 for t in TRAITS}
GREEN = "#2e8b3a"
BINS = [0, 20, 40, 60, 80, 100.1]
MIDS = [10, 30, 50, 70, 90]
RANGES = ["0–20", "20–40", "40–60", "60–80", "80–100"]


# ---------- helpers ----------
def z(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def short(m):
    return m.split("/")[-1]


def short2(m):
    return m.replace("-", "\n", 1)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def save(fig, path):
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def fit_all(agents, dv):
    """Per-model OLS on credulity (dv) with z-scored traits + controls.
    beta = pp per +1 SD trait; beta_std = SD units (dv z-scored too)."""
    covs = [c for c in ["qualification_ord", "age"] if c in agents.columns]
    rows = []
    for model, sub in agents.groupby("model"):
        sub = sub.dropna(subset=[dv]).copy()
        if len(sub) < 15 or sub[dv].std(ddof=0) == 0:
            continue
        for c in TRAITS + covs:
            sub[f"z{c}"] = z(sub[c])
        sub["zdv"] = z(sub[dv])
        rhs = " + ".join(f"z{c}" for c in TRAITS + covs)
        raw = smf.ols(f"{dv} ~ {rhs}", data=sub).fit()
        std = smf.ols(f"zdv ~ {rhs}", data=sub).fit()
        for t in TRAITS:
            k = f"z{t}"
            rows.append({"model": short(model), "trait": t, "beta": raw.params[k],
                         "beta_std": std.params[k], "p": raw.pvalues[k]})
    return pd.DataFrame(rows)


def human_arrow(ax):
    """Direction-only arrow in axes coordinates."""
    ax.add_patch(FancyArrow(.06, .12, .82, .72, width=.07, head_width=.17, head_length=.12,
                            length_includes_head=True, color=GREEN, alpha=.13, lw=0, transform=ax.transAxes))


# ---------- 25-panel grid ----------
def grid_data(a, dv):
    models = [m for m in MODELS if m in a.model.unique()]
    bands, bases = {}, {}
    for mod in models:
        s = a[a.model == mod]
        bases[mod] = s[dv].mean()
        bands[mod] = [s.groupby(pd.cut(s[t], BINS, right=False), observed=False)[dv].mean().values
                      for t in TRAITS]
    span = max(max(np.nanmax(np.abs(np.array(bands[m]) - bases[m])), 1) for m in models) * 1.3
    return models, bands, bases, span


def draw_cell(ax, res, a, mod, t, j, bands, bases, span, yscale):
    """yscale='centered': y = pp from the model's own mean, identical axis for all models.
    yscale='absolute': y = % belief, identical 0..max axis for all models."""
    mu, sd = a[t].mean(), a[t].std(ddof=0)
    x = np.linspace(0, 100, 50)
    off = 0 if yscale == "absolute" else bases[mod]
    human_arrow(ax)
    r = res[(res.trait == t) & (res.model == mod)].iloc[0]
    sig = r.p < .05
    ax.plot(x, bases[mod] - off + r.beta * (x - mu) / sd, color=COLORS[mod], lw=2.6 if sig else 1.2,
            ls="-" if sig else "--")
    ax.plot(MIDS, np.array(bands[mod][j]) - off, "o", color=COLORS[mod], ms=4, alpha=.7)
    if yscale == "absolute":
        top = max(bases[m] + span for m in bases)
        ax.set_ylim(0, top)
    else:
        ax.set_ylim(-span, span)
        ax.axhline(0, color="gray", lw=.7)
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=8); ax.grid(axis="y", ls="--", alpha=.3)
    ax.set_xticks(MIDS)


def ylab(yscale):
    return "belief in fake news (%)" if yscale == "absolute" else "belief vs model mean (pp)"


def grid(res, a, dv, out, yscale):
    models, bands, bases, span = grid_data(a, dv)
    fig, axes = plt.subplots(len(models), 5, figsize=(16, 2.4 * len(models)), sharex=True, sharey=True)
    for i, mod in enumerate(models):
        for j, t in enumerate(TRAITS):
            ax = axes[i, j]
            draw_cell(ax, res, a, mod, t, j, bands, bases, span, yscale)
            if i == 0:
                ax.set_title(t, fontsize=12)
            if j == 0:
                lab = short2(mod) if yscale == "absolute" else f"{short2(mod)}\n(mean {bases[mod]:.0f}%)"
                ax.set_ylabel(lab, fontsize=9)
            if i == len(models) - 1:
                ax.set_xticks(MIDS, RANGES, fontsize=7)
    fig.suptitle("Belief in fake news by trait: LLMs vs human direction", fontsize=14, y=1.0)
    fig.text(.5, -.005, f"y = {ylab(yscale)}  ·  green arrow = human direction (Wolverton & Stevens, 2019)  ·  "
             "dots = observed means  ·  thick line = p<.05", ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(); save(fig, out)


def grid_singles(res, a, dv, folder, yscale):
    os.makedirs(folder, exist_ok=True)
    models, bands, bases, span = grid_data(a, dv)
    for mod in models:
        for j, t in enumerate(TRAITS):
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            draw_cell(ax, res, a, mod, t, j, bands, bases, span, yscale)
            ax.set_xticks(MIDS, RANGES, fontsize=8)
            ax.set_xlabel(t, fontsize=10)
            ax.set_ylabel(ylab(yscale), fontsize=9)
            ttl = f"{mod} — {t}" if yscale == "absolute" else f"{mod} — {t} (mean {bases[mod]:.0f}%)"
            ax.set_title(ttl, fontsize=10)
            fig.tight_layout()
            save(fig, os.path.join(folder, f"{mod}_{t}"))
    shutil.make_archive(folder, "zip", os.path.dirname(folder) or ".", os.path.basename(folder))


# ---------- heatmap ----------
def heatmap(res, out):
    models = [m for m in MODELS if m in res.model.unique()]
    M = res.pivot(index="trait", columns="model", values="beta_std").loc[TRAITS, models]
    P = res.pivot(index="trait", columns="model", values="p").loc[TRAITS, models]
    lim = np.nanmax(np.abs(M.values))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(M.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(TRAITS)):
        for j in range(len(models)):
            v = M.values[i, j]
            ax.text(j, i, f"{v:+.2f}{stars(P.values[i, j])}", ha="center", va="center", fontsize=11,
                    color="white" if abs(v) > .6 * lim else "black")
    hx = len(models) + .15
    for i, t in enumerate(TRAITS):
        ax.add_patch(plt.Rectangle((hx - .5, i - .5), 1, 1, color="#f4d6d6" if HUMAN_SIGN[t] > 0 else "#d6e4f4",
                                   clip_on=False))
        ax.text(hx, i, "+  ↑" if HUMAN_SIGN[t] > 0 else "−  ↓", ha="center", va="center", fontsize=13, clip_on=False)
    ax.set_xlim(-.5, hx + .5)
    ax.set_xticks(list(range(len(models))) + [hx], [short2(m) for m in models] + ["Humans\n(W&S 2019)"], fontsize=10)
    ax.set_yticks(range(len(TRAITS)), TRAITS, fontsize=11)
    ax.axvline(len(models) - .5 + .075, color="black", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels(["less\nbelief", "0", "more\nbelief"])
    cb.ax.tick_params(labelsize=9)
    ax.set_title("Effect of each trait on belief in fake news (standardized β)", fontsize=13, pad=12)
    fig.text(.5, -.02, "* p<.05  ** p<.01  *** p<.001  ·  human column: direction reported by Wolverton & Stevens (2019)",
             ha="center", fontsize=9, color="dimgray")
    save(fig, out)


# ---------- replication grid ----------
def replication(res, out):
    models = [m for m in MODELS if m in res.model.unique()]
    code = np.zeros((len(TRAITS), len(models)), dtype=int)  # 0 n.s., 1 same, 2 opposite
    for i, t in enumerate(TRAITS):
        for j, m in enumerate(models):
            r = res[(res.trait == t) & (res.model == m)].iloc[0]
            if r.p < .05:
                code[i, j] = 1 if np.sign(r.beta) == HUMAN_SIGN[t] else 2
    sym = {0: "n.s.", 1: "✓", 2: "✗"}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.imshow(code, cmap=ListedColormap(["#e6e6e6", "#8fd19e", "#f19c9c"]), vmin=0, vmax=2, aspect="auto")
    for i in range(len(TRAITS)):
        for j in range(len(models)):
            ax.text(j, i, sym[code[i, j]], ha="center", va="center", fontsize=15 if code[i, j] else 10)
    ax.set_xticks(range(len(models)), [short2(m) for m in models], fontsize=10)
    ax.set_yticks(range(len(TRAITS)), TRAITS, fontsize=11)
    ax.set_xticks(np.arange(-.5, len(models)), minor=True)
    ax.set_yticks(np.arange(-.5, len(TRAITS)), minor=True)
    ax.grid(which="minor", color="white", lw=3); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Replication of the human trend (Wolverton & Stevens, 2019)", fontsize=13, pad=12)
    fig.text(.5, -.02, "✓ same direction as humans (p<.05)   ✗ opposite direction (p<.05)   n.s. = not significant",
             ha="center", fontsize=9, color="dimgray")
    save(fig, out)


def run(path, dv, singles, yscale):
    """Figures are saved next to the agents CSV, prefixed with its run name."""
    agents = pd.read_csv(path)
    a = agents.assign(model=agents["model"].map(short))
    res = fit_all(agents, dv)
    base = os.path.basename(path)
    name = base[:-len("-agents.csv")] if base.endswith("-agents.csv") else os.path.splitext(base)[0]
    tag = "" if dv == "conf_on_fake" else f"_{dv}"
    pre = os.path.join(os.path.dirname(path), f"{name}-")
    scales = ["centered", "absolute"] if yscale == "both" else [yscale]
    for ys in scales:
        suf = f"_{ys}" if yscale == "both" else ""
        grid(res, a, dv, pre + "single_traits" + suf + tag, ys)
        if singles:
            grid_singles(res, a, dv, pre + "single_traits_singles" + suf + tag, ys)
    heatmap(res, pre + "heatmap_belief" + tag)
    replication(res, pre + "replication" + tag)
    print(f"[SUCCESS] {path} -> {os.path.dirname(path) or '.'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="run name, reads output/<name>-agents.csv")
    ap.add_argument("--agents", nargs="+", default=None,
                    help="one or more paths to agents CSVs (e.g. output/run1/pilot100-agents.csv ...)")
    ap.add_argument("--dv", default="conf_on_fake")
    ap.add_argument("--singles", action="store_true", help="also save the 25 panels as single files + zip")
    ap.add_argument("--yscale", choices=["centered", "absolute", "both"], default="centered",
                    help="centered: pp vs each model's mean (same axis for all); absolute: same 0-max %% axis for all; both: generate both")
    args = ap.parse_args()
    paths = args.agents or [f"output/{args.name or 'pilot100'}-agents.csv"]
    for p in paths:
        run(p, args.dv, args.singles, args.yscale)
