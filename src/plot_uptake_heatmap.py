"""
Generates the traits' incorporation heatmap from the manipulation check's file.

How to:
    python plot_uptake_heatmap.py --name mcheck (or the name you set for the mcheck)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def normalise(df):
    """
    Reconcile the two raw formats.
      legacy:  no counterbalancing (A = high pole), `intensity` field
      current: `a_is_high` column + `confidence` field
    Output: `high_choice` = agent picked the high-pole option.

    """
    if "a_is_high" not in df.columns:
        df["a_is_high"] = True
    if "high_choice" not in df.columns:
        df["high_choice"] = np.where(df["choice"].isna(), None,
                                     (df["choice"] == "A") == df["a_is_high"])
    return df


def pct_high(s):
    #% high pole choices, unparsable were not counted
    s = s.dropna()
    return np.nan if len(s) == 0 else s.astype(bool).mean() * 100


def choice_gap(d):
    #High-minus-low gap, in % of high-pole choices
    hi = pct_high(d[d["group"] == "high"]["high_choice"])
    lo = pct_high(d[d["group"] == "low"]["high_choice"])
    if hi != hi or lo != lo:
        return np.nan
    return hi - lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mcheck")
    ap.add_argument("--weak", type=float, default=25,
                    help="threshold below which a cell is a 'gap' (highlighted in red)")
    args = ap.parse_args()

    # sep=None auto-detects ',' vs ';' (Italian Excel uses ';')
    df = normalise(pd.read_csv(f"output/mcheck/{args.name}-raw.csv", sep=None, engine="python"))

    models = sorted(df["model"].unique())
    mat = np.full((len(TRAITS), len(models)), np.nan)
    for i, t in enumerate(TRAITS):
        for j, m in enumerate(models):
            d = df[(df["trait"] == t) & (df["model"] == m)]
            mat[i, j] = choice_gap(d)

    # symmetric scale only if a real inversion exists (negative gap);
    # with fixed vmin=0 a true inversion is indistinguishable from zero
    vmin = -100 if np.nanmin(mat) < 0 else 0

    cmap = LinearSegmentedColormap.from_list("recept", ["#F4E4D4", "#E8A87C", "#2E6B8A"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=100, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(TRAITS)))
    ax.set_yticklabels(TRAITS, fontsize=10)

    for i in range(len(TRAITS)):
        for j in range(len(models)):
            v = mat[i, j]
            if v != v:
                continue
            weak = v < args.weak
            label = f"{v:.0f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10,
                    color="#B02020" if weak else ("white" if v > 55 else "#3A3A3A"),
                    fontweight="bold" if weak else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("high − low gap on high-pole choice (percentage points)", fontsize=9)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = f"output/mcheck/{args.name}-uptake_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[SUCCESS] heatmap saved to {out}")


if __name__ == "__main__":
    main()
