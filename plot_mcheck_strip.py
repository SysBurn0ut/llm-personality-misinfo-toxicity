"""
Strip plot of manipulation-check confidence, per trait and model, coloured by pole.

How to:
    python plot_mcheck_strip.py --name mcheck
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
COLORS = {"high": "#E8A87C", "low": "#2E6B8A"}
OFFSET = {"high": -0.2, "low": 0.2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mcheck")
    ap.add_argument("--csv", default=None, help="path to the raw csv (default output/<name>-raw.csv)")
    args = ap.parse_args()

    path = args.csv or f"output/{args.name}-raw.csv"
    df = pd.read_csv(path, sep=None, engine="python").dropna(subset=["confidence"])
    models = sorted(df["model"].unique())
    rng = np.random.default_rng(0)

    fig, grid = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    axes = grid.ravel()
    for ax, t in zip(axes, TRAITS):
        for j, m in enumerate(models):
            for g in ("high", "low"):
                y = df[(df["trait"] == t) & (df["model"] == m) & (df["group"] == g)]["confidence"]
                if y.empty:
                    continue
                x = j + OFFSET[g] + rng.uniform(-0.1, 0.1, len(y))
                ax.scatter(x, y, s=22, color=COLORS[g], alpha=0.8,
                           edgecolors="white", linewidths=0.4, zorder=2)
        ax.set_title(t, fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=30, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_ylim(60, 100)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6, zorder=0)
        for k in range(1, len(models)):
            ax.axvline(k - 0.5, color="#EEEEEE", lw=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in grid[:, 0]:
        ax.set_ylabel("confidence (0–100)")
    axes[-1].axis("off")

    handles = [Line2D([], [], marker="o", ls="", color=COLORS["high"], label="high pole (90)"),
               Line2D([], [], marker="o", ls="", color=COLORS["low"], label="low pole (10)")]
    fig.tight_layout()
    axes[-1].legend(handles=handles, loc="center", frameon=False, fontsize=11)
    out = f"output/{args.name}-strip_trait_pole.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[SUCCESS] saved {out}")


if __name__ == "__main__":
    main()
