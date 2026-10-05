"""
Summary of all Holm CSVs (*_holm.csv) produced by analyze_per_model.py.
Searches recursively under --root, groups runs into "variants" (file name
without the seed token) and reports, for each variant:
  1. significant effects before/after Holm, per seed;
  2. stable effects: model x trait retained in >= --min-seeds seeds with
     the same sign across all seeds where retained;
  3. a cross-variant matrix (number of seeds retained per variant), to
     compare variants (e.g. recalibrated vs orig, paper vs poles).

Usage:
    python summarize_holm.py --root output
    python summarize_holm.py --root output --min-seeds 3 --out holm_summary
    python summarize_holm.py --root output --exclude=-orig --out holm_recal
"""
import argparse
import glob
import os
import re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="output", help="folder to search recursively")
ap.add_argument("--pattern", default="*_holm.csv")
ap.add_argument("--exclude", default=None,
                help="regex: skip files whose run name matches (e.g. --exclude=-orig)")
ap.add_argument("--seed-regex", default=r"(?:^|-)S(\d+)(?=-|$)",
                help="regex identifying the seed token in the run name")
ap.add_argument("--min-seeds", type=int, default=2,
                help="minimum seeds retained for an effect to be called stable")
ap.add_argument("--all-rows", action="store_true",
                help="heatmap: show all model x trait rows, not only those retained at least once")
ap.add_argument("--order", default=None,
                help="heatmap: comma-separated column order of variants (others appended)")
ap.add_argument("--labels", default=None,
                help='heatmap: readable column names, e.g. "100A-NR=paper/recal,100A-NR-orig=paper/orig"'
                     ' ("/" splits group and sub-label; columns of the same group are framed together)')
ap.add_argument("--out", default="holm_summary", help="output file prefix")
args = ap.parse_args()

seed_re = re.compile(args.seed_regex)
frames = []
for f in sorted(glob.glob(os.path.join(args.root, "**", args.pattern), recursive=True)):
    run = re.sub(r"-per_model_.*_holm\.csv$", "", os.path.basename(f))
    if args.exclude and re.search(args.exclude, run):
        continue
    m = seed_re.search(run)
    if not m:
        print(f"[skip] no seed in {f}")
        continue
    d = pd.read_csv(f)
    d["seed"] = "S" + m.group(1)
    d["variant"] = seed_re.sub("", run, count=1).strip("-") or "base"
    d["model"] = d["model"].astype(str).str.split("/").str[-1]
    frames.append(d)

if not frames:
    raise SystemExit("No Holm CSV found.")
a = pd.concat(frames, ignore_index=True)
print(f"Read {len(frames)} files, {a['variant'].nunique()} variants, "
      f"{a['seed'].nunique()} seeds\n")

# 1. counts before/after per variant x seed
cnt = (a.groupby(["variant", "seed"])
        .agg(k=("p", "size"), sig_raw=("sig_raw", "sum"), sig_holm=("sig_holm", "sum"))
        .reset_index())
tot = cnt.groupby("variant")[["k", "sig_raw", "sig_holm"]].sum().reset_index()
tot["seed"] = "TOTAL"
cnt = pd.concat([cnt, tot], ignore_index=True).sort_values(["variant", "seed"])
bad = cnt[(cnt.seed != "TOTAL") & (cnt.k != cnt[cnt.seed != "TOTAL"].k.mode()[0])]
print("=" * 70, "\n SIGNIFICANT BEFORE -> AFTER HOLM\n" + "=" * 70)
print(cnt.to_string(index=False))
if not bad.empty:
    print("\n[WARNING] families with non-standard size:\n", bad.to_string(index=False))

# 2. stability per variant x model x trait
def summarize(g):
    r = g[g.sig_holm]
    signs = set((r.coef > 0).tolist())
    return pd.Series({
        "n_seeds": g.seed.nunique(),
        "n_sig_raw": int(g.sig_raw.sum()),
        "n_retained": int(g.sig_holm.sum()),
        "seeds_retained": ",".join(sorted(r.seed)),
        "sign_consistent": len(signs) <= 1,
        "sign": ("+" if signs == {True} else "-" if signs == {False}
                 else "mixed" if signs else ""),
        "coef_mean_retained": r.coef.mean() if len(r) else float("nan"),
        "coef_min": g.coef.min(), "coef_max": g.coef.max(),
    })

st = (a.groupby(["variant", "model", "trait"]).apply(summarize, include_groups=False)
       .reset_index())
st["stable"] = (st.n_retained >= args.min_seeds) & st.sign_consistent
st = st.sort_values(["variant", "stable", "n_retained"], ascending=[True, False, False])

print("\n" + "=" * 70)
print(f" STABLE EFFECTS (retained in >= {args.min_seeds} seeds, consistent sign)")
print("=" * 70)
cols = ["model", "trait", "n_retained", "n_seeds", "seeds_retained", "sign",
        "coef_mean_retained"]
for v, g in st.groupby("variant"):
    s = g[g.stable]
    print(f"\n## {v}")
    print(s[cols].round(2).to_string(index=False) if len(s) else "   (none)")
    once = g[(g.n_retained > 0) & ~g.stable]
    if len(once):
        print("   retained but not stable: " + "; ".join(
            f"{r.model} x {r.trait} [{r.seeds_retained}, {r.sign}]"
            for r in once.itertuples()))

# 3. cross-variant matrix
mat = st.pivot_table(index=["model", "trait"], columns="variant",
                     values="n_retained", aggfunc="first").fillna(0).astype(int)
sgn = st.pivot_table(index=["model", "trait"], columns="variant",
                     values="sign", aggfunc="first").fillna("")
mat = mat[(mat > 0).any(axis=1)]
cell = mat.astype(str) + sgn.loc[mat.index, mat.columns]
cell = cell.where(mat > 0, "-")
print("\n" + "=" * 70)
print(" CROSS-VARIANT MATRIX: seeds retained + sign (- = never retained)")
print("=" * 70)
print(cell.to_string())
mixed = st[st.sign == "mixed"]
flips = (st[st.n_retained > 0].groupby(["model", "trait"]).sign
         .agg(lambda s: {"+", "-"} <= set(s) or "mixed" in set(s)))
flips = flips[flips]
if len(flips):
    print("\n[WARNING] retained effects with opposite sign across seeds or variants:")
    for (mo, tr) in flips.index:
        print(f"   {mo} x {tr}")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
cnt.to_csv(f"{args.out}_counts.csv", index=False)
st.to_csv(f"{args.out}_stability.csv", index=False)
cell.to_csv(f"{args.out}_matrix.csv")
print(f"\n[SUCCESS] saved {args.out}_counts.csv, {args.out}_stability.csv, "
      f"{args.out}_matrix.csv")


# 4. summary heatmap: rows = model x trait, columns = variants
def plot_summary_heatmap(st, path, all_rows, order=None, labels=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    variants = sorted(st.variant.unique())
    if order:
        first = [v for v in order.split(",") if v in variants]
        variants = first + [v for v in variants if v not in first]
    lab_map = {}
    if labels:
        for kv in labels.split(","):
            k, v = kv.split("=", 1)
            lab_map[k.strip()] = v.strip()
    col_lab = [lab_map.get(v, v) for v in variants]   # "group/sub" -> two lines
    groups = [c.split("/")[0] for c in col_lab]

    keys = st[["model", "trait"]].drop_duplicates().sort_values(["model", "trait"])
    if not all_rows:
        keep = st[st.n_retained > 0][["model", "trait"]].drop_duplicates()
        keys = keys.merge(keep, on=["model", "trait"])
    keys = list(keys.itertuples(index=False, name=None))
    idx = st.set_index(["model", "trait", "variant"])
    vmax = int(st.n_seeds.max())

    RED = ["#F4C7BE", "#E07A66", "#B2281A"]
    BLUE = ["#C3D8EE", "#6A9FD1", "#1F4E8C"]
    GREY, EMPTY = "#BDBDBD", "#F7F7F7"

    def shade(pal, n, tot):
        return pal[min(len(pal) - 1, max(0, round(n / tot * len(pal)) - 1))]

    # fixed size per cell, so the grid does not shrink with few columns
    CW, RH, L, R, T, B = 1.1, 0.42, 3.4, 3.8, 0.9, 0.2
    W, H = L + CW * len(variants) + R, T + RH * len(keys) + B
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, CW * len(variants) / W, RH * len(keys) / H])
    for i, (mo, tr) in enumerate(keys):
        for j, v in enumerate(variants):
            color, txt, dark = EMPTY, "", False
            if (mo, tr, v) not in idx.index:
                txt = "n/a"
            else:
                r = idx.loc[(mo, tr, v)]
                n, tot = int(r.n_retained), int(r.n_seeds)
                if n > 0:
                    if r.sign == "mixed":
                        color, sym = GREY, "±"
                    else:
                        pal = RED if r.sign == "+" else BLUE
                        color = shade(pal, n, tot)
                        sym = "+" if r.sign == "+" else "−"
                        dark = color in (RED[-1], BLUE[-1])
                    txt = f"{sym} {n}/{tot}"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color,
                                       edgecolor="white", lw=2))
            if txt:
                ax.text(j + .5, i + .5, txt, ha="center", va="center", fontsize=9.5,
                        weight="bold", color="white" if dark else "black")

    ax.set_xlim(0, len(variants)); ax.set_ylim(len(keys), 0)
    ax.set_xticks(np.arange(len(variants)) + .5)
    ax.set_xticklabels([c.replace("/", "\n") for c in col_lab], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(keys)) + .5)
    ax.set_yticklabels([f"{mo}  ×  {tr}" for mo, tr in keys], fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(1, len(keys)):
        if keys[i][0] != keys[i - 1][0]:
            ax.axhline(i, color="#444444", lw=1.2)
    for j in range(1, len(variants)):
        if groups[j] != groups[j - 1]:
            ax.axvline(j, color="#444444", lw=1.2)

    ax.set_title("Model × trait effects that remain significant after Holm correction",
                 fontsize=11.5, pad=38 if any("/" in c for c in col_lab) else 24)
    handles = ([Patch(facecolor=c, label=f"+  {k + 1}/{vmax} seeds") for k, c in enumerate(RED)]
               + [Patch(facecolor=c, label=f"−  {k + 1}/{vmax} seeds") for k, c in enumerate(BLUE)])
    leg = ax.legend(handles=handles, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1),
                    frameon=False, fontsize=9, title="retained in", title_fontsize=9)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x0 + 0.01, bb.y0 - 0.02,
            "+  same direction as Wolverton & Stevens (2019)\n"
            "−  opposite direction (inverts the human result)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] heatmap saved to {path}")


plot_summary_heatmap(st, f"{args.out}_heatmap.png", args.all_rows, args.order, args.labels)
