# -*- coding: utf-8 -*-
"""
LEXICAL MARKERS IN THE AGENTS' REASONING: continuous vs poles runs.

1. Overall: share of reasoning texts containing each marker, mean length and
   share of distinct texts, per condition and per model.
2. By trait: agents with a LOW score (<= 40) vs a HIGH score (>= 60). With
   poles runs the same thresholds select exactly the 0-10 and 90-100 groups.

  sources    appeal to evidence or sources
  first      first-person reference
  emotional  emotional / evaluative vocabulary

Dictionary-based and not manually validated: exploratory.

Usage:
    python reasoning_markers.py --continuous DIR/run1 DIR/run2 DIR/run3 \
                                --poles DIR/p1 DIR/p2 DIR/p3 --name markers
"""
import argparse
import os

import pandas as pd

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TRAITS_IT = {"Openness": "Apertura", "Conscientiousness": "Coscienziosità",
             "Extraversion": "Estroversione", "Agreeableness": "Amicalità",
             "Neuroticism": "Nevroticismo"}
MARKERS = {
    "sources": r"(?i)credible|source|report|outlet|evidence",
    "first": r"(?i)\b(?:i|my|me)\b",
    "emotional": r"(?i)scary|outrage|shock|ridiculous|absurd|crazy|disgust|angry|worry|fear",
}
MARKERS_IT = {"sources": "Appello a fonti", "first": "Prima persona",
              "emotional": "Lessico emotivo"}
LOW, HIGH = 40, 60


def load(runs):
    d = pd.concat([pd.read_csv(f"output/replica/{r}-raw.csv", sep=None, engine="python")
                   for r in runs], ignore_index=True)
    d = d.dropna(subset=["reasoning"])
    d["reasoning"] = d["reasoning"].astype(str)
    for k, pat in MARKERS.items():
        d[k] = d["reasoning"].str.contains(pat, regex=True).astype(int)
    return d


def overall(d, condition):
    """Marker shares, length and diversity per model and in total."""
    d = d.assign(model=d["model"].str.split("/").str[-1],
                 words=d["reasoning"].str.split().str.len())
    rows = []
    for m, g in list(d.groupby("model")) + [("all", d)]:
        row = {"condition": condition, "model": m, "n": len(g),
               "words_mean": g["words"].mean(),
               "distinct_pct": g["reasoning"].nunique() / len(g) * 100}
        row.update({f"{k}_pct": g[k].mean() * 100 for k in MARKERS})
        rows.append(row)
    return pd.DataFrame(rows)


def latex_overall(ov, path):
    cond_it = {"continuous": "Continuo", "poles": "Poli"}
    tot = ov[ov.model == "all"].set_index("condition")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * (len(MARKERS) + 2) + "}", r"\toprule",
             "Condizione & " + " & ".join(MARKERS_IT[k] for k in MARKERS)
             + r" & Parole medie & Motivazioni distinte \\", r"\midrule"]
    for c, r in tot.iterrows():
        cells = [f"{r[k + '_pct']:.1f}\\%" for k in MARKERS]
        lines.append(f"{cond_it.get(c, c)} & " + " & ".join(cells)
                     + f" & {r['words_mean']:.1f} & {r['distinct_pct']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Motivazioni nei run continui e ai poli: quota di risposte che "
              r"contengono ciascun marcatore lessicale, lunghezza media e quota di "
              r"motivazioni distinte.}",
              r"\label{tab:reasoning-overall}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compare(d, condition):
    rows = []
    for t in TRAITS:
        lo, hi = d[d[t] <= LOW], d[d[t] >= HIGH]
        for k in MARKERS:
            rows.append({"condition": condition, "trait": t, "marker": k,
                         "n_low": len(lo), "n_high": len(hi),
                         "pct_low": lo[k].mean() * 100, "pct_high": hi[k].mean() * 100})
    r = pd.DataFrame(rows)
    r["diff_pp"] = r["pct_high"] - r["pct_low"]
    return r


def latex(res, path):
    conds = list(dict.fromkeys(res["condition"]))
    cond_it = {"continuous": "continuo", "poles": "poli"}
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\begin{tabular}{ll" + "c" * len(MARKERS) + "}", r"\toprule",
             "Tratto (basso $\\rightarrow$ alto) & Condizione & "
             + " & ".join(MARKERS_IT[k] for k in MARKERS) + r" \\", r"\midrule"]
    for t in TRAITS:
        for i, c in enumerate(conds):
            s = res[(res.trait == t) & (res.condition == c)].set_index("marker")
            cells = [f"{s.loc[k, 'pct_low']:.1f}\\% $\\rightarrow$ {s.loc[k, 'pct_high']:.1f}\\%"
                     for k in MARKERS]
            name = TRAITS_IT[t] if i == 0 else ""
            lines.append(f"{name} & {cond_it.get(c, c)} & " + " & ".join(cells) + r" \\")
        if t != TRAITS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Quota di motivazioni che contengono ciascun marcatore lessicale, "
              r"agenti con tratto basso ($\leq 40$) e alto ($\geq 60$).}",
              r"\label{tab:reasoning-markers}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuous", nargs="*", default=[])
    ap.add_argument("--poles", nargs="*", default=[])
    ap.add_argument("--name", default="reasoning_markers")
    args = ap.parse_args()

    parts, ovs = [], []
    for cond, runs in [("continuous", args.continuous), ("poles", args.poles)]:
        if runs:
            d = load(runs)
            parts.append(compare(d, cond))
            ovs.append(overall(d, cond))
    if not parts:
        raise SystemExit("pass at least one of --continuous / --poles")
    res = pd.concat(parts, ignore_index=True)
    ov = pd.concat(ovs, ignore_index=True)

    os.makedirs(os.path.dirname(f"output/replica/{args.name}.csv"), exist_ok=True)
    ov.round(2).to_csv(f"output/replica/{args.name}-overall.csv", index=False)
    latex_overall(ov, f"output/replica/{args.name}-overall.tex")
    print(ov.round(1).to_string(index=False))
    print()
    res.round(2).to_csv(f"output/replica/{args.name}.csv", index=False)
    latex(res, f"output/replica/{args.name}.tex")

    piv = res.pivot_table(index=["trait", "condition"], columns="marker",
                          values=["pct_low", "pct_high"], sort=False).round(1)
    print(piv.to_string())
    print(f"\n[SUCCESS] output/replica/{args.name}-overall.csv/.tex (continuous vs poles)"
          f" and output/replica/{args.name}.csv/.tex (low vs high trait)")
