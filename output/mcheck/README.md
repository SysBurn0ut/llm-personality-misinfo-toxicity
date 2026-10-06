# Manipulation check (`output/mcheck/`)

For each trait, agents with the trait at 90 (high pole) or 10 (low pole), the other four at 50, face a behavioural probe with two options (A/B). The primary measure is the choice, normalised as `high_choice` (agent picked the high-pole option, whatever its position); confidence is descriptive only. A trait counts as taken up if the high group picks the high pole more often than the low group (one-sided Fisher exact test, p < .05).

[Back to the main README](../../README.md)

## Run

| Folder | Design | Seed | Collection command |
|---|---|---|---|
| `MCHECK-100A-NR` | 100 repetitions × 2 poles × 5 traits × 5 models = 5000 calls, temperature 0, A/B order counterbalanced | 42 (A/B order only) | `python src/manipulation_check.py --name MCHECK-100A-NR/mcheck --n_per_group 100 --counterbalance` |

The collection command calls the models and is not needed to reproduce the results.

## Analysis (no model calls)

```bash
python src/manipulation_check.py --from_csv --name MCHECK-100A-NR/mcheck   # -> mcheck-summary.csv (+ per-model table printed)
python src/plot_uptake_heatmap.py --name MCHECK-100A-NR/mcheck              # -> mcheck-uptake_heatmap.png
python src/plot_mcheck_strip.py --name MCHECK-100A-NR/mcheck                # -> mcheck-strip_trait_pole.png
```

## Files and thesis results

| File | Content | Thesis |
|---|---|---|
| `mcheck-raw.csv` | one row per call: trait, group, model, A/B order, choice, confidence, provider (collected data) | |
| `mcheck-summary.csv` | per trait: % high-pole choices in the high and low group, choice gap, Fisher p, mean confidence, verdict | uptake results, Ch. 4 |
| `mcheck-strip_trait_pole.png` | confidence per trait and model, by pole | Fig. 4.1 |
| `mcheck-uptake_heatmap.png` | choice gap per trait × model | Fig. E.1 |
