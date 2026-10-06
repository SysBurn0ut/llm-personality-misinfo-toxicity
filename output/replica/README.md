# Fake-news replication (`output/replica/`)

In-silico replication of Wolverton & Stevens (2019): each agent rates the truthfulness of true and fake headlines (belief TRUE/FALSE and `confidence_true` 0-100). Main dependent variable: `conf_on_fake`, the mean confidence given to the fake headlines.

[Back to the main README](../../README.md)

## Runs

| Folder | Variant | Seed | Personas | Collection command |
|---|---|---|---|---|
| `100A-S12-NR`, `100A-S42-NR`, `100A-S73-NR` | original headlines (Tab. C.1) | 12 / 42 / 73 | 100 × 5 models | `python src/main.py --name 100A-S42-NR/100A-S42-NR --n_personas 100 --seed 42` |
| `100A-S12-NR-NEW`, `100A-S42-NR-NEW`, `100A-S73-NR-NEW` | recent headlines (Tab. C.2); the Fauci headline is a known-fact control, excluded from the analysis | 12 / 42 / 73 | 100 × 5 | add `--headlines recent` |
| `100A-S12-NR-POLES`, `100A-S42-NR-POLES`, `100A-S73-NR-POLES` | traits at the poles (0-10 / 90-100) | 12 / 42 / 73 | 100 × 5 | add `--persona_mode poles --pole_margin 10` |
| `30A-R-OFF` | reasoning variant, reasoning off | 42 | 30 × 5 | `python src/main.py --name 30A-R-OFF/30A-R-OFF --n_personas 30 --seed 42` |
| `30A-R-LOW` | reasoning variant, effort `low` | 42 | 30 × 5 | add `--reasoning low` |

`NR` = no reasoning. Collection commands call the models; they document the parameters and are not needed to reproduce the results.

## Per-run pipeline

For each run `R` (no model calls):

```bash
python src/main.py --from_raw --name R/R                        # -> R-agents.csv, R-trait_effects.png
python src/main.py --from_raw --no_recal --name R/R             # -> R-orig-agents.csv
python src/analyze_per_model.py --pooled --figs --name R/R        # recalibrated data
python src/analyze_per_model.py --pooled --figs --name R/R-orig   # non-recalibrated data
python src/thesis_figures.py --name R/R --yscale both           # add --singles for the 25 single panels (+ zip)
```

PowerShell loop over the 9 main runs:

```powershell
$runs = "100A-S12-NR","100A-S42-NR","100A-S73-NR",
        "100A-S12-NR-NEW","100A-S42-NR-NEW","100A-S73-NR-NEW",
        "100A-S12-NR-POLES","100A-S42-NR-POLES","100A-S73-NR-POLES"
foreach ($r in $runs) {
    python src/main.py --from_raw --name "$r/$r"
    python src/main.py --from_raw --no_recal --name "$r/$r"
    python src/analyze_per_model.py --pooled --figs --name "$r/$r"
    python src/analyze_per_model.py --pooled --figs --name "$r/$r-orig"
    python src/thesis_figures.py --name "$r/$r" --yscale both
}
```

### Files in a run folder

| File | Content | Produced by |
|---|---|---|
| `R-raw.csv` | one row per agent × headline: belief, confidence, reasoning text, tokens, provider | `main.py` (model calls) |
| `R-agents.csv`, `R-orig-agents.csv` | one row per agent: traits, demographics, `conf_on_fake`, `conf_on_true`, `conf_discernment`, `n_fake_correct`, ... | `main.py --from_raw` |
| `R[-orig]-reg_<dv>.csv` | pooled regression (model fixed effects + traits + controls), one file per dependent variable | `analyze_per_model.py --pooled` |
| `R[-orig]-replication_table.csv` | replication summary against the human study | `analyze_per_model.py --pooled` |
| `R[-orig]-per_model_conf_on_fake.csv` | per-model regressions (25 model × trait tests) | `analyze_per_model.py` |
| `R[-orig]-per_model_conf_on_fake_holm.csv` / `.png` | same with Holm correction (`p_holm`, `sig_raw`, `sig_holm`); family = the 25 tests of the run | `analyze_per_model.py` |
| `R[-orig]-interaction_conf_on_fake.csv` | trait × model interaction, cluster-robust Wald test | `analyze_per_model.py` |
| `R-ceiling.png`, `R-human_vs_llm.png` | figures | `analyze_per_model.py --figs` |
| `R-single_traits_{centered,absolute}.*`, `R-heatmap_belief.*`, `R-replication.*` | figures | `thesis_figures.py` |

## Thesis results

| Result | Source |
|---|---|
| Tab. 4.2 (original headlines), 4.4 (recent), 4.5 (poles); F.2–F.4 on non-recalibrated data | `R[-orig]-per_model_conf_on_fake_holm.csv` of the three seeds of each variant: sign of `coef` and `sig_raw` (symbols + / −), `sig_holm` (⊕ / ⊖) |
| Pooled ("extended") regressions in the text of Ch. 4 and App. F | `R-reg_<dv>.csv` and `R-orig-reg_<dv>.csv` |
| Trait × model interaction (F tests in the text) | `R[-orig]-interaction_conf_on_fake.csv` |
| Fig. 4.5, mean rating of fake headlines by imposed-score band (seed 42) | `100A-S42-NR/100A-S42-NR-single_traits_absolute.png` |
| Fig. 4.6 (original headlines), 4.9 (recent), 4.10 (poles), seed 42; E.2–E.7 for seeds 12 and 73 | `R-heatmap_belief.png` of the corresponding run |
| Tab. 4.3, model × headline interaction | `HEADLINE-SENS/headline_interaction.txt` |
| Tab. E.2, aggregate vs single-headline regression | `HEADLINE-SENS/headline_sens_table.txt` |
| Fig. 4.7, mean rating per model and headline | `HEADLINE-SENS/heatmap_model_headline.png` (`HEADLINE-SENS-orig/` for the original data) |
| Tab. 4.6, significant combinations before/after Holm | `holm_summary/holm_summary_counts.csv` |
| Tab. 4.7 and 4.8, lexical markers in the agents' reasoning | `markers/markers-overall.csv` and `markers/markers.csv` (`.tex` versions alongside) |
| Tab. 4.9 (recalibrated) and F.5 (non-recalibrated), reasoning variant | means per model of `30A-R-*-agents.csv` and `30A-R-*-orig-agents.csv` (command below) |
| Reasoning tokens per response (118 / 479 / 1621) | mean of `reasoning_tokens` in `30A-R-LOW-raw.csv` (command below) |

### Commands for the shared outputs

```bash
python src/headline_sensitivity.py              # -> HEADLINE-SENS/  (seeds 12, 42, 73, original headlines)
python src/headline_sensitivity.py --no_recal   # -> HEADLINE-SENS-orig/
python src/summarize_holm.py                    # -> holm_summary/   (reads every *_holm.csv under output/replica)
python src/reasoning_markers.py --continuous 100A-S12-NR/100A-S12-NR 100A-S42-NR/100A-S42-NR 100A-S73-NR/100A-S73-NR --poles 100A-S12-NR-POLES/100A-S12-NR-POLES 100A-S42-NR-POLES/100A-S42-NR-POLES 100A-S73-NR-POLES/100A-S73-NR-POLES --name markers/markers
```

Reasoning variant: run `main.py --from_raw` (with and without `--no_recal`) on `30A-R-OFF` and `30A-R-LOW` first, then:

```bash
# Tab. 4.9 (rows without -orig) and Tab. F.5 (rows with -orig)
python -c "import pandas as pd; [print(r+s, pd.read_csv(f'output/replica/{r}/{r}{s}-agents.csv').groupby('model')[['conf_on_fake','conf_on_true','conf_discernment']].mean().round(1), sep='\n') for s in ['','-orig'] for r in ['30A-R-OFF','30A-R-LOW']]"

# reasoning tokens per response
python -c "import pandas as pd; print(pd.read_csv('output/replica/30A-R-LOW/30A-R-LOW-raw.csv').groupby('model')['reasoning_tokens'].mean().round(0))"
```

In Tab. F.5 two values are rounded half-up (GPT-4o mini 76.450 → 76.5, Qwen3.7-Plus 82.450 → 82.5); pandas prints 76.4 and 82.4 because of floating-point representation.
