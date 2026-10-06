# Scripts (`src/`)

Run every script **from the repository root** (e.g. `python src/main.py ...`). Each script reads and writes under the `output/` folder of its experiment; `--name` takes `<run folder>/<file prefix>`, relative to that folder.

[Back to the main README](../README.md)

## Shared modules

| File | Role |
|---|---|
| `utils.py` | OpenRouter client, provider policy per model (`PROVIDER_POLICY`), reasoning switch, token and provider logging. Reads `OPENROUTER_API_KEY` on import. |
| `personas.py` | persona generation (names, age, education, Big Five via Latin Hypercube Sampling) and rendering of the personality block |
| `prompt.py` | system and task prompts of the fake-news experiment |
| `citizen.py`, `world.py` | agent and simulation loop of the fake-news experiment |
| `recalibration.py` | recalibration of `confidence_true` when it contradicts the stated belief |

## Data collection (call the models)

| Script | Experiment | Output folder |
|---|---|---|
| `manipulation_check.py` | manipulation check | `output/mcheck/` |
| `bfi_test.py` | BFI-44 administration | `output/bfi/` |
| `main.py` | fake-news replication (also re-analysis with `--from_raw`) | `output/replica/` |
| `social_sim_mixed.py` | toxicity threads | `output/toxicity/` |
| `rescore_judge.py` | LLM-judge rescoring of the toxicity threads | next to the input files |

## Analysis (no model calls)

| Script | Input → output |
|---|---|
| `main.py --from_raw [--no_recal]` | `-raw.csv` → `-agents.csv` / `-orig-agents.csv` |
| `analyze_per_model.py [--pooled] [--figs]` | `-agents.csv` → per-model regressions, Holm correction, trait × model interaction, pooled regressions, figures |
| `thesis_figures.py` | `-agents.csv` → per-model trait-effect grids, heatmap, replication grid |
| `headline_sensitivity.py [--no_recal]` | raw files of the original-headline runs → headline-level sensitivity and model × headline interaction |
| `summarize_holm.py` | all `*_holm.csv` → Holm summary (Tab. 4.6) |
| `reasoning_markers.py` | raw files → lexical markers in the reasoning texts |
| `bfi_diagnostics.py [--no_recal]` | BFI raw/scores + replication run → diagnostics and incremental validity |
| `plot_uptake_heatmap.py`, `plot_mcheck_strip.py` | manipulation-check raw → figures |
| `tox_analysis.py` | `tox_*-judged.csv` → aggregate toxicity analysis |

Usage examples and the commands behind every thesis result are in the folder READMEs ([replica](../output/replica/README.md), [bfi](../output/bfi/README.md), [mcheck](../output/mcheck/README.md), [toxicity](../output/toxicity/README.md)).

## Setup checks (`src/check/`)

Diagnostic scripts used to choose and configure the models. They call the OpenRouter API and do not produce thesis results.

| Script | Purpose | Command |
|---|---|---|
| `check_models.py` | lists, for each model, whether it supports and bills reasoning (public endpoint, no key needed) | `python src/check/check_models.py` |
| `check_providers.py` | lists the providers available for each model, with exact slug and quantization (no key needed); used to fill `PROVIDER_POLICY` | `python src/check/check_providers.py` |
| `check_reasoning.py` | measures the reasoning tokens actually spent per model | `cd src` then `python -m check.check_reasoning` |
| `check_reasoning_off.py` | verifies that each model accepts disabling reasoning (needs a real key) | `python src/check/check_reasoning_off.py` |

`check_reasoning.py` imports `utils`, so it must be run as a module from `src/`.
