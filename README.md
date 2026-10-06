# LLM agents with Big Five personalities: misinformation and toxicity

Code and data for the Master's thesis *"Incarnazione o imitazione: i limiti degli agenti artificiali in contesti di disinformazione e tossicità"* (Francesco Riccardo Perna, MSc in Computer Security, Università degli Studi di Milano, A.Y. 2025-2026).

The thesis tests whether LLM agents prompted with Big Five personality profiles can stand in for human participants, in three studies:

1. **Trait validation**: a manipulation check and the BFI-44 questionnaire administered to the agents.
2. **Fake-news replication**: an in-silico replication of Wolverton & Stevens (2019), with three variants (original headlines, recent headlines, traits at the poles) and an exploratory reasoning variant.
3. **Toxicity simulation**: agents debating in a single comment thread, compared with the human dynamics described by Avalle et al. (2024).

**Thesis version:** the code and data used for the thesis correspond to tag [`v1.0-tesi`](https://github.com/SysBurn0ut/llm-personality-misinfo-toxicity/releases/tag/v1.0-tesi).

## Repository layout

| Path | Content | Documentation |
|---|---|---|
| `src/` | All scripts (data collection and analysis) | [src/README.md](src/README.md) |
| `output/bfi/` | BFI-44 administration, diagnostics, incremental validity (Tab. 4.1) | [output/bfi/README.md](output/bfi/README.md) |
| `output/mcheck/` | Manipulation check | [output/mcheck/README.md](output/mcheck/README.md) |
| `output/replica/` | Fake-news replication: 9 main runs, reasoning variant, headline sensitivity, Holm summary, lexical markers | [output/replica/README.md](output/replica/README.md) |
| `output/toxicity/` | Toxicity simulation: threads, LLM-judge rescoring, aggregate analysis | [output/toxicity/README.md](output/toxicity/README.md) |

In every run folder, the file ending in `-raw.csv` (and, for toxicity, `-judged.csv`) holds the data collected from the models. **Every other file is regenerated from it by the analysis scripts**, with no model calls.

## Setup

- Python **3.12.8**
- `pip install -r requirements.txt` (pinned versions)
- Run every command **from the repository root**.
- The environment variable `OPENROUTER_API_KEY` must be defined, even for re-analysis only: `src/utils.py` reads it on import. When no model is called (all analysis commands in this README), any value works:
  - PowerShell: `$env:OPENROUTER_API_KEY = "dummy"`
  - bash: `export OPENROUTER_API_KEY=dummy`

## Models and providers

All generations go through OpenRouter. Each model is pinned to a single provider with fallbacks disabled (`PROVIDER_POLICY` in `src/utils.py`).

| Model (OpenRouter slug) | Provider | Reasoning |
|---|---|---|
| `deepseek/deepseek-v4-pro` (preview, build 0423) | **DeepSeek** for manipulation check, BFI-44 and replication; **StreamLake** for toxicity (DeepSeek provider no longer available) | supported, disabled except in the reasoning variant |
| `google/gemini-3.1-flash-lite` | Google | supported, disabled except in the reasoning variant |
| `meta-llama/llama-4-maverick` | DeepInfra | not supported |
| `openai/gpt-4o-mini` | OpenAI | not supported |
| `qwen/qwen3.7-plus` | Alibaba | supported, disabled except in the reasoning variant |

The current `src/utils.py` routes DeepSeek to StreamLake (the last experiment run). To re-collect the first two studies with the original setup, set its `"only"` field to `["DeepSeek"]`.

Toxicity scoring: Detoxify (`unbiased`), RoBERTa (`s-nlp/roberta_toxicity_classifier`) and an LLM judge, `anthropic/claude-haiku-4.5` at temperature 0.

## Main experimental parameters

| Experiment | Personas / agents | Seeds | Temperature | Other |
|---|---|---|---|---|
| Manipulation check | 100 repetitions × 2 poles × 5 traits × 5 models | 42 (A/B order only) | 0 | traits at 90/10, others at 50; A/B order counterbalanced |
| BFI-44 | 100 personas × 5 models per seed | 12, 42, 73 | 0 | same personas as the replication |
| Replication | 100 personas × 5 models per run | 12, 42, 73 | 0 | personas via Latin Hypercube Sampling; poles variant: traits in 0-10 / 90-100 |
| Reasoning variant | 30 personas × 5 models | 42 | 0 | reasoning effort `low` vs off |
| Toxicity | continuous 50, grid 160, single 50 agents | 3 per configuration, drawn at random (listed in [output/toxicity/README.md](output/toxicity/README.md)) | 1 for comments, 0 for memory and judge | 4 rounds, window of the last 30 comments, induced vs natural condition |

Responses are not guaranteed to be deterministic even at temperature 0: re-collecting the data will not reproduce the raw files exactly. The seeds make the **design** reproducible; the **analyses** are reproducible exactly from the saved raw data.

## Reproducing the main results

Commands below; details, file lists and the remaining tables are in the folder READMEs.

| Thesis result | Command (from repo root) | Output |
|---|---|---|
| Tab. 4.1, BFI incremental validity (recalibrated) | `python src/bfi_diagnostics.py --name BFI-100A-S42/bfiS42 --behavior 100A-S42-NR/100A-S42-NR` (same for S12, S73) | `output/bfi/BFI-100A-S*/bfiS*-added_value.csv` |
| Tab. F.1, same on non-recalibrated data | same command with `--no_recal` | `output/bfi/BFI-100A-S*/bfiS*-added_value-orig.csv` |
| Tab. 4.2, 4.4, 4.5 and F.2–F.4, per-model regressions (original headlines, recent headlines, poles) | [replication pipeline](output/replica/README.md#per-run-pipeline) on the 9 runs | `output/replica/<run>/<run>[-orig]-per_model_conf_on_fake_holm.csv` |
| Tab. 4.3, model × headline interaction; Tab. E.2 | `python src/headline_sensitivity.py` | `output/replica/HEADLINE-SENS/` |
| Tab. 4.6, significant combinations before/after Holm | `python src/summarize_holm.py` | `output/replica/holm_summary/holm_summary_counts.csv` |
| Tab. 4.10, toxic comments by model, configuration and condition | `python src/tox_analysis.py` | `output/toxicity/tox_analysis/by_model.csv` |
| Tab. 4.11, trait effects on toxicity (grid) | `python src/tox_analysis.py` | `output/toxicity/tox_analysis/traits_grid.csv` |
| Fig. 4.11 and E.8, toxicity along the thread | `python src/tox_analysis.py` | `output/toxicity/tox_analysis/trajectory*.pdf` |
| Same toxicity analyses with threshold 0.6 | `python src/tox_analysis.py --thr-detox 0.6 --out output/toxicity/tox_analysis_thr06` | `output/toxicity/tox_analysis_thr06/` |

### Recalibrated and original data

In the fake-news study, `confidence_true` is recalibrated when it contradicts the stated belief (`src/recalibration.py`); this mostly affects GPT-4o mini. The recalibrated data are the main analysis; the non-recalibrated ones are reported in Appendix F. Files of the non-recalibrated version carry `-orig` in their name and are produced with `--no_recal`.

## Reproducibility check

All analyses listed above, plus those in the folder READMEs, were re-run from the saved raw data on 6 October 2026, and the regenerated CSV/TeX outputs were compared with the committed ones: all values used in the thesis are reproduced exactly. PDF figures always differ at byte level because they embed the creation date.
