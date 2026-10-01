# llm-personality-misinfo-toxicity
Code and data for my MSc thesis (Università degli Studi di Milano): do LLM agents with imposed Big Five traits behave like humans? Tested on five models in fake news recognition (Wolverton &amp; Stevens replication) and online toxicity (after Avalle et al.).

# llm-personas-embodiment

Code and data for the MSc thesis *"Incarnazione o imitazione: i limiti degli agenti artificiali in contesti di disinformazione e tossicità"* (Francesco Riccardo Perna, Università degli Studi di Milano, A.Y. 2025-2026).

The project tests whether LLM agents with imposed Big Five (OCEAN) traits behave like humans with the same personality. Five models are compared: DeepSeek-V4-Pro, Gemini 3.1 Flash-Lite, Llama 4 Maverick, GPT-4o mini and Qwen3.7-Plus.

## Experiments

1. **Trait installation**: manipulation check and BFI-44 administration, with diagnostics on response quality.
2. **Fake news recognition**: in-silico replication of Wolverton & Stevens (2020), with three variants (recent headlines, extreme traits, reasoning on).
3. **Toxicity simulation**: multi-agent online thread, compared with the patterns reported by Avalle et al. (2024).

## Files

| Area | Scripts |
|---|---|
| Personas and prompts | `personas.py`, `prompt.py` |
| Model access (OpenRouter) | `utils.py` |
| Setup checks | `check_providers.py`, `check_models.py`, `check_reasoning.py`, `check_reasoning_off.py` |
| Manipulation check | `manipulation_check.py`, `plot_uptake_heatmap.py`, `plot_confidence_strip.py` |
| BFI-44 | `bfi_test.py`, `bfi_diagnostics.py` |
| Fake news replication | `main.py`, `world.py`, `citizen.py`, `analyze_per_model.py`, `recalibration.py`, `reasoning_markers.py`, `thesis_figures.py` |
| Toxicity simulation | `social_sim_mixed.py`, `rescore_judge.py`, `tox_analysis.py` |

Results (CSV, figures) are in `output/`.

## Setup

Python 3.12.

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key      # Windows: set OPENROUTER_API_KEY=your_key
```

Each script exposes its parameters via command line:

```bash
python <script>.py --help
```

## Notes on reproducibility

- Every model is pinned to a single provider with fallbacks disabled (`PROVIDER_POLICY` in `utils.py`).
- DeepSeek-V4-Pro was served by the **DeepSeek** provider in the manipulation check, BFI-44 and replication, and by **StreamLake** in the toxicity simulation (DeepSeek was no longer available). `utils.py` is currently set to StreamLake: switch it back to DeepSeek to rerun the first experiments as in the thesis.
- Seeds fix personas and presentation order, not model outputs: responses are not guaranteed to be deterministic, even at temperature 0.
- The `confidence_true` recalibration (`recalibration.py`) was defined after observing the data; non-recalibrated results are kept in `output/` and reported in Appendix F of the thesis.
