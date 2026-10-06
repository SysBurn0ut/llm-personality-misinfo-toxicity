# Toxicity simulation (`output/toxicity/`)

Agents debate in a single comment thread opened by a divisive question (deporting undocumented immigrants). Each agent has a persona (name, age, education, Big Five profile) and a starting position drawn uniformly in [−1, 1]. The study checks whether toxicity emerges and evolves as in the human conversations of Avalle et al. (2024).

[Back to the main README](../../README.md)

## Design

| Configuration | Folder | Agents | Purpose |
|---|---|---|---|
| continuous | `tox-cont/` | 50 distinct LHS personas, models in rotation | closest to a real comment section |
| grid | `tox-grid/` | 32 profiles (all combinations of the 5 traits at 0/100) × 5 models = 160 | trait effects |
| single | `tox-single/` | 10 profiles (one trait at 0 or 100, others at 50) × 5 models = 50 | trait effects |

Every run: 4 rounds; each agent sees the opening message, the last 30 comments and its own memory; comments at temperature 1, memory updates at temperature 0. Each configuration is run with 3 seeds, drawn at random and different across configurations (continuous: 199723, 886078, 986795; grid: 369744, 616294, 782497; single: 478663, 540295, 791328), each in two conditions: **induced** (the system message asks the agent to reproduce human bias and toxicity) and **natural** (`noinduced` in the file names, sentence removed).

## Pipeline

**1. Threads** (model calls; documented, not needed to reproduce the results). The script writes under `output/toxicity/`:

```bash
python src/social_sim_mixed.py --persona-mode continuous --n 50 --rounds 4 --window 30 --seeds 199723 886078 986795 --out tox-cont/tox_cont
python src/social_sim_mixed.py --persona-mode grid   --rounds 4 --window 30 --seeds 369744 616294 782497 --out tox-grid/tox_grid
python src/social_sim_mixed.py --persona-mode single --rounds 4 --window 30 --seeds 478663 540295 791328 --out tox-single/tox_single
```

Output: `tox_<config>-<seed>-<induced|noinduced>.csv` (+ `.png`). Each row is a comment with model, persona, traits, starting position, refusal flag, Detoxify and RoBERTa scores.

**2. LLM judge** (model calls: Claude Haiku 4.5, temperature 0, 0-100 interpersonal toxicity rescaled to 0-1). Writes `-judged.csv` next to each thread file:

```bash
python src/rescore_judge.py --glob "output/toxicity/tox-cont/tox_cont-*.csv"
python src/rescore_judge.py --glob "output/toxicity/tox-grid/tox_grid-*.csv"
python src/rescore_judge.py --glob "output/toxicity/tox-single/tox_single-*.csv"
```

**3. Aggregate analysis** (no model calls; reads every `output/toxicity/*/tox_*-judged.csv`):

```bash
python src/tox_analysis.py                                                               # threshold 0.5 -> tox_analysis/
python src/tox_analysis.py --thr-detox 0.6 --out output/toxicity/tox_analysis_thr06     # robustness, threshold 0.6 -> tox_analysis_thr06/
```

A comment is toxic if its score is ≥ the threshold (0.5 for all scorers; 0.6 for Detoxify and RoBERTa in the robustness check, as in Avalle et al.). The toxicity of a thread or agent is the share of its toxic comments; refusals and failed calls are not counted.

## Files in `tox_analysis/` (and `tox_analysis_thr06/`)

| File | Content | Thesis |
|---|---|---|
| `by_model.csv` | share of toxic comments by model, configuration and condition (Detoxify and judge) | Tab. 4.10 |
| `trajectory.pdf/.png`, `trajectory_curves.csv`, `trajectory_tests.csv` | toxicity in 10 equal intervals of the thread, mean over seeds; Mann-Kendall test and slope per thread | Fig. 4.11 |
| `trajectory_per_run.pdf/.png` | the same for every single thread | Fig. E.8 |
| `scorers.csv`, `disagreements.csv` | Detoxify vs RoBERTa vs judge: Spearman, agreement, McNemar test; comments where Detoxify and judge disagree | "Agreement between measures", Ch. 4 |
| `traits_grid.csv` | agent toxicity ~ traits (high vs low) + model and seed fixed effects, cluster SE on the profile | Tab. 4.11 |
| `traits_single.csv` | trait at 100 vs 0 in the single configuration | text, Ch. 4 |

`tox_analysis_thr06/` is the robustness check cited in the thesis: the conclusions do not change with threshold 0.6.
