# BFI-44 (`output/bfi/`)

The 44 items of the Big Five Inventory (John & Srivastava, 1999) are administered to the agents. The recovered scores are compared with the scores imposed in the prompt, and the incremental validity test checks whether they predict behaviour in the replication beyond the imposed scores.

[Back to the main README](../../README.md)

## Runs

| Folder | Seed | Personas | Collection command |
|---|---|---|---|
| `BFI-100A-S12` | 12 | 100 × 5 models | `python src/bfi_test.py --name BFI-100A-S12/bfiS12 --n_personas 100 --seed 12` |
| `BFI-100A-S42` | 42 | 100 × 5 | same with `S42` / `--seed 42` |
| `BFI-100A-S73` | 73 | 100 × 5 | same with `S73` / `--seed 73` |
| `BFI-POOLED` | 12 + 42 + 73 | 1500 administrations | aggregate of the three runs (no collection) |

Personas are the same as the replication runs with the same seed (same seed, same system message), so each BFI score is linked to the agent's behaviour through persona × model. Temperature 0; item order shuffled at every administration. Collection commands call the models and are not needed to reproduce the results.

## Analysis (no model calls)

```bash
# per seed: diagnostics + incremental validity against the replication run with the same seed
python src/bfi_diagnostics.py --name BFI-100A-S42/bfiS42 --behavior 100A-S42-NR/100A-S42-NR
python src/bfi_diagnostics.py --name BFI-100A-S42/bfiS42 --behavior 100A-S42-NR/100A-S42-NR --no_recal
# (repeat for S12 and S73)

# pooled diagnostics on the three seeds (incremental validity is computed per seed only)
python src/bfi_diagnostics.py --name BFI-100A-S12/bfiS12 BFI-100A-S42/bfiS42 BFI-100A-S73/bfiS73
```

The incremental validity reads `output/replica/100A-S<seed>-NR/` (raw and agents files), so the replication pipeline for those runs must have been run first.

## Files

| File | Content |
|---|---|
| `bfiS<seed>-raw.csv` | item-by-item answers, imposed and recovered scores (collected data) |
| `bfiS<seed>-scores.csv`, `bfiS<seed>-corr.csv`, `bfiS<seed>-bfi.png` | recovered scores, imposed × recovered correlation per model and trait, scatter grid (`bfi_test.py`) |
| `bfiS<seed>-diagnostics.csv`, `-scale_use.csv`, `-diag_*.png/.pdf` | per-seed diagnostics (`bfi_diagnostics.py`) |
| `bfiS<seed>-added_value.csv` | incremental validity, recalibrated `conf_on_fake` |
| `bfiS<seed>-added_value-orig.csv` | incremental validity, non-recalibrated `conf_on_fake` |
| `BFI-POOLED/bfi_pooled-*` | pooled diagnostics and figures |

## Thesis results

| Result | Source |
|---|---|
| Tab. 4.1, contribution of the BFI scores (R², ΔR² imposed, ΔR² BFI, linear and robustness Wald tests) | `bfiS12/42/73-added_value.csv` |
| Tab. F.1, same on non-recalibrated data | `bfiS12/42/73-added_value-orig.csv` |
| Tab. E.1, diagnostics per model and trait on the pooled sample | `BFI-POOLED/bfi_pooled-diagnostics.csv` |
| Fig. 4.3 (flat answers and recovery), Fig. 4.4 (Cronbach's alpha) | `BFI-POOLED/bfi_pooled-diag_flat_recovery.*`, `bfi_pooled-diag_alpha.*` |
