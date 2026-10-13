# Polybot PFS v5 — GP_U / GP_Q / RF / KDE and ablations

## Dataset and design
- Historical valid rows: 75; removed: 0.
- Finite beta oracle grid: 21 values within [-0.5, 1.5].
- NBI numerically feasible: 21/21; tolerance 0.0001.
- Seeds: 10; NBI beta-query budget per strategy: 12.
- All strategies share the same precomputed, fixed-GP NBI oracle; no new experimental results.
- Each query counts one beta evaluation, not the underlying SLSQP starts; use solver cost logs for compute-budget comparisons.

## Ablation: best surrogate utility (mean ± SD over seeds)
| Strategy | Mean | SD |
|---|---:|---:|
| full | 0.99815 | 0.00000 |
| no_gp_q | 0.99815 | 0.00000 |
| no_rf_kde | 0.99815 | 0.00000 |
| random_beta | 0.98427 | 0.01766 |

## Interpretation of models
- `full`: GP_U expected improvement + GP_Q predicted QoI utility adjustment + RF feasibility probability + KDE feasible-beta density.
- `no_gp_q`: GP_U + RF + KDE (remove GP_Q).
- `no_rf_kde`: GP_U + GP_Q (remove feasibility classification and KDE).
- `random_beta`: uniform random selection of remaining grid beta values after shared warm-start.
- GP_Q uses two regressors beta→(q1,q2), fitted only on evaluated feasible beta points.
- RF labels are numerical NBI solver feasible/infeasible. If labels have only one class, RF is NOT fitted; a constant feasibility prior is used.
- KDE uses successful beta evaluations only. Fallback is uniform if too few distinct betas exist.
- All β oracle points feasible: True; RF cannot learn a meaningful boundary on this finite grid.

## Highest-utility oracle solution (diagnostic only)
- β = (0.3000, 0.7000); predicted conductivity = 1257.52 S/cm; coverage = 97.14%; c = 0.086467; residual = 3.59e-11.
- DMSO concentration (Vol%): 1.618149904866444e-16
- EG concentration (Vol%): 4.819827551317387
- Coating speed (mm/sec): 1.0
- Coating temperature (Celsius): 90.0
- Post-processing solvent No.: 4.0
- Post coating speed (mm/sec): 1.0
- Post coating temperature (Celsius): 60.0

## Limitations and validity
- Only DMSO and EG are optimized continuously; other settings are fixed to observed templates.
- NBI equality is verified numerically, but multistart SLSQP does not certify global max c.
- Φ and n are built from predicted finite candidate anchors; extrapolations outside this design are not validated.
- Property GP was fitted on ALL available historical records; no prospective independent test.
- GP_U/GP_Q/RF/KDE are lightweight analogues; exact implementation may differ from original alloy paper.
- Utility and all performance numbers are predicted by fixed surrogates; ablation is not real-world BO performance evidence.
- A single global best candidate may dominate both objectives; check Pareto size before interpreting PFS improvements.
- Solvent labels are integer IDs; resolve the chemistry before any wet-lab run.

## Output contract
- dashboard.png: one multipanel figure (a–f).
- all_results.csv: historical/oracle/iteration rows, distinguished by `record_type`.
- REPORT.md: this report.
