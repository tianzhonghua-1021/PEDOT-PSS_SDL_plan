# Polybot PEDOT:PSS — PFS toy experiment report

- Input: 75 records (75 valid); split: {'Test': 45, 'Train': 30}. 
- Target 1: maximize conductivity (S/cm); target 2: maximize coverage (%).
- DMSO and EG are additive concentrations; the post-processing solvent is supplied only as a numeric category ID.
- Conductivity is modeled in log10 space because values range across orders of magnitude.

## Next experiment (model recommendation)
- Selection: **PFS GP_U + finite-pool approximate NBI**
- **DMSO concentration (Vol%)**: 0.173
- **EG concentration (Vol%)**: 4.711
- **Coating speed (mm/sec)**: 1.000
- **Coating temperature (Celsius)**: 95.000
- **Post-processing solvent No.**: 5.000
- **Post coating speed (mm/sec)**: 1.000
- **Post coating temperature (Celsius)**: 60.000
- Predicted conductivity: **1228.206 S/cm** (model log10 SD 0.627).
- Predicted coverage: **98.57%** (model SD 13.94 percentage points).

## Offline validation (Random Forest sanity-check on original Train/Test split)
- Conductivity log10 MAE: 0.835; R²: -7.500; test size: 45.
- Coverage fraction MAE: 0.227; R²: -2.186; test size: 45.

## PFS / NBI
- Evaluated β values: 15; feasible NBI solutions: 15.
- β = (β1, 1−β1) specifies the NBI CHIM reference, not raw property percentages.
- The inner method searches a finite candidate pool: sampled DMSO/EG concentrations and observed combinations of other settings; the solvent ID is categorical.
- Finite-pool NBI uses orthogonal projection and residual tolerance 0.035 in normalized objective space. It is NOT an exact equality-constrained NBI solver.
- GP_U-style Expected Improvement searches β using predicted balanced utility from the property surrogate. GP_Q, RF feasibility and KDE are not implemented in this simplified proof of concept.

## Critical limitations
- All NBI results use surrogate predictions, not fresh measurements; this is NOT a validated autonomous wet-lab closed loop.
- The original Test rows are used in exploratory surrogate fitting and recommendation; the RF Train/Test diagnostic is separate and must NOT be treated as independent validation of the final GP.
- Solvent category numbers cannot be translated into solvent chemical identities without a lookup table.
- Recommendations interpolate DMSO and EG only within observed ranges; prediction SDs are not experimentally calibrated.
- Very low conductivity and high measured replicate SD in some rows warrant inspection before experimental execution.
- Geometric-mean joint utility is an assumed preference; change it if experimental priorities differ.
- If original Train/Test baseline R² is negative, treat recommendation as exploratory only; inspect distribution shift and validate with new data.

## Output schema
- all_results.csv: record_type historical / candidate / pfs_beta / next_experiment.
- dashboard.png: six-panel summary.
- REPORT.md: this file.
