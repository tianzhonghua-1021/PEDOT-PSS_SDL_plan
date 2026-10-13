# Polybot PFS V9 report

## 1. Data and constraints
- Valid rows after cleaning: **75**
- Dropped rows: **0**
- Conductivity constraint: **sigma >= 665.000 S/cm**
- Coverage constraint: **coverage >= 72.000%**
- Continuous-variable bounds are set to **90% of observed minimum** and **110% of observed maximum**.
- Post-processing solvent is discrete and follows the supplied 8-code table.

### Post-processing solvent mapping
| No. | MeOH | EtOH | H2O |
|---:|---:|---:|---:|
| 1 | 100 | 0 | 0 |
| 2 | 20 | 80 | 0 |
| 3 | 40 | 60 | 0 |
| 4 | 50 | 50 | 0 |
| 5 | 60 | 40 | 0 |
| 6 | 80 | 20 | 0 |
| 7 | 0 | 100 | 0 |
| 8 | 80 | 0 | 20 |

## 2. Core formulas implemented in code
1. **QoI / loss map**
   - `predict_q(...)` defines
     - `q1 = (log_sigma_max - log10(sigma)) / (log_sigma_max - log_sigma_min)`
     - `q2 = 1 - coverage_fraction`
2. **Threshold-centered utility**
   - `utility_from_predicted_properties(...)` defines
     - `U = 1.5 * ReLU((sigma - sigma_min)/sigma_min) + 1.0 * ReLU((coverage - coverage_min)/coverage_min)`
3. **Exact NBI subproblem**
   - in `build_oracle_solver(...)`, SLSQP solves
     - `max c`
     - subject to `Q(x) - Phi*beta - c*n = 0`
     - plus `sigma >= sigma_min` and `coverage >= coverage_min`
4. **Outer-loop models**
   - `select_next_beta(...)` contains
     - `GP_U`: beta -> utility
     - `GP_Q`: beta -> (q1, q2) with two GPs
     - `RF`: beta -> feasibility label
     - `KDE`: density over feasible beta history

## 3. Oracle / feasibility summary
- Beta grid size: **31**
- Feasible oracle solutions: **28/31**
- Equality tolerance: **1.0e-04**
- RF trainable from oracle labels (both classes present): **True**

## 4. Ablation summary (best utility across budget)
| Strategy | Mean final best utility | Std |
|---|---:|---:|
| Full PFS | 3.312408 | 0.494658 |
| No GP_Q | 3.469358 | 0.371268 |
| No RF/KDE | 3.471356 | 0.365116 |
| Random beta | 3.474442 | 0.337360 |

## 5. Highest-utility feasible oracle solution
- beta = (**-0.167**, **1.167**)
- utility = **3.590248**
- c = **0.107149**
- sigma = **2083.859 S/cm**
- coverage = **100.067%**
- solvent no. = **5** (MeOH/EtOH/H2O = 60/40/0)
- DMSO = **0.000 vol%**, EG = **4.890 vol%**
- coating speed/temp = **1.136 mm/s**, **93.694 C**
- post speed/temp = **1.170 mm/s**, **68.774 C**

## 6. Limitations
- This is a fixed-surrogate offline study; no new wet-lab measurements are added online.
- SLSQP satisfying the NBI equality numerically does not certify a global optimum for `c`.
- When all or most beta points are feasible, RF feasibility classification may carry limited extra value.
- Utility is threshold-centered and dimensionless; it is suitable for ranking under the specified thresholds, but is not a physical property by itself.

## 7. V7 diagnostics and recommendation
- Cached oracle reused: **False**
- Shared random initialization per seed: **3 beta queries**
- Recommendation is saved in **next_experiment.json**, based on the queried Full PFS path.
- GP_U / GP_Q / RF / KDE can be inactive when insufficient diverse training labels exist.
