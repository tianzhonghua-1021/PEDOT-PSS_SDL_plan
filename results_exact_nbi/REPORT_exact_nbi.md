# PFS/NBI exact-equality diagnostic

- Historical rows: 75, dropped: 0
- Candidate Q points: 450; nondominated: 2
- Anchor distance: 0.095922132
- CHIM normal: [-0.16753025527666948, -0.9858669350206112]; orthogonality error: 3.469e-18
- Feasible NBI beta queries: **7/7** at equality tolerance 0.0001
- Inner problem solved with SLSQP equality constraints Q(x)-Phi*beta-c*n=0 and maximization of c.
- Solvent and coating/post-treatment settings are fixed per historical template; only DMSO and EG concentrations are continuous.
- Multi-start SLSQP is a **local solver**; meeting equality tolerance certifies numerical feasibility, NOT global optimality.
- NBI directions, CHIM and candidate Q predictions remain fixed. No real laboratory feedback.
- If no equality-feasible solution is found for a beta, no solution is fabricated.
- Optimization geometry uses GP predicted means, NOT actual measured Q values.
- GP_U labels are surrogate utilities; no true outer experiment feedback.
- No strict production-safe solvent chemical labels are available; solvent IDs must be checked.

## Geometry
Phi = [[0.0674255911376641, 0.16199204982866328], [0.022384523827249647, 0.006314664495997713]]
n = [-0.16753025527666948, -0.9858669350206112]

## Best equality-feasible beta query (surrogate-only)
- Round: 3; beta=(1.0000,0.0000)
- c=0.003211; residual=1.243e-11
- predicted conductivity=957.256 S/cm; coverage=98.078%
- DMSO concentration (Vol%): 1.5835845601588163
- EG concentration (Vol%): 0.9334950695547287
- Coating speed (mm/sec): 3.0
- Coating temperature (Celsius): 60.0
- Post-processing solvent No.: 4.0
- Post coating speed (mm/sec): 2.0
- Post coating temperature (Celsius): 70.0

## Outputs
- geometry_exact_nbi.png: six-panel overview
- geometry_by_iteration.gif: frame-by-frame exact NBI geometry
- iteration_trace_exact.csv: per beta search
- solver_attempts.csv: all SLSQP starts, including failures
- candidate_pool.csv: Q space and nondomination analysis
