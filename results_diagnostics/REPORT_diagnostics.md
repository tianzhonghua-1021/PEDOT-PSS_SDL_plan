# Polybot PFS/NBI geometry diagnostic report

- Historical records: 75 (dropped: 0)
- Unmeasured candidate pool: 650
- Outer BO queries: 9 out of 15 beta-grid points
- Candidate ray residual tolerance: 0.035
- Both Q objectives are minimized: q1 is normalized log conductivity loss; q2=1-coverage/100.
- CHIM normal is perpendicular by construction: n dot (anchor2-anchor1) = 0.000e+00
- **Phi, n and candidate Q(x) are fixed for all rounds.** No new experiments or GP retraining.
- NBI is finite-pool ray matching, NOT continuous equality-constrained NBI.
- If no candidate passes tolerance, a flagged fallback is shown; it is NOT an NBI solution.
- GP_U changes across iterations, but utility labels come from the same property surrogate.
- Solvent No. is categorical ID; identify chemistry before attempting experiments.
- GP uncertainty is not calibrated; recommendations should be validated independently.

## NBI geometry
Phi = [[0.04488 0.1357 ]
 [0.19886 0.03101]]
n = [-0.8795  -0.47589]
anchor candidate IDs = (98, 143)

## Best beta-queried (predicted utility, not measured)
- Round 4; beta=(0.2143, 0.7857)
- Candidate ID: 440; status: within_tolerance
- Predicted conductivity: 1022.655 S/cm
- Predicted coverage: 96.321%
- Projected c: 0.06308; residual: 0.00018

## Files
- geometry_panels.png: six-panel static summary
- geometry_iterations.gif: each beta query in Q-space (if Pillow available)
- iteration_trace.csv: phi, n, beta, c, Q, utility and GP_U diagnostics per round
- candidate_ray_diagnostics.csv: every candidate projection at every beta
- candidate_pool.csv: experimental recipe and predicted objectives per candidate
