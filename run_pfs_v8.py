#!/usr/bin/env python3
"""
Polybot PEDOT:PSS PFS V6
========================
Full offline PFS demonstration with:
- GP_U (beta -> utility)
- GP_Q (beta -> predicted QoIs / losses)
- RF feasibility classifier (beta -> P[NBI feasible])
- KDE-guided beta sampling on feasible beta history
- exact equality-constrained NBI solver
- explicit experimental and performance constraints

Outputs only FOUR files in the requested output folder:
    dashboard.png   (dpi=600)
    all_results.csv
    REPORT.md
    nbi_process.gif (1 second per round; rendered from 600 dpi frames)

Run from the folder containing polybot-PEDOT_experiment.csv, e.g.
    python run_pfs_v6.py

IMPORTANT SCIENTIFIC NOTES
--------------------------
1) This is still a FIXED-SURROGATE OFFLINE study. New lab data are NOT added online.
2) "Exact NBI" means the numerical optimizer enforces the equalities
       Q(x) = Phi beta + c n
   to within a numerical tolerance; it does NOT certify global optimality.
3) Utility uses threshold-centered ReLU margins as requested:
       U = 1.5 * ReLU_sigma + 1.0 * ReLU_coverage
   where
       ReLU_sigma    = max((sigma - sigma_min) / sigma_min, 0)
       ReLU_coverage = max((coverage - coverage_min) / coverage_min, 0)
   This keeps the two terms dimensionless and threshold-referenced.
"""

from __future__ import annotations
import argparse
import json
import hashlib
from dataclasses import asdict
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from scipy.optimize import minimize
from scipy.stats import norm, gaussian_kde
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# DATASET COLUMNS
# -----------------------------------------------------------------------------
FEATURES = [
    'DMSO concentration (Vol%)',
    'EG concentration (Vol%)',
    'Coating speed (mm/sec)',
    'Coating temperature (Celsius)',
    'Post-processing solvent No.',
    'Post coating speed (mm/sec)',
    'Post coating temperature (Celsius)',
]
NUMERIC_CONTROLS = [
    'DMSO concentration (Vol%)',
    'EG concentration (Vol%)',
    'Coating speed (mm/sec)',
    'Coating temperature (Celsius)',
    'Post coating speed (mm/sec)',
    'Post coating temperature (Celsius)',
]
SOLVENT_COL = 'Post-processing solvent No.'
CONDUCTIVITY = 'Average conductivity (S/cm)'
COVERAGE = 'Average coverage (%)'

# Post-processing solvent table supplied by user.
SOLVENT_MAP = {
    1: (100, 0, 0),
    2: (20, 80, 0),
    3: (40, 60, 0),
    4: (50, 50, 0),
    5: (60, 40, 0),
    6: (80, 20, 0),
    7: (0, 100, 0),
    8: (80, 0, 20),
}


# -----------------------------------------------------------------------------
# HELPER MATH
# -----------------------------------------------------------------------------
def expected_improvement(mu, sd, best, xi=0.01):
    """Standard EI for maximization."""
    sd = np.asarray(sd, dtype=float)
    mu = np.asarray(mu, dtype=float)
    imp = mu - best - xi
    z = imp / np.maximum(sd, 1e-12)
    ei = imp * norm.cdf(z) + sd * norm.pdf(z)
    return np.where(sd > 1e-12, ei, np.maximum(imp, 0.0))


def normalize01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def compute_pareto_mask(points):
    """For minimization in 2D."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if np.any(dominates):
            mask[i] = False
    return mask


# -----------------------------------------------------------------------------
# IO / DATA PREP
# -----------------------------------------------------------------------------
def load_clean_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    for col in FEATURES + [CONDUCTIVITY, COVERAGE]:
        if col not in df.columns:
            raise ValueError(f'Missing required CSV column: {col}')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    mask = (
        df[FEATURES].notna().all(axis=1)
        & df[CONDUCTIVITY].gt(0)
        & df[COVERAGE].between(0, 100)
    )
    clean = df.loc[mask].copy().reset_index(drop=True)
    dropped = int((~mask).sum())
    if len(clean) < 15:
        raise ValueError('Too few valid rows after cleaning (<15).')
    return clean, dropped


def build_control_bounds(clean: pd.DataFrame, expand_min=0.90, expand_max=1.10):
    """
    Experimental-domain constraints requested by user:
    all continuous variables use 90%*min and 110%*max bounds (±10%),
    while post-processing solvent is chosen by integer code 1..8.
    """
    bounds = {}
    for col in NUMERIC_CONTROLS:
        lo = float(clean[col].min())
        hi = float(clean[col].max())
        lb = expand_min * lo
        ub = expand_max * hi
        if col.endswith('(Vol%)') or 'speed' in col.lower() or 'temperature' in col.lower():
            lb = max(0.0, lb)
        bounds[col] = (lb, ub)
    return bounds


# -----------------------------------------------------------------------------
# PROPERTY SURROGATES: x -> conductivity, coverage, and Q(x)
# Core variables/formulas located here:
#   - transformer, gp_models
#   - q1, q2 definitions inside predict_q()
# -----------------------------------------------------------------------------
def train_property_surrogates(clean: pd.DataFrame):
    transformer = ColumnTransformer([
        ('num', StandardScaler(), [0, 1, 2, 3, 5, 6]),
        ('solvent', OneHotEncoder(handle_unknown='ignore'), [4]),
    ])
    X = transformer.fit_transform(clean[FEATURES])
    if hasattr(X, 'toarray'):
        X = X.toarray()

    y_log_sigma = np.log10(clean[CONDUCTIVITY].to_numpy(float))
    y_cov_frac = clean[COVERAGE].to_numpy(float) / 100.0

    gp_models = []
    for y in (y_log_sigma, y_cov_frac):
        kernel = (
            ConstantKernel(1.0, constant_value_bounds='fixed')
            * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds='fixed', nu=2.5)
            + WhiteKernel(noise_level=0.05, noise_level_bounds='fixed')
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,
            normalize_y=True,
            alpha=1e-5,
        )
        gp.fit(X, y)
        gp_models.append(gp)

    sigma_log_min = float(y_log_sigma.min())
    sigma_log_max = float(y_log_sigma.max())
    return transformer, gp_models, (sigma_log_min, sigma_log_max)


def predict_properties(df_like, transformer, gp_models):
    X = transformer.transform(pd.DataFrame(df_like, columns=FEATURES))
    if hasattr(X, 'toarray'):
        X = X.toarray()
    mu_log_sigma, sd_log_sigma = gp_models[0].predict(X, return_std=True)
    mu_cov_frac, sd_cov_frac = gp_models[1].predict(X, return_std=True)
    return mu_log_sigma, mu_cov_frac, sd_log_sigma, sd_cov_frac


def predict_q(df_like, transformer, gp_models, log_sigma_bounds):
    """
    QoI / loss map Q(x) used in NBI.

    q1 = normalized log-conductivity loss        (smaller is better)
    q2 = coverage loss = 1 - coverage_fraction   (smaller is better)
    """
    mu_log_sigma, mu_cov_frac, sd_log_sigma, sd_cov_frac = predict_properties(
        df_like, transformer, gp_models
    )
    lo, hi = log_sigma_bounds
    width = max(hi - lo, 1e-12)
    q1 = (hi - mu_log_sigma) / width
    q2 = 1.0 - mu_cov_frac
    q = np.column_stack([q1, q2])
    sigma = 10 ** mu_log_sigma
    coverage = 100.0 * mu_cov_frac
    return q, sigma, coverage, sd_log_sigma, sd_cov_frac


# -----------------------------------------------------------------------------
# OBJECTIVE / UTILITY
# Core formula location:
#   U = 1.5*ReLU_sigma + 1.0*ReLU_coverage
# -----------------------------------------------------------------------------
def utility_from_predicted_properties(sigma, coverage, sigma_min, coverage_min):
    relu_sigma = np.maximum((np.asarray(sigma) - sigma_min) / sigma_min, 0.0)
    relu_coverage = np.maximum((np.asarray(coverage) - coverage_min) / coverage_min, 0.0)
    return 1.5 * relu_sigma + 1.0 * relu_coverage


# -----------------------------------------------------------------------------
# CANDIDATE CLOUD FOR VISUALIZATION / CHIM ANCHORS
# -----------------------------------------------------------------------------
def sample_candidate_cloud(bounds, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    solvents = list(SOLVENT_MAP.keys())
    for _ in range(n):
        row = {
            'DMSO concentration (Vol%)': rng.uniform(*bounds['DMSO concentration (Vol%)']),
            'EG concentration (Vol%)': rng.uniform(*bounds['EG concentration (Vol%)']),
            'Coating speed (mm/sec)': rng.uniform(*bounds['Coating speed (mm/sec)']),
            'Coating temperature (Celsius)': rng.uniform(*bounds['Coating temperature (Celsius)']),
            'Post-processing solvent No.': int(rng.choice(solvents)),
            'Post coating speed (mm/sec)': rng.uniform(*bounds['Post coating speed (mm/sec)']),
            'Post coating temperature (Celsius)': rng.uniform(*bounds['Post coating temperature (Celsius)']),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def compute_chim_and_normal(q_cloud):
    """2-objective CHIM and a southwest-pointing unit normal for minimization."""
    a = int(np.argmin(q_cloud[:, 0]))
    b = int(np.argmin(q_cloud[:, 1]))
    phi = np.column_stack([q_cloud[a], q_cloud[b]])
    d = phi[:, 1] - phi[:, 0]
    if np.linalg.norm(d) < 1e-12:
        raise RuntimeError('Degenerate CHIM: identical anchor points.')
    n = np.array([d[1], -d[0]], dtype=float)
    n /= np.linalg.norm(n)
    if np.sum(n) > 0:
        n *= -1.0
    return phi, n, a, b


@dataclass
class OracleRecord:
    beta1: float
    beta2: float
    feasible: int
    solver_successes: int
    starts_tested: int
    c: float
    residual: float
    q1: float
    q2: float
    sigma: float
    coverage: float
    utility: float
    rf_target: int
    ref_q1: float
    ref_q2: float
    n1: float
    n2: float
    phi11: float
    phi12: float
    phi21: float
    phi22: float
    solvent_no: float
    meoh_pct: float
    etoh_pct: float
    h2o_pct: float
    dmso: float
    eg: float
    coat_speed: float
    coat_temp: float
    post_speed: float
    post_temp: float


# -----------------------------------------------------------------------------
# EXACT NBI SOLVER
# Core section requested by user. Main variables/formulas:
#   beta = [beta1, beta2]
#   ref  = Phi @ beta
#   NBI equality constraints:
#       Q(x) - Phi beta - c n = 0
#   Additional constraints:
#       sigma(x)   >= sigma_min
#       coverage(x) >= coverage_min
#       continuous controls inside 90%min / 110%max bounds
#       solvent fixed by integer code 1..8 and enumerated outside SLSQP
# -----------------------------------------------------------------------------
def build_oracle_solver(clean, bounds, transformer, gp_models, log_sigma_bounds,
                        phi, n, sigma_min, coverage_min, eq_tol,
                        start_rows_per_solvent=3, maxiter=200):
    numeric_names = NUMERIC_CONTROLS[:]  # 6 continuous controls
    raw = clean[FEATURES].copy()

    def eval_point(values_dict):
        frame = pd.DataFrame([values_dict], columns=FEATURES)
        q, sigma, coverage, _, _ = predict_q(frame, transformer, gp_models, log_sigma_bounds)
        return q[0], float(sigma[0]), float(coverage[0])

    # Precompute solver templates / starts from observed rows, split by solvent.
    starts_by_solvent = {}
    for s in SOLVENT_MAP:
        sub = raw.loc[raw[SOLVENT_COL] == s, numeric_names]
        if len(sub) == 0:
            sub = raw[numeric_names]
        starts_by_solvent[s] = sub.drop_duplicates().head(start_rows_per_solvent).to_numpy(float)

    lb = np.array([bounds[name][0] for name in numeric_names] + [-2.5], dtype=float)
    ub = np.array([bounds[name][1] for name in numeric_names] + [2.5], dtype=float)
    var_bounds = list(zip(lb, ub))

    def solve_for_beta(beta1):
        beta1 = float(beta1)
        beta2 = 1.0 - beta1
        ref = phi @ np.array([beta1, beta2], dtype=float)
        best = None
        total_attempts = 0
        success_count = 0

        for solvent_no, starts in starts_by_solvent.items():
            meoh, etoh, h2o = SOLVENT_MAP[solvent_no]
            for start in starts:
                total_attempts += 1
                x0_controls = np.clip(start, lb[:-1], ub[:-1])
                point0 = dict(zip(numeric_names, x0_controls))
                point0[SOLVENT_COL] = float(solvent_no)
                q0, _, _ = eval_point(point0)
                c0 = float(np.dot(q0 - ref, n))
                z0 = np.r_[x0_controls, np.clip(c0, lb[-1], ub[-1])]

                def unpack(z):
                    vals = {
                        'DMSO concentration (Vol%)': float(z[0]),
                        'EG concentration (Vol%)': float(z[1]),
                        'Coating speed (mm/sec)': float(z[2]),
                        'Coating temperature (Celsius)': float(z[3]),
                        'Post-processing solvent No.': float(solvent_no),
                        'Post coating speed (mm/sec)': float(z[4]),
                        'Post coating temperature (Celsius)': float(z[5]),
                    }
                    q, sigma, coverage = eval_point(vals)
                    c = float(z[6])
                    return vals, q, sigma, coverage, c

                def obj(z):
                    return -float(z[6])  # maximize c

                def eq_fun(z):
                    _, q, _, _, c = unpack(z)
                    # EXACT NBI equality: Q(x) - Phi beta - c n = 0
                    return q - ref - c * n

                def ineq_sigma(z):
                    _, _, sigma, _, _ = unpack(z)
                    return sigma - sigma_min

                def ineq_coverage(z):
                    _, _, _, coverage, _ = unpack(z)
                    return coverage - coverage_min

                constraints = [
                    {'type': 'eq', 'fun': lambda z, k=k: eq_fun(z)[k]} for k in range(2)
                ] + [
                    {'type': 'ineq', 'fun': ineq_sigma},
                    {'type': 'ineq', 'fun': ineq_coverage},
                ]

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res = minimize(
                        obj,
                        z0,
                        method='SLSQP',
                        bounds=var_bounds,
                        constraints=constraints,
                        options={'maxiter': maxiter, 'ftol': 1e-10},
                    )
                vals, q, sigma, coverage, c = unpack(res.x)
                residual = float(np.linalg.norm(q - ref - c * n))
                feasible = (
                    np.isfinite(residual)
                    and residual <= eq_tol
                    and sigma >= sigma_min - 1e-9
                    and coverage >= coverage_min - 1e-9
                )
                if feasible:
                    success_count += 1
                    utility = float(utility_from_predicted_properties(sigma, coverage, sigma_min, coverage_min))
                    row = OracleRecord(
                        beta1=beta1, beta2=beta2, feasible=1,
                        solver_successes=success_count, starts_tested=total_attempts,
                        c=float(c), residual=residual,
                        q1=float(q[0]), q2=float(q[1]), sigma=float(sigma), coverage=float(coverage),
                        utility=utility, rf_target=1,
                        ref_q1=float(ref[0]), ref_q2=float(ref[1]), n1=float(n[0]), n2=float(n[1]),
                        phi11=float(phi[0, 0]), phi12=float(phi[0, 1]), phi21=float(phi[1, 0]), phi22=float(phi[1, 1]),
                        solvent_no=float(solvent_no), meoh_pct=float(meoh), etoh_pct=float(etoh), h2o_pct=float(h2o),
                        dmso=float(vals['DMSO concentration (Vol%)']), eg=float(vals['EG concentration (Vol%)']),
                        coat_speed=float(vals['Coating speed (mm/sec)']), coat_temp=float(vals['Coating temperature (Celsius)']),
                        post_speed=float(vals['Post coating speed (mm/sec)']), post_temp=float(vals['Post coating temperature (Celsius)']),
                    )
                    if best is None or row.c > best.c:
                        best = row

        if best is None:
            return OracleRecord(
                beta1=beta1, beta2=beta2, feasible=0,
                solver_successes=0, starts_tested=total_attempts,
                c=np.nan, residual=np.nan, q1=np.nan, q2=np.nan,
                sigma=np.nan, coverage=np.nan, utility=np.nan,
                rf_target=0,
                ref_q1=float(ref[0]), ref_q2=float(ref[1]), n1=float(n[0]), n2=float(n[1]),
                phi11=float(phi[0, 0]), phi12=float(phi[0, 1]), phi21=float(phi[1, 0]), phi22=float(phi[1, 1]),
                solvent_no=np.nan, meoh_pct=np.nan, etoh_pct=np.nan, h2o_pct=np.nan,
                dmso=np.nan, eg=np.nan, coat_speed=np.nan, coat_temp=np.nan, post_speed=np.nan, post_temp=np.nan,
            )
        return best

    return solve_for_beta


# -----------------------------------------------------------------------------
# BETA-LEVEL MODELS FOR PFS OUTER LOOP
# Core sections requested by user:
#   1) GP_U: beta -> utility
#   2) GP_Q: beta -> q1, q2  (two separate GPs)
#   3) RF:   beta -> feasibility label
#   4) KDE:  feasible-beta density over candidate beta grid
# -----------------------------------------------------------------------------
def fit_gp_1d(x, y, noise=0.005):
    kernel = (
        ConstantKernel(1.0, constant_value_bounds='fixed')
        * Matern(length_scale=0.35, length_scale_bounds='fixed', nu=2.5)
        + WhiteKernel(noise_level=noise, noise_level_bounds='fixed')
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=True,
        alpha=1e-5,
    )
    gp.fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y, dtype=float))
    return gp


def select_next_beta(history, available, strategy, seed, q_hi_ref, warmup=None):
    """
    strategy in {'full', 'no_gp_q', 'no_rf_kde', 'random_beta'}

    Returns: chosen_beta, diagnostics dict
    """
    rng = np.random.default_rng(seed)
    available = np.array(sorted(available), dtype=float)

    # V7: shared random initialization per seed for ALL strategies, including random.
    # This avoids revealing the best beta to GP-driven strategies in round 2.
    if warmup is not None and len(history) < len(warmup):
        chosen = float(warmup[len(history)])
        assert np.any(np.isclose(available, chosen, atol=1e-8)), 'Warmup beta absent'
        return chosen, {'rf_trained': 0, 'kde_used': 0, 'gpq_used': 0, 'gpu_used': 0,
                        'score': np.nan, 'p_feas': np.nan, 'ei': np.nan, 'kde_score': np.nan,
                        'q_score': np.nan, 'mu_u': np.nan, 'sd_u': np.nan}

    if strategy == 'random_beta':
        chosen = float(rng.choice(available))
        return chosen, {'rf_trained': 0, 'kde_used': 0, 'gpq_used': 0, 'gpu_used': 0,
                        'score': np.nan, 'p_feas': np.nan, 'ei': np.nan, 'kde_score': np.nan,
                        'q_score': np.nan, 'mu_u': np.nan, 'sd_u': np.nan}

    # Before enough history is accumulated, choose a neutral yet unqueried beta.
    # With --warmup >= 3 this fallback normally does not run.
    if len(history) < 2:
        chosen = float(available[len(available)//2])
        return chosen, {'rf_trained': 0, 'kde_used': 0, 'gpq_used': 0, 'gpu_used': 0,
                        'score': np.nan, 'p_feas': np.nan, 'ei': np.nan, 'kde_score': np.nan,
                        'q_score': np.nan, 'mu_u': np.nan, 'sd_u': np.nan}

    hist = pd.DataFrame(history)
    x_all = hist['beta1'].to_numpy(float)
    y_feas = hist['feasible'].to_numpy(int)
    x_av = available.reshape(-1, 1)

    # ----- RF FEASIBILITY CLASSIFIER -----
    # Learns beta -> P(feasible). Requires both classes.
    p_feas = np.full(len(available), np.mean(y_feas) if len(y_feas) else 0.5)
    rf_trained = 0
    if strategy == 'full' and len(np.unique(y_feas)) == 2 and len(hist) >= 6:
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=seed,
        )
        rf.fit(x_all.reshape(-1, 1), y_feas)
        classes = list(rf.classes_)
        p_feas = rf.predict_proba(x_av)[:, classes.index(1)]
        rf_trained = 1

    # Feasible history only for GP_U / GP_Q / KDE.
    good = hist.loc[hist['feasible'] == 1].copy()
    kde_used = 0
    gpq_used = 0
    gpu_used = 0

    # Defaults if too little feasible data.
    q_score = np.zeros(len(available))
    kde_score = np.zeros(len(available))
    mu_u = np.full(len(available), np.nan)
    sd_u = np.full(len(available), np.nan)
    ei = np.full(len(available), np.nan)

    if len(good) >= 2:
        # ----- GP_U -----
        gpu_used = 1
        gp_u = fit_gp_1d(good['beta1'].to_numpy(float), good['utility'].to_numpy(float), noise=0.002)
        mu_u, sd_u = gp_u.predict(x_av, return_std=True)
        ei = expected_improvement(mu_u, sd_u, float(good['utility'].max()), xi=0.003)
        score = normalize01(ei)

        # ----- GP_Q -----
        # Two separate GPs: beta -> q1 and beta -> q2.
        if strategy in ('full', 'no_rf_kde') and len(good) >= 3:
            gpq_used = 1
            gp_q1 = fit_gp_1d(good['beta1'].to_numpy(float), good['q1'].to_numpy(float), noise=0.003)
            gp_q2 = fit_gp_1d(good['beta1'].to_numpy(float), good['q2'].to_numpy(float), noise=0.003)
            mu_q1 = gp_q1.predict(x_av)
            mu_q2 = gp_q2.predict(x_av)
            # Higher score if predicted losses are smaller.
            q_good1 = np.clip((q_hi_ref[0] - mu_q1) / max(q_hi_ref[0], 1e-12), 0.0, 1.0)
            q_good2 = np.clip((q_hi_ref[1] - mu_q2) / max(q_hi_ref[1], 1e-12), 0.0, 1.0)
            q_score = np.sqrt(q_good1 * q_good2)
            score = score + 0.25 * normalize01(q_score)

        # ----- KDE-GUIDED BETA SAMPLING -----
        # Density over feasible queried betas. Used only in 'full'.
        if strategy == 'full' and len(good) >= 4 and np.std(good['beta1'].to_numpy(float)) > 1e-6:
            kde_used = 1
            try:
                kde = gaussian_kde(good['beta1'].to_numpy(float), bw_method='scott')
                kde_score = kde.evaluate(available)
                score = score + 0.20 * normalize01(kde_score)
            except Exception:
                kde_score = np.zeros(len(available))

        # RF feasibility weight only in full.
        if strategy == 'full':
            score = score * (0.20 + 0.80 * p_feas)
    else:
        # If too little feasible data, explore around the center with a simple heuristic.
        score = -np.abs(available - 0.5)

    chosen_idx = int(np.nanargmax(score))
    chosen = float(available[chosen_idx])
    return chosen, {
        'rf_trained': rf_trained,
        'kde_used': kde_used,
        'gpq_used': gpq_used,
        'gpu_used': gpu_used,
        'score': float(score[chosen_idx]),
        'p_feas': float(p_feas[chosen_idx]) if np.isfinite(p_feas[chosen_idx]) else np.nan,
        'ei': float(ei[chosen_idx]) if np.isfinite(ei[chosen_idx]) else np.nan,
        'kde_score': float(kde_score[chosen_idx]) if np.isfinite(kde_score[chosen_idx]) else np.nan,
        'q_score': float(q_score[chosen_idx]) if np.isfinite(q_score[chosen_idx]) else np.nan,
        'mu_u': float(mu_u[chosen_idx]) if np.isfinite(mu_u[chosen_idx]) else np.nan,
        'sd_u': float(sd_u[chosen_idx]) if np.isfinite(sd_u[chosen_idx]) else np.nan,
    }


def run_strategy(strategy, beta_grid, oracle_map, budget, seed, q_hi_ref, warmup=None, progress=None, verbose=True):
    available = set(float(b) for b in beta_grid)
    history = []
    traces = []
    best_u = -np.inf
    for step in range(1, budget + 1):
        if not available:
            break
        beta, diag = select_next_beta(history, available, strategy, seed + step * 13, q_hi_ref, warmup=warmup)
        available.remove(beta)
        rec = oracle_map[round(beta, 8)]
        row = dict(
            record_type='strategy_step', strategy=strategy, seed=seed, step=step,
            beta1=rec.beta1, beta2=rec.beta2, feasible=rec.feasible,
            utility=rec.utility, best_utility=np.nan, c=rec.c, residual=rec.residual,
            q1=rec.q1, q2=rec.q2, sigma=rec.sigma, coverage=rec.coverage,
            solvent_no=rec.solvent_no, meoh_pct=rec.meoh_pct, etoh_pct=rec.etoh_pct, h2o_pct=rec.h2o_pct,
            dmso=rec.dmso, eg=rec.eg, coat_speed=rec.coat_speed, coat_temp=rec.coat_temp,
            post_speed=rec.post_speed, post_temp=rec.post_temp,
            rf_trained=diag['rf_trained'], kde_used=diag['kde_used'], gpq_used=diag['gpq_used'], gpu_used=diag['gpu_used'],
            score=diag['score'], p_feas=diag['p_feas'], ei=diag['ei'], kde_score=diag['kde_score'],
            q_score=diag['q_score'], mu_u=diag['mu_u'], sd_u=diag['sd_u'],
        )
        if rec.feasible == 1 and np.isfinite(rec.utility):
            best_u = max(best_u, rec.utility)
        row['best_utility'] = best_u if np.isfinite(best_u) else np.nan
        history.append(row)
        traces.append(row)
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(
                f'{strategy} seed={seed} beta={beta:+.2f} bestU={best_u:.4f}'
                if np.isfinite(best_u) else f'{strategy} seed={seed} beta={beta:+.2f} no feasible solution'
            )
        if verbose:
            tqdm.write(
                f'[PFS {strategy:>12} | seed={seed} | round={step:02d}] '
                f'beta=({beta:+.3f},{1-beta:+.3f}) '
                f'feasible={rec.feasible} c={rec.c:.5f} '
                f'residual={rec.residual:.2e} '
                f'sigma={rec.sigma:.1f} S/cm coverage={rec.coverage:.1f}% '
                f'U={rec.utility:.4f} bestU={best_u:.4f} '
                f'GP_U={diag["gpu_used"]} GP_Q={diag["gpq_used"]} '
                f'RF={diag["rf_trained"]} KDE={diag["kde_used"]}'
            )
    return traces


# -----------------------------------------------------------------------------
# PLOTS / GIF
# -----------------------------------------------------------------------------
def save_round_gif(path, candidate_q, pareto_mask, phi, round_df, oracle_map, beta_grid, dpi=600):
    frames = []
    # Fail loudly instead of silently skipping an empty animation.
    if len(round_df) == 0:
        raise RuntimeError('Cannot create GIF: no rounds selected; check seed filtering.')
    for _, row in round_df.iterrows():
        beta = float(row['beta1'])
        rec = oracle_map[round(beta, 8)]
        ref = phi @ np.array([rec.beta1, rec.beta2])
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        ax.scatter(candidate_q[:, 0], candidate_q[:, 1], s=8, alpha=0.35, label='Candidate Q(x)')
        ax.scatter(candidate_q[pareto_mask, 0], candidate_q[pareto_mask, 1], s=12, color='tab:green', label='Nondominated')
        ax.plot(phi[0, :], phi[1, :], '-o', color='tab:red', lw=1.2, ms=3.2, label='CHIM')
        ax.scatter([ref[0]], [ref[1]], s=35, color='tab:orange', label='Phi beta')
        if rec.feasible == 1:
            qsol = np.array([rec.q1, rec.q2])
            ray = np.vstack([ref, qsol])
            ax.plot(ray[:, 0], ray[:, 1], color='black', lw=1.2, label='c*n (exact equality)')
            ax.scatter([rec.q1], [rec.q2], marker='*', s=95, color='gold', edgecolor='black', linewidth=0.4, label='NBI solution')
        ax.set_title(f'Round {int(row.step)}: beta=({rec.beta1:.3f}, {rec.beta2:.3f})\n'
                     f'c={rec.c:.4f}, residual={rec.residual:.1e}', fontsize=9)
        ax.set_xlabel('q1: conductivity loss', fontsize=8)
        ax.set_ylabel('q2: coverage loss', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.8, loc='upper left', frameon=True)
        fig.tight_layout()
        tmp = path.parent / f'_gif_frame_{int(row.step):02d}.png'
        fig.savefig(tmp, dpi=dpi)
        plt.close(fig)
        frames.append(Image.open(tmp).convert('P', palette=Image.ADAPTIVE))

    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=1000,
            loop=0,
            optimize=False,
        )
    # Delete temporary PNG frames; keep only the four requested outputs.
    for row in round_df.itertuples():
        (path.parent / f'_gif_frame_{int(row.step):02d}.png').unlink(missing_ok=True)
    # Validate the actual GIF rather than claiming success on an empty animation.
    with Image.open(path) as check:
        if getattr(check, 'n_frames', 1) != len(round_df):
            raise RuntimeError(f'GIF has {check.n_frames} frames but expected {len(round_df)}')


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run(args):
    root = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean, dropped = load_clean_data(csv_path)
    bounds = build_control_bounds(clean, expand_min=0.90, expand_max=1.10)
    transformer, gp_models, log_sigma_bounds = train_property_surrogates(clean)

    candidate_df = sample_candidate_cloud(bounds, args.n_candidates, args.seed)
    candidate_q, candidate_sigma, candidate_coverage, _, _ = predict_q(
        candidate_df, transformer, gp_models, log_sigma_bounds
    )
    candidate_df['pred_sigma'] = candidate_sigma
    candidate_df['pred_coverage'] = candidate_coverage
    candidate_df['q1'] = candidate_q[:, 0]
    candidate_df['q2'] = candidate_q[:, 1]
    candidate_df['utility'] = utility_from_predicted_properties(
        candidate_sigma, candidate_coverage, args.sigma_min, args.coverage_min
    )
    candidate_df['record_type'] = 'candidate_cloud'
    candidate_df['pareto_nondominated'] = compute_pareto_mask(candidate_q).astype(int)

    q_hi_ref = np.maximum(candidate_q.max(axis=0), 1e-9)
    phi, n, anchor_a, anchor_b = compute_chim_and_normal(candidate_q)

    solve_beta = build_oracle_solver(
        clean=clean,
        bounds=bounds,
        transformer=transformer,
        gp_models=gp_models,
        log_sigma_bounds=log_sigma_bounds,
        phi=phi,
        n=n,
        sigma_min=args.sigma_min,
        coverage_min=args.coverage_min,
        eq_tol=args.eq_tol,
        start_rows_per_solvent=args.starts_per_solvent,
        maxiter=args.maxiter,
    )

    beta_grid = np.round(np.linspace(args.beta_min, args.beta_max, args.beta_steps), 8)
    oracle_map = {}
    oracle_rows = []
    tqdm.write(f'[DATA] {len(clean)} valid experiments; dropped={dropped}')
    tqdm.write(f'[GEOMETRY] Phi={np.array2string(phi, precision=4)}; n={np.array2string(n, precision=4)}')
    tqdm.write(f'[CONSTRAINTS] sigma >= {args.sigma_min:g} S/cm; coverage >= {args.coverage_min:g}%')

    # V7 NBI CACHE: hash data bytes + all geometry/model/solver settings.
    # Cache is stored OUTSIDE results_v7 so the results folder contains exactly 5 deliverables.
    cache_root = root / '.pfs_v7_cache'
    cache_root.mkdir(exist_ok=True)
    cache_basis = {
        'version': 7, 'input_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'seed': args.seed, 'n_candidates': args.n_candidates,
        'beta_min': args.beta_min, 'beta_max': args.beta_max,
        'beta_steps': args.beta_steps, 'sigma_min': args.sigma_min,
        'coverage_min': args.coverage_min, 'eq_tol': args.eq_tol,
        'starts_per_solvent': args.starts_per_solvent, 'maxiter': args.maxiter,
        'phi': np.round(phi, 12).tolist(), 'normal': np.round(n, 12).tolist(),
    }
    cache_key = hashlib.sha256(json.dumps(cache_basis, sort_keys=True).encode()).hexdigest()[:20]
    cache_file = cache_root / f'oracle_{cache_key}.json'
    cache_used = False
    entries = {}
    if cache_file.exists() and not args.recompute:
        try:
            entries = json.loads(cache_file.read_text(encoding='utf-8'))
            if set(entries) == {f'{b:.8f}' for b in beta_grid}:
                for beta in beta_grid:
                    rec = OracleRecord(**entries[f'{beta:.8f}'])
                    oracle_map[round(float(beta), 8)] = rec
                    oracle_rows.append({'record_type': 'oracle_beta', 'strategy': 'oracle',
                                        'seed': np.nan, 'step': np.nan, **rec.__dict__})
                cache_used = True
                tqdm.write(f'[CACHE] Loaded {len(oracle_map)} NBI beta results: {cache_file.name}')
        except Exception as exc:
            tqdm.write(f'[CACHE] Invalid cache, recomputing: {exc}')
            oracle_map.clear(); oracle_rows.clear(); entries = {}
    if not cache_used:
        tqdm.write(f'[NBI] Solving {len(beta_grid)} beta formulations (may take several minutes)...')
        if args.recompute:
            entries = {}
        with tqdm(beta_grid, desc='Stage 1/3 | NBI oracle', unit='beta', dynamic_ncols=True) as bar:
            for i, beta in enumerate(bar, 1):
                key = f'{beta:.8f}'
                if key in entries:
                    rec = OracleRecord(**entries[key])
                else:
                    rec = solve_beta(float(beta))
                oracle_map[round(float(beta), 8)] = rec
                oracle_rows.append({'record_type': 'oracle_beta', 'strategy': 'oracle',
                                    'seed': np.nan, 'step': np.nan, **rec.__dict__})
                entries[f'{beta:.8f}'] = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v,np.integer) else v)
                                           for k,v in rec.__dict__.items()}
                # Save incrementally after every beta: unexpected interruption loses at most one beta.
                cache_file.write_text(json.dumps(entries, indent=2, allow_nan=True), encoding='utf-8')
                bar.set_postfix_str(f'beta={beta:+.2f} feas={rec.feasible} c={rec.c:.4f} U={rec.utility:.3f}')
                if not args.quiet:
                    tqdm.write(f'[NBI {i:02d}/{len(beta_grid)}] beta={beta:+.3f} '
                               f'feasible={rec.feasible} c={rec.c:+.5f} residual={rec.residual:.2e} '
                               f'sigma={rec.sigma:.1f} coverage={rec.coverage:.1f}% U={rec.utility:.4f}')
        tqdm.write(f'[CACHE] Saved complete NBI oracle: {cache_file.name}')

    strategies = ['full', 'no_gp_q', 'no_rf_kde', 'random_beta']
    strategy_names = {
        'full': 'Full PFS',
        'no_gp_q': 'No GP_Q',
        'no_rf_kde': 'No RF/KDE',
        'random_beta': 'Random beta',
    }

    strategy_rows = []
    total_steps = len(strategies)*args.seeds*min(args.budget,len(beta_grid))
    with tqdm(total=total_steps, desc='Stage 2/3 | PFS ablations', unit='round', dynamic_ncols=True) as bar:
        for strategy in strategies:
            for seed in range(args.seeds):
                actual_seed = args.seed + 1000 * seed
                shared_warmup = np.random.default_rng(actual_seed + 709).choice(
                    beta_grid, size=min(args.warmup, len(beta_grid)), replace=False
                ).tolist()
                traces = run_strategy(
                    strategy, beta_grid, oracle_map, args.budget,
                    actual_seed, q_hi_ref, warmup=shared_warmup,
                    progress=bar, verbose=not args.quiet
                )
                strategy_rows.extend(traces)
    oracle_df = pd.DataFrame(oracle_rows)
    steps_df = pd.DataFrame(strategy_rows)

    # Aggregation for ablation curves.
    curve = (steps_df.groupby(['strategy', 'step'])['best_utility']
             .agg(['mean', 'std']).reset_index())

    tqdm.write('[EXPORT] Stage 3/3: rendering 600 dpi dashboard, CSV, report, and GIF...')
    # Dashboard.
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0))
    axa, axb, axc, axd, axe, axf = axes.ravel()

    pareto_mask = candidate_df['pareto_nondominated'].to_numpy(bool)
    # (a)
    axa.scatter(candidate_q[:, 0], candidate_q[:, 1], s=8, alpha=0.30, label='Candidate GP means')
    feas_or = oracle_df.loc[oracle_df['feasible'] == 1]
    axa.scatter(feas_or['q1'], feas_or['q2'], s=72, marker='*', linewidths=0.5, edgecolors='tab:orange', facecolors='none', label='NBI feasible')
    axa.plot(phi[0, :], phi[1, :], '-o', lw=1.2, ms=3.2, label='CHIM')
    axa.set_title('(a) Objective geometry', loc='left', fontweight='bold')
    axa.set_xlabel('q1: conductivity loss')
    axa.set_ylabel('q2: coverage loss')
    axa.legend(fontsize=7)

    # (b)
    axb.axhline(0.0, color='0.4', lw=0.8)
    axb.plot(oracle_df['beta1'], oracle_df['c'], '-o', ms=3)
    axb.set_title('(b) NBI c vs beta', loc='left', fontweight='bold')
    axb.set_xlabel('beta1')
    axb.set_ylabel('c (feasible solutions)')

    # (c)
    residual_plot = np.clip(np.abs(oracle_df['residual'].to_numpy(float)), 1e-12, None)
    axc.axhline(args.eq_tol, color='tab:red', linestyle=':', lw=1.0, label='tolerance')
    axc.scatter(oracle_df['beta1'], residual_plot, s=16, label='equality residual')
    axc.set_yscale('log')
    axc.set_title('(c) Numerical feasibility', loc='left', fontweight='bold')
    axc.set_xlabel('beta1')
    axc.set_ylabel('equality residual (log scale)')
    axc.legend(fontsize=7)

    # (d)
    color_map = {
        'full': 'tab:blue',
        'no_gp_q': 'tab:orange',
        'no_rf_kde': 'tab:green',
        'random_beta': 'tab:red',
    }
    marker_map = {
        'full': 'o',
        'no_gp_q': '^',
        'no_rf_kde': 'x',
        'random_beta': 's',
    }
    for strategy in strategies:
        sub = curve.loc[curve['strategy'] == strategy]
        marker = marker_map[strategy]
        axd.plot(sub['step'], sub['mean'], '-', marker=marker, ms=5, mew=1.0, alpha=0.82, color=color_map[strategy], label=strategy)
        lo = sub['mean'] - sub['std'].fillna(0)
        hi = sub['mean'] + sub['std'].fillna(0)
        axd.fill_between(sub['step'], lo, hi, color=color_map[strategy], alpha=0.10)
    axd.set_title('(d) Ablation: search budget', loc='left', fontweight='bold')
    axd.set_xlabel('NBI beta queries')
    axd.set_ylabel('Best predicted utility')
    axd.legend(fontsize=7)

    # (e) Show the first ACTUAL seed (args.seed), not literal zero.
    display_seed = args.seed
    for strategy in strategies:
        sub = steps_df.loc[(steps_df['strategy'] == strategy) & (steps_df['seed'] == display_seed)]
        marker = marker_map[strategy]
        axe.plot(sub['step'], sub['beta1'], '-', marker=marker, ms=5, mew=1.0, alpha=0.82, color=color_map[strategy], label=strategy)
    axe.set_title('(e) Beta search paths', loc='left', fontweight='bold')
    axe.set_xlabel('Outer round')
    axe.set_ylabel(f'beta1 (seed {display_seed})')
    axe.legend(fontsize=7)

    # (f)
    axf.scatter(candidate_df['pred_sigma'], candidate_df['pred_coverage'], s=8, alpha=0.35, label='Predicted candidate Q')
    axf.scatter(feas_or['sigma'], feas_or['coverage'], s=78, marker='*', linewidths=0.5, edgecolors='tab:red', facecolors='none', label='NBI solutions')
    axf.set_xscale('log')
    axf.set_title('(f) Predicted property space', loc='left', fontweight='bold')
    axf.set_xlabel('Predicted conductivity (S/cm)')
    axf.set_ylabel('Predicted coverage (%)')
    axf.legend(fontsize=7)

    fig.suptitle('Polybot PFS v8: GP_U + GP_Q + RF + KDE with thresholded utility and constrained exact NBI', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_dir / 'dashboard.png', dpi=600)
    plt.close(fig)

    # GIF from full strategy, using the same actual seed as Panel (e).
    gif_rounds = steps_df.loc[(steps_df['strategy'] == 'full') & (steps_df['seed'] == display_seed)].copy()
    save_round_gif(out_dir / 'nbi_process.gif', candidate_q, pareto_mask, phi, gif_rounds, oracle_map, beta_grid, dpi=600)

    # all_results.csv: candidate cloud + oracle beta table + strategy steps + summary rows.
    candidate_export = candidate_df.copy()
    candidate_export = candidate_export.rename(columns={
        'DMSO concentration (Vol%)': 'dmso',
        'EG concentration (Vol%)': 'eg',
        'Coating speed (mm/sec)': 'coat_speed',
        'Coating temperature (Celsius)': 'coat_temp',
        'Post-processing solvent No.': 'solvent_no',
        'Post coating speed (mm/sec)': 'post_speed',
        'Post coating temperature (Celsius)': 'post_temp',
        'pred_sigma': 'sigma',
        'pred_coverage': 'coverage',
    })
    candidate_export['meoh_pct'] = candidate_export['solvent_no'].map(lambda x: SOLVENT_MAP[int(x)][0])
    candidate_export['etoh_pct'] = candidate_export['solvent_no'].map(lambda x: SOLVENT_MAP[int(x)][1])
    candidate_export['h2o_pct'] = candidate_export['solvent_no'].map(lambda x: SOLVENT_MAP[int(x)][2])
    for col in ['strategy', 'seed', 'step', 'best_utility', 'feasible', 'c', 'residual',
                'rf_trained', 'kde_used', 'gpq_used', 'gpu_used', 'score', 'p_feas',
                'ei', 'kde_score', 'q_score', 'mu_u', 'sd_u']:
        if col not in candidate_export:
            candidate_export[col] = np.nan

    summary_rows = []
    summary = (steps_df.groupby(['strategy', 'seed'])['best_utility']
               .max()
               .groupby('strategy')
               .agg(['mean', 'std'])
               .reset_index())
    for _, r in summary.iterrows():
        summary_rows.append({
            'record_type': 'summary',
            'strategy': r['strategy'],
            'seed': np.nan, 'step': np.nan,
            'best_utility': r['mean'],
            'score': r['std'],
        })
    summary_df = pd.DataFrame(summary_rows)

    all_results = pd.concat([
        candidate_export,
        oracle_df,
        steps_df,
        summary_df,
    ], ignore_index=True, sort=False)
    all_results.to_csv(out_dir / 'all_results.csv', index=False)

    # REPORT.md
    feasible_count = int(oracle_df['feasible'].sum())
    rf_trainable = int(len(np.unique(oracle_df['rf_target'])) == 2)
    best_oracle = oracle_df.loc[oracle_df['utility'].idxmax()] if feasible_count > 0 else None
    report = []
    report.append('# Polybot PFS V8 report')
    report.append('')
    report.append('## 1. Data and constraints')
    report.append(f'- Valid rows after cleaning: **{len(clean)}**')
    report.append(f'- Dropped rows: **{dropped}**')
    report.append(f'- Conductivity constraint: **sigma >= {args.sigma_min:.3f} S/cm**')
    report.append(f'- Coverage constraint: **coverage >= {args.coverage_min:.3f}%**')
    report.append('- Continuous-variable bounds are set to **90% of observed minimum** and **110% of observed maximum**.')
    report.append('- Post-processing solvent is discrete and follows the supplied 8-code table.')
    report.append('')
    report.append('### Post-processing solvent mapping')
    report.append('| No. | MeOH | EtOH | H2O |')
    report.append('|---:|---:|---:|---:|')
    for k, v in SOLVENT_MAP.items():
        report.append(f'| {k} | {v[0]} | {v[1]} | {v[2]} |')
    report.append('')
    report.append('## 2. Core formulas implemented in code')
    report.append('1. **QoI / loss map**')
    report.append('   - `predict_q(...)` defines')
    report.append('     - `q1 = (log_sigma_max - log10(sigma)) / (log_sigma_max - log_sigma_min)`')
    report.append('     - `q2 = 1 - coverage_fraction`')
    report.append('2. **Threshold-centered utility**')
    report.append('   - `utility_from_predicted_properties(...)` defines')
    report.append('     - `U = 1.5 * ReLU((sigma - sigma_min)/sigma_min) + 1.0 * ReLU((coverage - coverage_min)/coverage_min)`')
    report.append('3. **Exact NBI subproblem**')
    report.append('   - in `build_oracle_solver(...)`, SLSQP solves')
    report.append('     - `max c`')
    report.append('     - subject to `Q(x) - Phi*beta - c*n = 0`')
    report.append('     - plus `sigma >= sigma_min` and `coverage >= coverage_min`')
    report.append('4. **Outer-loop models**')
    report.append('   - `select_next_beta(...)` contains')
    report.append('     - `GP_U`: beta -> utility')
    report.append('     - `GP_Q`: beta -> (q1, q2) with two GPs')
    report.append('     - `RF`: beta -> feasibility label')
    report.append('     - `KDE`: density over feasible beta history')
    report.append('')
    report.append('## 3. Oracle / feasibility summary')
    report.append(f'- Beta grid size: **{len(beta_grid)}**')
    report.append(f'- Feasible oracle solutions: **{feasible_count}/{len(beta_grid)}**')
    report.append(f'- Equality tolerance: **{args.eq_tol:.1e}**')
    report.append(f'- RF trainable from oracle labels (both classes present): **{bool(rf_trainable)}**')
    report.append('')
    report.append('## 4. Ablation summary (best utility across budget)')
    report.append('| Strategy | Mean final best utility | Std |')
    report.append('|---|---:|---:|')
    for _, r in summary.iterrows():
        report.append(f"| {strategy_names.get(r['strategy'], r['strategy'])} | {r['mean']:.6f} | {0.0 if pd.isna(r['std']) else r['std']:.6f} |")
    report.append('')
    if best_oracle is not None:
        report.append('## 5. Highest-utility feasible oracle solution')
        report.append(f'- beta = (**{best_oracle.beta1:.3f}**, **{best_oracle.beta2:.3f}**)')
        report.append(f'- utility = **{best_oracle.utility:.6f}**')
        report.append(f'- c = **{best_oracle.c:.6f}**')
        report.append(f'- sigma = **{best_oracle.sigma:.3f} S/cm**')
        report.append(f'- coverage = **{best_oracle.coverage:.3f}%**')
        report.append(f'- solvent no. = **{int(best_oracle.solvent_no)}** (MeOH/EtOH/H2O = {int(best_oracle.meoh_pct)}/{int(best_oracle.etoh_pct)}/{int(best_oracle.h2o_pct)})')
        report.append(f'- DMSO = **{best_oracle.dmso:.3f} vol%**, EG = **{best_oracle.eg:.3f} vol%**')
        report.append(f'- coating speed/temp = **{best_oracle.coat_speed:.3f} mm/s**, **{best_oracle.coat_temp:.3f} C**')
        report.append(f'- post speed/temp = **{best_oracle.post_speed:.3f} mm/s**, **{best_oracle.post_temp:.3f} C**')
        report.append('')
    report.append('## 6. Limitations')
    report.append('- This is a fixed-surrogate offline study; no new wet-lab measurements are added online.')
    report.append('- SLSQP satisfying the NBI equality numerically does not certify a global optimum for `c`.')
    report.append('- When all or most beta points are feasible, RF feasibility classification may carry limited extra value.')
    report.append('- Utility is threshold-centered and dimensionless; it is suitable for ranking under the specified thresholds, but is not a physical property by itself.')
    report.append('')
    (out_dir / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')

    # V7: one prospective recommendation JSON, based on FULL PFS actually queried,
    # rather than silently peeking at the best solution in the entire oracle grid.
    full = steps_df.loc[(steps_df['strategy'] == 'full') & (steps_df['seed'] == args.seed)]
    good_full = full.loc[(full['feasible'] == 1) & full['utility'].notna()].copy()
    if len(good_full):
        best_step = good_full.loc[good_full['utility'].idxmax()]
        recommended = oracle_map[round(float(best_step['beta1']), 8)]
        experimental_controls = {
            'DMSO concentration (Vol%)': float(recommended.dmso),
            'EG concentration (Vol%)': float(recommended.eg),
            'Coating speed (mm/sec)': float(recommended.coat_speed),
            'Coating temperature (Celsius)': float(recommended.coat_temp),
            'Post-processing solvent No.': int(recommended.solvent_no),
            'Post-processing solvent MeOH (vol%)': float(recommended.meoh_pct),
            'Post-processing solvent EtOH (vol%)': float(recommended.etoh_pct),
            'Post-processing solvent H2O (vol%)': float(recommended.h2o_pct),
            'Post coating speed (mm/sec)': float(recommended.post_speed),
            'Post coating temperature (Celsius)': float(recommended.post_temp),
        }
        suggestion = {
            'status': 'predicted_not_experimentally_validated',
            'recommendation_basis': 'best NBI solution actually queried by full PFS, first seed',
            'selected_beta': [float(recommended.beta1), float(recommended.beta2)],
            'selected_round': int(best_step['step']),
            'experimental_conditions': experimental_controls,
            'predictions': {
                'conductivity_S_per_cm': float(recommended.sigma),
                'coverage_percent': float(recommended.coverage),
                'utility': float(recommended.utility),
                'nbi_c': float(recommended.c),
                'nbi_equality_residual': float(recommended.residual),
                'q1_conductivity_loss': float(recommended.q1),
                'q2_coverage_loss': float(recommended.q2),
            },
            'minimum_thresholds': {'conductivity_S_per_cm': float(args.sigma_min),
                                   'coverage_percent': float(args.coverage_min)},
            'limitations': ['Fixed property GP predictions, not lab measurements',
                            'SLSQP local optimum only',
                            'Check chemical feasibility and experimental uncertainty before execution'],
        }
    else:
        suggestion = {'status': 'no_feasible_recommendation',
                      'reason': 'No feasible beta was queried by full PFS in the first seed.'}
    (out_dir / 'next_experiment.json').write_text(
        json.dumps(suggestion, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8'
    )
    report_extra = ('\n## 7. V7 diagnostics and recommendation\n'
                    f'- Cached oracle reused: **{cache_used}**\n'
                    f'- Shared random initialization per seed: **{args.warmup} beta queries**\n'
                    '- Recommendation is saved in **next_experiment.json**, based on the queried Full PFS path.\n'
                    '- GP_U / GP_Q / RF / KDE can be inactive when insufficient diverse training labels exist.\n')
    with (out_dir / 'REPORT.md').open('a', encoding='utf-8') as f:
        f.write(report_extra)
    if not (out_dir / 'nbi_process.gif').is_file():
        raise RuntimeError('GIF output missing after render')
    with Image.open(out_dir / 'nbi_process.gif') as gif:
        expected = len(gif_rounds)
        if gif.n_frames != expected:
            raise RuntimeError(f'GIF frames {gif.n_frames} != expected {expected}')
    tqdm.write(f'[EXPORT] Verified GIF frames={len(gif_rounds)} and Panel (e) seed={display_seed}')
    print(f'Wrote: {out_dir / "next_experiment.json"}')
    print(f'Wrote: {out_dir / "dashboard.png"}')
    print(f'Wrote: {out_dir / "all_results.csv"}')
    print(f'Wrote: {out_dir / "REPORT.md"}')
    print(f'Wrote: {out_dir / "nbi_process.gif"}')


def build_parser():
    p = argparse.ArgumentParser(description='Polybot PFS V6 with constrained exact NBI and ablations.')
    p.add_argument('--csv', default='polybot-PEDOT_experiment.csv')
    p.add_argument('--quiet', action='store_true', help='Show tqdm bars without per-round log lines')
    p.add_argument('--out', default='results_v7')
    p.add_argument('--recompute', action='store_true', help='Ignore saved NBI cache')
    p.add_argument('--warmup', type=int, default=3, help='Shared random initial beta evaluations per seed')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-candidates', type=int, default=350)
    p.add_argument('--beta-min', type=float, default=-0.5)
    p.add_argument('--beta-max', type=float, default=1.5)
    p.add_argument('--beta-steps', type=int, default=21)
    p.add_argument('--budget', type=int, default=12)
    p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--sigma-min', type=float, default=665.0)
    p.add_argument('--coverage-min', type=float, default=72.0)
    p.add_argument('--eq-tol', type=float, default=1e-4)
    p.add_argument('--starts-per-solvent', type=int, default=2)
    p.add_argument('--maxiter', type=int, default=120)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.budget > args.beta_steps:
        raise SystemExit('--budget must be <= --beta-steps')
    if args.warmup > args.budget:
        raise SystemExit('--warmup must be <= --budget')
    run(args)
