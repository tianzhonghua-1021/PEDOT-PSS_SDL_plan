#!/usr/bin/env python3
"""Polybot PEDOT:PSS: transparent finite-pool, two-objective PFS/NBI diagnostics.

Run from folder containing polybot-PEDOT_experiment.csv:
    python run_pfs_diagnostics.py

Only requires numpy, pandas, scipy, scikit-learn, matplotlib, pillow (for GIF).
This is a fixed-surrogate offline demonstration, not prospective lab optimization.
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

FEATURES = [
    'DMSO concentration (Vol%)', 'EG concentration (Vol%)',
    'Coating speed (mm/sec)', 'Coating temperature (Celsius)',
    'Post-processing solvent No.', 'Post coating speed (mm/sec)',
    'Post coating temperature (Celsius)',
]
CONDUCTIVITY = 'Average conductivity (S/cm)'
COVERAGE = 'Average coverage (%)'


def expected_improvement(mu, sd, best, xi=0.01):
    sd = np.asarray(sd, dtype=float)
    imp = np.asarray(mu) - best - xi
    z = imp / np.maximum(sd, 1e-12)
    result = imp * norm.cdf(z) + sd * norm.pdf(z)
    return np.where(sd > 1e-12, result, np.maximum(imp, 0))


def load_data(path):
    df = pd.read_csv(path)
    for name in FEATURES + [CONDUCTIVITY, COVERAGE]:
        if name not in df.columns:
            raise ValueError(f'Missing CSV column: {name}')
        df[name] = pd.to_numeric(df[name], errors='coerce')
    mask = (df[FEATURES].notna().all(axis=1)
            & df[CONDUCTIVITY].gt(0)
            & df[COVERAGE].between(0, 100))
    clean = df.loc[mask].copy().reset_index(drop=True)
    if len(clean) < 12:
        raise ValueError('Fewer than 12 valid records after cleaning')
    return clean, int((~mask).sum())


def train_models(df):
    # Explicitly separate solvent ID (categorical) from physical numeric controls.
    trans = ColumnTransformer([
        ('numbers', StandardScaler(), [0, 1, 2, 3, 5, 6]),
        ('solvent', OneHotEncoder(handle_unknown='ignore'), [4]),
    ])
    z = trans.fit_transform(df[FEATURES])
    if hasattr(z, 'toarray'):
        z = z.toarray()
    # Fixed kernel hyperparameters, for stable and reasonably fast diagnostic run.
    y1 = np.log10(df[CONDUCTIVITY].to_numpy(dtype=float))
    y2 = df[COVERAGE].to_numpy(dtype=float) / 100.0
    models = []
    for y in (y1, y2):
        kernel = (ConstantKernel(1.0, constant_value_bounds='fixed')
                  * Matern(length_scale=np.ones(z.shape[1]),
                           length_scale_bounds='fixed', nu=2.5)
                  + WhiteKernel(noise_level=0.05, noise_level_bounds='fixed'))
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                      normalize_y=True, alpha=1e-5)
        gp.fit(z, y)
        models.append(gp)
    bounds = (float(y1.min()), float(y1.max()))
    return trans, models, bounds


def make_candidates(df, count, seed):
    rng = np.random.default_rng(seed)
    old = df[FEATURES].to_numpy(dtype=float)
    idx = rng.integers(0, len(old), size=count)
    arr = old[idx].copy()
    # Only two additive concentrations interpolated; other variables use observed settings.
    for j in (0, 1):
        arr[:, j] = rng.uniform(old[:, j].min(), old[:, j].max(), size=count)
        arr[:, j] = np.round(arr[:, j], 3)
    existing_keys = {tuple(np.round(x, 5)) for x in old}
    unique = []
    seen = set()
    for row in arr:
        key = tuple(np.round(row, 5))
        if key not in existing_keys and key not in seen:
            unique.append(row)
            seen.add(key)
    if len(unique) < 15:
        raise RuntimeError('Not enough distinct unmeasured candidates')
    return np.array(unique)


def predict_objectives(arr, transformer, models, log_bounds):
    z = transformer.transform(pd.DataFrame(arr, columns=FEATURES))
    if hasattr(z, 'toarray'):
        z = z.toarray()
    log_cond, std_log = models[0].predict(z, return_std=True)
    cover, std_cover = models[1].predict(z, return_std=True)
    low, high = log_bounds
    width = max(high-low, 1e-8)
    # Minimize BOTH. No clipping: don't artificially flatten predicted variation.
    q = np.column_stack(((high-log_cond)/width, 1-cover))
    return q, log_cond, cover, std_log, std_cover


def compute_nbi_geometry(q):
    a = int(np.argmin(q[:, 0]))
    b = int(np.argmin(q[:, 1]))
    phi = np.column_stack((q[a], q[b]))
    d = phi[:, 1] - phi[:, 0]
    if np.linalg.norm(d) < 1e-10:
        raise RuntimeError('Two single-objective anchors coincide: degenerate CHIM.')
    normal = np.array([d[1], -d[0]], dtype=float)
    normal /= np.linalg.norm(normal)
    # In minimization geometry choose the southwest-pointing normal when possible.
    if np.sum(normal) > 0:
        normal *= -1
    return phi, normal, a, b


def choose_for_beta(beta1, q, phi, normal, tol):
    beta_vec = np.array([beta1, 1.0-beta1])
    ref = phi @ beta_vec
    diff = q-ref
    c = diff @ normal
    residual = np.linalg.norm(diff-c[:, None]*normal[None, :], axis=1)
    valid = np.where((residual <= tol) & (c >= 0))[0]
    # Explicitly label fallback; no false claim of NBI feasibility.
    if len(valid):
        index = int(valid[np.argmax(c[valid])])
        status = 'within_tolerance'
    else:
        index = int(np.argmin(residual + 0.10*np.maximum(-c, 0)))
        status = 'NO_FEASIBLE_RAY_fallback'
    return dict(beta=np.array(beta_vec), ref=ref, diff=diff, c=c,
                residual=residual, index=index, status=status,
                n_valid=len(valid))


def u_of_q(q, qlow, qhigh):
    # Fixed illustrative utility for the outer beta-space search.
    scores = np.clip((qhigh-q)/(qhigh-qlow+1e-12), 0, 1)
    return float(np.sqrt(scores[0]*scores[1]))


def outer_beta_bo(beta_values, iterations, q, phi, nvec, tol, seed):
    rng = np.random.default_rng(seed+101)
    qlo = np.min(q, axis=0)
    qhi = np.max(q, axis=0)
    cache = {}
    trace = []
    evaluated = set()
    for t in range(min(iterations, len(beta_values))):
        available = [float(b) for b in beta_values if round(float(b), 10) not in evaluated]
        if t == 0:
            b, gp_mean, gp_std, ei = min(available, key=lambda x: abs(x-.5)), np.nan, np.nan, np.nan
        elif t == 1:
            b, gp_mean, gp_std, ei = min(available), np.nan, np.nan, np.nan
        elif t == 2:
            b, gp_mean, gp_std, ei = max(available), np.nan, np.nan, np.nan
        else:
            xx = np.array([v['beta1'] for v in trace]).reshape(-1, 1)
            yy = np.array([v['utility'] for v in trace])
            kernel = (ConstantKernel(1.0, constant_value_bounds='fixed')
                      * Matern(length_scale=0.25,
                               length_scale_bounds='fixed', nu=2.5)
                      + WhiteKernel(noise_level=.002, noise_level_bounds='fixed'))
            g = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                         normalize_y=True, alpha=1e-6)
            g.fit(xx, yy)
            mu, sd = g.predict(np.array(available).reshape(-1, 1), return_std=True)
            ei_values = expected_improvement(mu, sd, yy.max())
            j = int(np.argmax(ei_values))
            b, gp_mean, gp_std, ei = available[j], float(mu[j]), float(sd[j]), float(ei_values[j])
        info = choose_for_beta(b, q, phi, nvec, tol)
        i = info['index']
        utility = u_of_q(q[i], qlo, qhi)
        trace.append(dict(round=t+1, beta1=b, beta2=1-b,
                          ref_q1=float(info['ref'][0]), ref_q2=float(info['ref'][1]),
                          phi11=float(phi[0, 0]), phi12=float(phi[0, 1]),
                          phi21=float(phi[1, 0]), phi22=float(phi[1, 1]),
                          n1=float(nvec[0]), n2=float(nvec[1]),
                          dot_normal_chim=float(nvec @ (phi[:,1]-phi[:,0])),
                          candidate_id=i, q1=float(q[i, 0]), q2=float(q[i, 1]),
                          c=float(info['c'][i]), residual=float(info['residual'][i]),
                          n_feasible_candidates=info['n_valid'], status=info['status'],
                          utility=utility, gp_u_mu=gp_mean,
                          gp_u_std=gp_std, acquisition_ei=ei))
        cache[t+1] = info
        evaluated.add(round(float(b), 10))
    return pd.DataFrame(trace), cache


def create_plots(out, trace, diagnostics, q, phi, nvec, candidate_arr,
                 predicted_log, predicted_cov, seed):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    axs = axs.ravel()
    axs[0].scatter(q[:,0], q[:,1], s=9, alpha=.28, label='Candidate Q(x)')
    axs[0].plot(phi[0], phi[1], 'o-', color='firebrick', label='CHIM')
    cmap = plt.cm.viridis(np.linspace(0,1,len(trace)))
    for k, row in trace.iterrows():
        ref = np.array([row.ref_q1, row.ref_q2])
        axs[0].scatter(*ref, s=22, color=cmap[k])
        axs[0].plot([ref[0], ref[0]+row.c*nvec[0]],
                    [ref[1], ref[1]+row.c*nvec[1]], color=cmap[k], alpha=.5)
    last = trace.iloc[-1]
    axs[0].scatter(last.q1,last.q2,marker='*',s=180,c='gold',edgecolor='black',label='Last selected')
    axs[0].set(xlabel='q1: conductivity loss', ylabel='q2: coverage loss',
               title='A  |  Objective space, CHIM and rays')
    axs[0].legend(fontsize=7)
    axs[1].plot(trace['round'],trace.beta1,'o-',label='beta1')
    axs[1].plot(trace['round'],trace.beta2,'s--',label='beta2')
    axs[1].set(xlabel='Outer BO iteration',ylabel='beta coefficient',title='B  |  PFS beta trajectory',ylim=(-.05,1.05))
    axs[1].legend()
    axs[2].plot(trace['round'],trace.c,'o-',label='projected c')
    axs[2].plot(trace['round'],trace.residual,'s-',label='ray residual')
    axs[2].axhline(0.035,color='gray',ls=':',label='example tolerance=0.035')
    axs[2].set(xlabel='Outer BO iteration',title='C  |  c versus constraint residual')
    axs[2].legend(fontsize=7)
    axs[3].plot(trace['round'],trace.n1,'o-',label='n1')
    axs[3].plot(trace['round'],trace.n2,'o-',label='n2')
    axs[3].plot(trace['round'],trace.dot_normal_chim,'k:',label='n·CHIM direction')
    axs[3].set(xlabel='Outer BO iteration',title='D  |  Fixed unit CHIM normal')
    axs[3].legend(fontsize=7)
    axs[4].plot(trace['round'],trace.utility,'o-',label='observed surrogate utility')
    axs[4].plot(trace['round'],trace.gp_u_mu,'s--',label='GP_U prior-to-query mean')
    axs[4].set(xlabel='Outer BO iteration',title='E  |  Beta → utility / GP_U')
    axs[4].legend(fontsize=7)
    axs[5].scatter(10**predicted_log, predicted_cov*100,
                   c=q[:,0],s=12,cmap='viridis',alpha=.65)
    axs[5].scatter(10**predicted_log[trace.candidate_id.to_numpy(dtype=int)],
                   predicted_cov[trace.candidate_id.to_numpy(dtype=int)]*100,
                   marker='x',color='red',s=42,label='beta-selected')
    axs[5].set(xlabel='Predicted conductivity (S/cm)',ylabel='Predicted coverage (%)',
               title='F  |  Property predictions',xscale='log')
    axs[5].legend(fontsize=7)
    fig.suptitle('Polybot PFS / NBI geometric diagnostics — fixed property GP (offline)',fontsize=14)
    fig.savefig(out/'geometry_panels.png', dpi=170)
    plt.close(fig)
    # Animated frame, same Q(x) across rounds as GP is not retrained.
    try:
        fig, ax = plt.subplots(figsize=(8,7))
        writer = PillowWriter(fps=1.3)
        with writer.saving(fig, str(out/'geometry_iterations.gif'), dpi=105):
            for _, row in trace.iterrows():
                ax.clear()
                ax.scatter(q[:,0], q[:,1], s=12, alpha=.22, c='slateblue')
                ax.plot(phi[0],phi[1],'-o',c='firebrick',label='CHIM')
                p=np.array([row.ref_q1,row.ref_q2])
                line_end=p+row.c*nvec
                ax.scatter(*p,c='darkorange',s=65,label='Phi beta')
                ax.plot([p[0],line_end[0]],[p[1],line_end[1]],color='green',lw=2,label='NBI normal ray')
                ax.scatter(row.q1,row.q2,marker='*',s=190,c='gold',edgecolor='black',label='Chosen candidate')
                ax.set(xlabel='q1: conductivity loss',ylabel='q2: coverage loss',
                       title=f"Round {int(row['round'])} | beta={row.beta1:.3f} | c={row.c:.3f}\n"
                             f"residual={row.residual:.4f} | {row.status}")
                ax.legend(loc='best',fontsize=8)
                writer.grab_frame()
        plt.close(fig)
    except (ImportError, RuntimeError) as exc:
        warnings.warn(f'GIF unavailable: {exc}. PNG and CSV outputs still available.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',default='polybot-PEDOT_experiment.csv')
    parser.add_argument('--out',default='results_diagnostics')
    parser.add_argument('--candidates',type=int,default=650)
    parser.add_argument('--beta-steps',type=int,default=15)
    parser.add_argument('--iterations',type=int,default=9)
    parser.add_argument('--tol',type=float,default=.035)
    parser.add_argument('--seed',type=int,default=42)
    a=parser.parse_args()
    base=Path(__file__).resolve().parent
    input_path=Path(a.csv)
    if not input_path.is_absolute(): input_path=base/input_path
    output_path=Path(a.out)
    if not output_path.is_absolute(): output_path=base/output_path
    output_path.mkdir(parents=True, exist_ok=True)
    df, dropped=load_data(input_path)
    transformer, models, log_bounds=train_models(df)
    candidates=make_candidates(df,a.candidates,a.seed)
    q, logs, cov, sdlog, sdcov=predict_objectives(candidates, transformer, models, log_bounds)
    phi, nvec, ia, ib=compute_nbi_geometry(q)
    beta_grid=np.linspace(0,1,max(4,a.beta_steps))
    trace, infos=outer_beta_bo(beta_grid,a.iterations,q,phi,nvec,a.tol,a.seed)
    for j,f in enumerate(FEATURES):
        trace['selected_'+f]=candidates[trace.candidate_id.to_numpy(dtype=int),j]
    ci=trace.candidate_id.to_numpy(dtype=int)
    trace['pred_conductivity_S_cm']=10**logs[ci]
    trace['pred_coverage_percent']=100*cov[ci]
    trace.to_csv(output_path/'iteration_trace.csv',index=False)
    rows=[]
    for _,r in trace.iterrows():
        info=infos[int(r['round'])]
        for j in range(len(candidates)):
            rows.append((int(r['round']),float(r.beta1),j,float(q[j,0]),float(q[j,1]),
                         float(info['ref'][0]),float(info['ref'][1]),
                         float(info['diff'][j,0]),float(info['diff'][j,1]),
                         float(info['c'][j]),float(info['residual'][j]),
                         bool(info['residual'][j]<=a.tol and info['c'][j]>=0),
                         bool(j==info['index'])))
    cols=['round','beta1','candidate_id','q1','q2','ref_q1','ref_q2',
          'd1','d2','c_proj','ray_residual','within_tolerance','selected']
    pd.DataFrame(rows,columns=cols).to_csv(output_path/'candidate_ray_diagnostics.csv',index=False)
    candidate_table=pd.DataFrame(candidates,columns=FEATURES)
    candidate_table.insert(0,'candidate_id',np.arange(len(candidates)))
    candidate_table['q1']=q[:,0];candidate_table['q2']=q[:,1]
    candidate_table['pred_conductivity_S_cm']=10**logs
    candidate_table['pred_coverage_percent']=100*cov
    candidate_table['log_conductivity_GP_std']=sdlog
    candidate_table['coverage_GP_std_percent']=100*sdcov
    candidate_table.to_csv(output_path/'candidate_pool.csv',index=False)
    create_plots(output_path,trace,infos,q,phi,nvec,candidates,logs,cov,a.seed)
    last=trace.iloc[-1]
    best=trace.loc[trace.utility.idxmax()]
    lines=[
        '# Polybot PFS/NBI geometry diagnostic report','',
        f'- Historical records: {len(df)} (dropped: {dropped})',
        f'- Unmeasured candidate pool: {len(candidates)}',
        f'- Outer BO queries: {len(trace)} out of {len(beta_grid)} beta-grid points',
        f'- Candidate ray residual tolerance: {a.tol}',
        '- Both Q objectives are minimized: q1 is normalized log conductivity loss; q2=1-coverage/100.',
        '- CHIM normal is perpendicular by construction: n dot (anchor2-anchor1) = '
          f'{nvec @ (phi[:,1]-phi[:,0]):.3e}',
        '- **Phi, n and candidate Q(x) are fixed for all rounds.** No new experiments or GP retraining.',
        '- NBI is finite-pool ray matching, NOT continuous equality-constrained NBI.',
        '- If no candidate passes tolerance, a flagged fallback is shown; it is NOT an NBI solution.',
        '- GP_U changes across iterations, but utility labels come from the same property surrogate.',
        '- Solvent No. is categorical ID; identify chemistry before attempting experiments.',
        '- GP uncertainty is not calibrated; recommendations should be validated independently.',
        '', '## NBI geometry',
        f'Phi = {np.array2string(phi,precision=5)}',
        f'n = {np.array2string(nvec,precision=5)}',
        f'anchor candidate IDs = ({ia}, {ib})',
        '', '## Best beta-queried (predicted utility, not measured)',
        f"- Round {int(best['round'])}; beta=({best.beta1:.4f}, {best.beta2:.4f})",
        f'- Candidate ID: {int(best.candidate_id)}; status: {best.status}',
        f'- Predicted conductivity: {best.pred_conductivity_S_cm:.3f} S/cm',
        f'- Predicted coverage: {best.pred_coverage_percent:.3f}%',
        f'- Projected c: {best.c:.5f}; residual: {best.residual:.5f}',
        '', '## Files',
        '- geometry_panels.png: six-panel static summary',
        '- geometry_iterations.gif: each beta query in Q-space (if Pillow available)',
        '- iteration_trace.csv: phi, n, beta, c, Q, utility and GP_U diagnostics per round',
        '- candidate_ray_diagnostics.csv: every candidate projection at every beta',
        '- candidate_pool.csv: experimental recipe and predicted objectives per candidate',
    ]
    (output_path/'REPORT_diagnostics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Finished:',len(df),'historical records;',len(candidates),'candidates;',len(trace),'PFS rounds')
    print('Normal check n dot CHIM:',float(nvec @ (phi[:,1]-phi[:,0])))
    print('NBI feasible rounds:',int((trace.status=='within_tolerance').sum()),'/',len(trace))
    print('Results:',output_path)
    print('Important: Q cloud and n remain fixed until actual new measurements/retraining.')


if __name__=='__main__':
    main()
