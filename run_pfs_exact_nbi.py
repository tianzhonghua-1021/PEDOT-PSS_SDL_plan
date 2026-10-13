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
from scipy.optimize import minimize
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




def run_v4(args):
    base_dir=Path(__file__).resolve().parent
    input_path=Path(args.csv)
    if not input_path.is_absolute(): input_path=base_dir/input_path
    out=Path(args.out)
    if not out.is_absolute(): out=base_dir/out
    out.mkdir(parents=True,exist_ok=True)
    df,dropped=load_data(input_path)
    trans,models,scales=train_models(df)
    candidates=make_candidates(df,args.candidates,args.seed)
    qcloud,logs,covs,_,_=predict_objectives(candidates,trans,models,scales)
    phi,n,ia,ib=compute_nbi_geometry(qcloud)
    # Restrict continuous optimization to additive percentages, with observed valid process templates.
    original=df[FEATURES].to_numpy(dtype=float)
    templates=np.unique(original[:,2:],axis=0)
    rng=np.random.default_rng(args.seed)
    limits=np.array([[original[:,0].min(),original[:,0].max()],
                     [original[:,1].min(),original[:,1].max()]])
    bounds=[tuple(limits[0]),tuple(limits[1]),(args.c_min,args.c_max)]
    qlow=qcloud.min(axis=0);qhigh=qcloud.max(axis=0)
    # Cache GP evaluation of 2D additives for each fixed process, no retraining.
    def q_value(add, templ):
        arr=np.r_[np.asarray(add,dtype=float),templ].reshape(1,-1)
        q,lc,cv,_,_=predict_objectives(arr,trans,models,scales)
        return q[0],float(lc[0]),float(cv[0])
    def utility(q):
        z=np.clip((qhigh-q)/(qhigh-qlow+1e-12),0,1)
        return float(np.sqrt(z[0]*z[1]))
    beta_grid=np.linspace(0,1,args.beta_steps)
    trace=[]; candidates_records=[]; state=[]; used=set()
    for t in range(min(args.iterations,len(beta_grid))):
        available=np.array([b for b in beta_grid if round(float(b),8) not in used])
        if t==0: b=float(available[np.argmin(abs(available-.5))]);mu=sd=ei=np.nan
        elif t==1: b=float(available[0]);mu=sd=ei=np.nan
        elif t==2: b=float(available[-1]);mu=sd=ei=np.nan
        else:
            from sklearn.gaussian_process.kernels import RBF
            bx=np.array([v['beta1'] for v in trace]).reshape(-1,1)
            uy=np.array([v['utility'] for v in trace]); valid=np.isfinite(uy)
            if valid.sum()<2:
                b=float(available[len(available)//2]);mu=sd=ei=np.nan
            else:
                gp=GaussianProcessRegressor(kernel=ConstantKernel(1.0,constant_value_bounds='fixed')*Matern(length_scale=.3,length_scale_bounds='fixed',nu=2.5)+WhiteKernel(noise_level=.005,noise_level_bounds='fixed'),normalize_y=True,optimizer=None,alpha=1e-6)
                gp.fit(bx[valid],uy[valid]);means,stds=gp.predict(available.reshape(-1,1),return_std=True)
                eis=expected_improvement(means,stds,float(np.nanmax(uy)))
                j=int(np.argmax(eis));b=float(available[j]);mu=float(means[j]);sd=float(stds[j]);ei=float(eis[j])
        used.add(round(b,8));ref=phi@np.array([b,1-b])
        diff=qcloud-ref
        projections=diff@n
        residuals=np.linalg.norm(diff-projections[:,None]*n,axis=1)
        # Select process templates represented near target ray; diversify via known objective best.
        template_scores=[]
        for tid,templ in enumerate(templates):
            mask=np.all(np.isclose(candidates[:,2:],templ[None,:]),axis=1)
            if mask.any():
                j=np.flatnonzero(mask)
                score=float(np.min(residuals[j]+.08*np.maximum(0,-projections[j])))
                template_scores.append((score,tid))
        template_scores.sort()
        chosen_ids=[id for _,id in template_scores[:args.templates]]
        solutions=[]
        for tid in chosen_ids:
            templ=templates[tid]
            mask=np.all(np.isclose(candidates[:,2:],templ[None,:]),axis=1)
            ids=np.flatnonzero(mask)
            rank=ids[np.argsort(residuals[ids]+.08*np.maximum(0,-projections[ids]))[:args.starts]] if len(ids) else []
            starts=[candidates[k,:2] for k in rank]
            starts.append(limits.mean(axis=1))
            for k,start in enumerate(starts):
                qp,_,_=q_value(start,templ)
                cinit=float((qp-ref)@n)
                z0=np.r_[start,np.clip(cinit,args.c_min,args.c_max)]
                def eq(z):
                    qq,_,_=q_value(z[:2],templ)
                    return qq-ref-z[2]*n
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    res=minimize(lambda z:-z[2],z0,method='SLSQP',bounds=bounds,
                                 constraints=[{'type':'eq','fun':eq}],
                                 options={'maxiter':args.maxiter,'ftol':1e-10,'disp':False})
                qq,lc,cv=q_value(res.x[:2],templ)
                r=float(np.linalg.norm(qq-ref-res.x[2]*n))
                feasible=bool(r<=args.eq_tol and (args.c_min-1e-7 <=res.x[2]<=args.c_max+1e-7))
                record=dict(round=t+1,beta1=b,template_id=tid,start_id=k,
                            success_slsqp=bool(res.success),feasible=feasible,
                            solver_message=str(res.message),nit=int(res.nit),
                            residual=r,c=float(res.x[2]),q1=float(qq[0]),q2=float(qq[1]),
                            pred_conductivity=float(10**lc),pred_coverage=float(cv*100),
                            **{FEATURES[j]:float(np.r_[res.x[:2],templ][j]) for j in range(7)})
                candidates_records.append(record)
                if feasible: solutions.append(record)
        best=max(solutions,key=lambda x:x['c']) if solutions else None
        row=dict(round=t+1,beta1=b,beta2=1-b,ref_q1=float(ref[0]),ref_q2=float(ref[1]),
                 n1=float(n[0]),n2=float(n[1]),phi11=float(phi[0,0]),phi12=float(phi[0,1]),
                 phi21=float(phi[1,0]),phi22=float(phi[1,1]),
                 feasible=(best is not None),n_feasible=len(solutions),
                 n_attempts=len([r for r in candidates_records if r['round']==t+1]),
                 c=float(best['c']) if best else np.nan,
                 residual=float(best['residual']) if best else np.nan,
                 q1=float(best['q1']) if best else np.nan,
                 q2=float(best['q2']) if best else np.nan,
                 utility=utility(np.array([best['q1'],best['q2']])) if best else np.nan,
                 GP_U_mean=mu,GP_U_std=sd,EI=ei,
                 template_id=best['template_id'] if best else np.nan,
                 pred_conductivity=best['pred_conductivity'] if best else np.nan,
                 pred_coverage=best['pred_coverage'] if best else np.nan)
        if best:
            for field in FEATURES:row[field]=best[field]
        trace.append(row)
        state.append((b,ref,best))
        print(f"round {t+1:02d} beta={b:.3f} feasible={best is not None} c={row['c']:.5f} residual={row['residual']:.2e}",flush=True)
    trace_df=pd.DataFrame(trace)
    trace_df.to_csv(out/'iteration_trace_exact.csv',index=False)
    pd.DataFrame(candidates_records).to_csv(out/'solver_attempts.csv',index=False)
    pool_df=pd.DataFrame(candidates,columns=FEATURES)
    pool_df['q1']=qcloud[:,0];pool_df['q2']=qcloud[:,1]
    pool_df['pred_conductivity']=10**logs;pool_df['pred_coverage']=100*covs
    # Pareto nondomination (min-min); record fraction and anchor separation.
    pareto=np.ones(len(qcloud),bool)
    for i in range(len(qcloud)):
        pareto[i]=not np.any(np.all(qcloud<=qcloud[i],axis=1)&np.any(qcloud<qcloud[i],axis=1))
    pool_df['pareto']=pareto
    pool_df.to_csv(out/'candidate_pool.csv',index=False)
    # Visual output: geometry, beta, residual, two QoIs, pareto, feasibility
    fig,axs=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    a=axs.ravel()
    a[0].scatter(qcloud[:,0],qcloud[:,1],alpha=.2,s=12,label='Predicted Q(x)')
    a[0].scatter(qcloud[pareto,0],qcloud[pareto,1],s=28,c='darkgreen',label='Nondominated')
    a[0].plot(phi[0],phi[1],'-o',color='firebrick',label='CHIM')
    for j,(b,ref,best) in enumerate(state):
        a[0].scatter(*ref,c='orange',s=23)
        if best:
            end=np.array([best['q1'],best['q2']]);a[0].plot([ref[0],end[0]],[ref[1],end[1]],alpha=.6,lw=1.5)
            a[0].scatter(*end,marker='x',c='black',s=38)
    a[0].set(xlabel='q1 = normalized log conductivity loss',ylabel='q2 = coverage loss',title='A | Exact NBI: CHIM, rays and predicted solutions');a[0].legend(fontsize=7)
    a[1].plot(trace_df['round'],trace_df['beta1'],'o-',label='beta1');a[1].plot(trace_df['round'],trace_df['beta2'],'s--',label='beta2')
    a[1].set(xlabel='Outer BO round',ylabel='beta',title='B | Beta search');a[1].legend()
    a[2].plot(trace_df['round'],trace_df['c'],'o-',label='c (feasible only)');a[2].plot(trace_df['round'],trace_df['residual'],'s-',label='equality residual')
    a[2].axhline(args.eq_tol,color='gray',ls=':',label='tolerance');a[2].set(xlabel='Round',title='C | NBI equality feasibility');a[2].legend(fontsize=7)
    a[3].plot(trace_df['round'],trace_df['n1'],'o-',label='n1');a[3].plot(trace_df['round'],trace_df['n2'],'o-',label='n2')
    a[3].set(xlabel='Round',title='D | Fixed, exact CHIM normal');a[3].legend(fontsize=7)
    a[4].plot(trace_df['round'],trace_df['utility'],'o-',label='Surrogate utility');a[4].plot(trace_df['round'],trace_df['GP_U_mean'],'s--',label='GP_U pre-query mean')
    a[4].set(xlabel='Round',title='E | Utility / GP_U');a[4].legend(fontsize=7)
    a[5].scatter(10**logs,covs*100,s=13,alpha=.3,label='Candidate predictions')
    good=trace_df[trace_df.feasible]
    if len(good):a[5].scatter(good.pred_conductivity,good.pred_coverage,c='red',marker='x',s=50,label='Exact-NBI solutions')
    a[5].set(xlabel='Conductivity S/cm',ylabel='Coverage %',xscale='log',title='F | Pareto-related property space');a[5].legend(fontsize=7)
    fig.suptitle('PEDOT:PSS PFS / equality-constrained NBI | fixed property surrogate (offline)')
    fig.savefig(out/'geometry_exact_nbi.png',dpi=165);plt.close(fig)
    # Each BO round as a separate frame: reference, exact ray, and optimized Q(x).
    try:
        fig2, ax2=plt.subplots(figsize=(8,7))
        writer=PillowWriter(fps=1.1)
        with writer.saving(fig2,str(out/'geometry_by_iteration.gif'),dpi=100):
            for t,(b,ref,best) in enumerate(state,1):
                ax2.clear()
                ax2.scatter(qcloud[:,0],qcloud[:,1],s=13,alpha=.25,label='Candidate Q(x)')
                ax2.scatter(qcloud[pareto,0],qcloud[pareto,1],s=18,c='green',alpha=.6,label='Nondominated')
                ax2.plot(phi[0],phi[1],'-o',c='firebrick',label='CHIM')
                ax2.scatter(*ref,c='darkorange',s=70,label='Phi beta')
                if best:
                    target=np.array([best['q1'],best['q2']])
                    ax2.plot([ref[0],target[0]],[ref[1],target[1]],'k-',lw=2,label='c*n (exact equality)')
                    ax2.scatter(*target,marker='*',s=160,c='gold',edgecolor='black',label='NBI solution')
                    suffix=f"c={best['c']:.4f}, residual={best['residual']:.1e}"
                else: suffix='NO numerical feasible solution'
                ax2.set(xlabel='q1: log conductivity loss',ylabel='q2: coverage loss',
                        title=f'Round {t}: beta=({b:.3f}, {1-b:.3f})\n{suffix}')
                ax2.legend(fontsize=8)
                writer.grab_frame()
        plt.close(fig2)
    except Exception as err:
        warnings.warn('Could not create GIF: '+str(err))
    n_success=int(trace_df.feasible.sum())
    best=trace_df.loc[trace_df.utility.idxmax()] if n_success else None
    lines=['# PFS/NBI exact-equality diagnostic','',
           f'- Historical rows: {len(df)}, dropped: {dropped}',
           f'- Candidate Q points: {len(qcloud)}; nondominated: {int(pareto.sum())}',
           f'- Anchor distance: {float(np.linalg.norm(phi[:,1]-phi[:,0])):.8g}',
           f'- CHIM normal: {n.tolist()}; orthogonality error: {float(n@(phi[:,1]-phi[:,0])):.3e}',
           f'- Feasible NBI beta queries: **{n_success}/{len(trace_df)}** at equality tolerance {args.eq_tol:g}',
           '- Inner problem solved with SLSQP equality constraints Q(x)-Phi*beta-c*n=0 and maximization of c.',
           '- Solvent and coating/post-treatment settings are fixed per historical template; only DMSO and EG concentrations are continuous.',
           '- Multi-start SLSQP is a **local solver**; meeting equality tolerance certifies numerical feasibility, NOT global optimality.',
           '- NBI directions, CHIM and candidate Q predictions remain fixed. No real laboratory feedback.',
           '- If no equality-feasible solution is found for a beta, no solution is fabricated.',
           '- Optimization geometry uses GP predicted means, NOT actual measured Q values.',
           '- GP_U labels are surrogate utilities; no true outer experiment feedback.',
           '- No strict production-safe solvent chemical labels are available; solvent IDs must be checked.',
           '', '## Geometry',f'Phi = {phi.tolist()}',f'n = {n.tolist()}',
           '', '## Best equality-feasible beta query (surrogate-only)']
    if best is not None:
        lines += [f'- Round: {int(best["round"])}; beta=({best.beta1:.4f},{best.beta2:.4f})',
                  f'- c={best.c:.6f}; residual={best.residual:.3e}',
                  f'- predicted conductivity={best.pred_conductivity:.3f} S/cm; coverage={best.pred_coverage:.3f}%',
                  *[f'- {field}: {best[field]}' for field in FEATURES]]
    else: lines+=['- No numerically feasible solution found under this search budget.']
    lines += ['', '## Outputs','- geometry_exact_nbi.png: six-panel overview',
              '- geometry_by_iteration.gif: frame-by-frame exact NBI geometry',
              '- iteration_trace_exact.csv: per beta search',
              '- solver_attempts.csv: all SLSQP starts, including failures',
              '- candidate_pool.csv: Q space and nondomination analysis']
    (out/'REPORT_exact_nbi.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:',out, 'feasible=',n_success,'/',len(trace_df))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='polybot-PEDOT_experiment.csv')
    p.add_argument('--out',default='results_exact_nbi')
    p.add_argument('--candidates',type=int,default=450)
    p.add_argument('--beta-steps',type=int,default=13)
    p.add_argument('--iterations',type=int,default=7)
    p.add_argument('--templates',type=int,default=6)
    p.add_argument('--starts',type=int,default=2)
    p.add_argument('--maxiter',type=int,default=100)
    p.add_argument('--eq-tol',type=float,default=1e-4)
    p.add_argument('--c-min',type=float,default=-1.0)
    p.add_argument('--c-max',type=float,default=2.0)
    p.add_argument('--seed',type=int,default=42)
    run_v4(p.parse_args())

if __name__=='__main__':main()
