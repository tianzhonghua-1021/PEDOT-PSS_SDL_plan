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
from scipy.stats import norm, gaussian_kde
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
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





def run(args):
    root=Path(__file__).resolve().parent
    inp=Path(args.csv); inp=inp if inp.is_absolute() else root/inp
    out=Path(args.out);out=out if out.is_absolute() else root/out
    out.mkdir(parents=True,exist_ok=True)
    df,dropped=load_data(inp)
    trans,models,scales=train_models(df)
    pool=make_candidates(df,args.candidates,args.seed)
    qcloud,logc,cov,_,_=predict_objectives(pool,trans,models,scales)
    phi,n,_,_=compute_nbi_geometry(qcloud)
    raw=df[FEATURES].to_numpy(float)
    templates=np.unique(raw[:,2:],axis=0)
    bounds=[(raw[:,0].min(),raw[:,0].max()),(raw[:,1].min(),raw[:,1].max()),(args.c_min,args.c_max)]
    qlo=qcloud.min(axis=0);qhi=qcloud.max(axis=0)
    beta_grid=np.round(np.linspace(args.beta_min,args.beta_max,args.beta_steps),8)
    allrows=[]; cache={}; calls=0
    def utility(q):
        u=np.clip((qhi-q)/(qhi-qlo+1e-9),0,1)
        return float(np.sqrt(u[0]*u[1]))
    def eval_q(x):
        qs,l,cv,_,_=predict_objectives(np.asarray(x).reshape(1,-1),trans,models,scales)
        return qs[0],float(l[0]),float(cv[0])
    def oracle(beta):
        nonlocal calls
        key=round(float(beta),8)
        if key in cache:return cache[key]
        calls+=1
        ref=phi@np.array([key,1-key]);diff=qcloud-ref
        proj=diff@n;ray_res=np.linalg.norm(diff-proj[:,None]*n,axis=1)
        scores=[]
        for tid,t in enumerate(templates):
            mask=np.all(np.isclose(pool[:,2:],t),axis=1)
            ids=np.flatnonzero(mask)
            if len(ids):scores.append((float(np.min(ray_res[ids]+.06*np.maximum(-proj[ids],0))),tid))
        scores.sort();sol=[];attempts=0
        for _,tid in scores[:args.templates]:
            templ=templates[tid]
            mask=np.all(np.isclose(pool[:,2:],templ),axis=1)
            inds=np.flatnonzero(mask)
            inds=inds[np.argsort(ray_res[inds]+.06*np.maximum(-proj[inds],0))[:args.starts]]
            for ix in inds:
                start=pool[ix,:2]
                q0,_,_=eval_q(np.r_[start,templ]); c0=float((q0-ref)@n)
                z0=np.array([*start,float(np.clip(c0,args.c_min,args.c_max))])
                def ceq(z):return eval_q(np.r_[z[:2],templ])[0]-ref-z[2]*n
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result=minimize(lambda z:-z[2],z0,method='SLSQP',bounds=bounds,
                                    constraints=[{'type':'eq','fun':ceq}],
                                    options={'maxiter':args.maxiter,'ftol':1e-9})
                attempts+=1
                if not np.all(np.isfinite(result.x)):continue
                q,lc,cv=eval_q(np.r_[result.x[:2],templ]);r=float(np.linalg.norm(q-ref-result.x[2]*n))
                if r<=args.eq_tol and args.c_min-1e-6<=result.x[2]<=args.c_max+1e-6:
                    sol.append(dict(c=float(result.x[2]),residual=r,q1=float(q[0]),q2=float(q[1]),
                        utility=utility(q),conductivity=float(10**lc),coverage=float(cv*100),
                        **{f:float(v) for f,v in zip(FEATURES,np.r_[result.x[:2],templ])}))
        best=max(sol,key=lambda v:v['c']) if sol else None
        record=dict(beta=key,beta2=1-key,feasible=int(best is not None),attempts=attempts,
                    ref_q1=float(ref[0]),ref_q2=float(ref[1]),n1=float(n[0]),n2=float(n[1]),
                    phi11=float(phi[0,0]),phi12=float(phi[0,1]),phi21=float(phi[1,0]),phi22=float(phi[1,1]))
        if best:record.update(best)
        else:record.update(c=np.nan,residual=np.nan,q1=np.nan,q2=np.nan,utility=np.nan,conductivity=np.nan,coverage=np.nan)
        cache[key]=record
        return record
    # Precompute the finite-grid surrogate oracle once. Every method uses exactly the same
    # NBI solver settings/results, with repeated queries charged according to the same budget.
    print('Building shared finite-beta NBI oracle...',flush=True)
    for i,b in enumerate(beta_grid):
        res=oracle(b)
        print(f'  NBI {i+1}/{len(beta_grid)} beta={b:.3f} feasible={bool(res["feasible"])}',flush=True)
    oracle_df=pd.DataFrame(list(cache.values()))
    # The classification target is numeric solver feasibility, not wet-lab feasibility.
    def score_candidates(history, available, method, rng):
        b_seen=np.array([r['beta'] for r in history]); feas=np.array([r['feasible'] for r in history])
        good=[r for r in history if r['feasible']]
        feats=available.reshape(-1,1)
        p=np.full(len(available),np.mean(feas) if len(feas) else .5)
        rf_trained=False
        if method=='full' and len(history)>=4 and len(np.unique(feas))==2:
            rf=RandomForestClassifier(n_estimators=60,max_depth=4,random_state=args.seed,class_weight='balanced')
            rf.fit(b_seen.reshape(-1,1),feas)
            p=rf.predict_proba(feats)[:,list(rf.classes_).index(1)]
            rf_trained=True
        # GP_U learns utility as a function of beta; GP_Q learns both solved QoIs.
        if len(good)>=2:
            x=np.array([r['beta'] for r in good]).reshape(-1,1)
            y=np.array([r['utility'] for r in good])
            kern=ConstantKernel(1.,constant_value_bounds='fixed')*Matern(length_scale=.32,length_scale_bounds='fixed',nu=2.5)+WhiteKernel(noise_level=.005,noise_level_bounds='fixed')
            gu=GaussianProcessRegressor(kernel=kern,normalize_y=True,optimizer=None,alpha=1e-5).fit(x,y)
            mu,sd=gu.predict(feats,return_std=True)
            ei=expected_improvement(mu,sd,max(y),xi=.004)
            if method in ('full','no_rf_kde') and len(good)>=3:
                pred=[]
                for target in ('q1','q2'):
                    gy=np.array([r[target] for r in good])
                    gq=GaussianProcessRegressor(kernel=kern,normalize_y=True,optimizer=None,alpha=1e-5).fit(x,gy)
                    pred.append(gq.predict(feats))
                q_pred=np.column_stack(pred)
                q_score=np.sqrt(np.prod(np.clip((qhi-q_pred)/(qhi-qlo+1e-9),0,1),axis=1))
                q_score=(q_score-q_score.min())/(np.ptp(q_score)+1e-9)
                ei=ei/(max(ei.max(),1e-10))+.20*q_score
            else:ei=ei/(max(ei.max(),1e-10))
        else:ei=np.ones(len(available))
        if method=='full':
            ei=ei*np.maximum(.05,p)
            if len(good)>=3 and np.std([r['beta'] for r in good])>1e-5:
                try:
                    kde=gaussian_kde([r['beta'] for r in good],bw_method=.6)
                    den=kde(available);den=den/(den.max()+1e-12)
                    ei=ei*(.6+.4*den)
                except (ValueError,np.linalg.LinAlgError):pass
        return ei,p,rf_trained
    strategies=['full','no_gp_q','no_rf_kde','random_beta']
    records=[]
    # shared warm-start betas and shared finite evaluation budget for all variants
    warm=[float(beta_grid[np.argmin(abs(beta_grid-v))]) for v in (.5,0,1)]
    warm=list(dict.fromkeys(warm))
    for seed in range(args.seeds):
        for name in strategies:
            rng=np.random.default_rng(args.seed+seed*107)
            hist=[];chosen=set()
            for t in range(min(args.budget,len(beta_grid))):
                avail=np.array([b for b in beta_grid if round(float(b),8) not in chosen])
                if t<len(warm):b=warm[t];score=np.nan;p=np.nan;rf_used=False
                elif name=='random_beta':b=float(rng.choice(avail));score=np.nan;p=np.nan;rf_used=False
                else:
                    values,ps,rf_used=score_candidates(hist,avail,name,rng)
                    ix=int(np.argmax(values));b=float(avail[ix]);score=float(values[ix]);p=float(ps[ix])
                chosen.add(round(b,8))
                obs=oracle(b).copy();hist.append(obs)
                best=max([r['utility'] for r in hist if r['feasible']],default=np.nan)
                records.append(dict(record_type='iteration',strategy=name,seed=seed,round=t+1,
                        acquisition=score,p_feasible=p,rf_trained=rf_used,best_utility=best,**obs))
    trace=pd.DataFrame(records)
    # Assess actual FNS comparison; alpha GP_Q and classifier can be uninformative when all labels feasible.
    long=[]
    for r in oracle_df.to_dict('records'):long.append(dict(record_type='oracle',strategy='shared_nbi',**r))
    long+=records
    for j,row in df.iterrows():
        long.append(dict(record_type='historical',sample_id=j,conductivity=row[CONDUCTIVITY],coverage=row[COVERAGE],
                         **{f:row[f] for f in FEATURES}))
    results=pd.DataFrame(long)
    results.to_csv(out/'all_results.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(18,10),constrained_layout=True);a=axs.ravel()
    a[0].scatter(qcloud[:,0],qcloud[:,1],s=11,alpha=.2,label='Candidate GP means')
    a[0].plot(phi[0],phi[1],'o-',color='firebrick',label='CHIM')
    subset=oracle_df[oracle_df.feasible==1]
    if len(subset):a[0].scatter(subset.q1,subset.q2,marker='x',s=55,c='orange',label='NBI feasible')
    a[0].set(xlabel='q1: conductivity loss',ylabel='q2: coverage loss');a[0].legend(fontsize=8)
    a[1].plot(oracle_df.beta,oracle_df.c,'o-',label='c');a[1].set(xlabel='beta1',ylabel='c (feasible solutions)');a[1].axhline(0,c='gray',lw=.8)
    a[2].scatter(oracle_df.beta,oracle_df.residual,marker='o',label='equality residual')
    a[2].axhline(args.eq_tol,ls=':',c='firebrick',label='tolerance');a[2].set(xlabel='beta1',ylabel='equality residual');a[2].legend(fontsize=8)
    for name in strategies:
        dfm=trace[trace.strategy==name]
        agg=dfm.groupby('round').best_utility.agg(['mean','std']).reset_index()
        a[3].plot(agg['round'],agg['mean'],marker='o',label=name)
        a[3].fill_between(agg['round'].to_numpy(),(agg['mean']-agg['std'].fillna(0)).to_numpy(),(agg['mean']+agg['std'].fillna(0)).to_numpy(),alpha=.1)
    a[3].set(xlabel='NBI beta queries',ylabel='Best predicted utility');a[3].legend(fontsize=8)
    for name in strategies:
        v=trace[(trace.strategy==name)&(trace.seed==0)]
        a[4].plot(v['round'],v['beta'],marker='o',label=name)
    a[4].set(xlabel='Outer round',ylabel='beta1 (seed 0)');a[4].legend(fontsize=8)
    a[5].scatter(10**logc,cov*100,s=10,alpha=.3,label='Predicted candidate Q')
    if len(subset):a[5].scatter(subset.conductivity,subset.coverage,marker='x',c='red',s=45,label='NBI solutions')
    a[5].set(xlabel='Predicted conductivity (S/cm)',ylabel='Predicted coverage (%)',xscale='log');a[5].legend(fontsize=8)
    labels=['a  |  Objective geometry','b  |  NBI c vs beta','c  |  Numerical feasibility','d  |  Ablation: search budget','e  |  Beta search paths','f  |  Predicted property space']
    for ax,label in zip(a,labels):
        ax.text(.02,.98,label,transform=ax.transAxes,ha='left',va='top',fontsize=10,fontweight='bold',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    fig.suptitle('Polybot PFS v5: GP_U + GP_Q + RF + KDE | fixed-surrogate offline comparison',fontsize=15)
    fig.savefig(out/'dashboard.png',dpi=160);plt.close(fig)
    final=trace[trace['round']==trace['round'].max()].groupby('strategy').best_utility.agg(['mean','std'])
    all_feasible=bool(oracle_df.feasible.eq(1).all())
    winner=oracle_df.loc[oracle_df.utility.idxmax()] if subset.shape[0] else None
    rep=['# Polybot PFS v5 — GP_U / GP_Q / RF / KDE and ablations','',
         '## Dataset and design',f'- Historical valid rows: {len(df)}; removed: {dropped}.',
         f'- Finite beta oracle grid: {len(beta_grid)} values within [{args.beta_min}, {args.beta_max}].',
         f'- NBI numerically feasible: {int(oracle_df.feasible.sum())}/{len(oracle_df)}; tolerance {args.eq_tol:g}.',
         f'- Seeds: {args.seeds}; NBI beta-query budget per strategy: {args.budget}.',
         '- All strategies share the same precomputed, fixed-GP NBI oracle; no new experimental results.',
         '- Each query counts one beta evaluation, not the underlying SLSQP starts; use solver cost logs for compute-budget comparisons.',
         '', '## Ablation: best surrogate utility (mean ± SD over seeds)',
         '| Strategy | Mean | SD |','|---|---:|---:|']
    for name,r in final.iterrows():rep.append(f'| {name} | {r["mean"]:.5f} | {r["std"]:.5f} |')
    rep+=['','## Interpretation of models',
          '- `full`: GP_U expected improvement + GP_Q predicted QoI utility adjustment + RF feasibility probability + KDE feasible-beta density.',
          '- `no_gp_q`: GP_U + RF + KDE (remove GP_Q).',
          '- `no_rf_kde`: GP_U + GP_Q (remove feasibility classification and KDE).',
          '- `random_beta`: uniform random selection of remaining grid beta values after shared warm-start.',
          '- GP_Q uses two regressors beta→(q1,q2), fitted only on evaluated feasible beta points.',
          '- RF labels are numerical NBI solver feasible/infeasible. If labels have only one class, RF is NOT fitted; a constant feasibility prior is used.',
          '- KDE uses successful beta evaluations only. Fallback is uniform if too few distinct betas exist.',
          f'- All β oracle points feasible: {all_feasible}; ' + ('RF cannot learn a meaningful boundary on this finite grid.' if all_feasible else 'Check class balance before trusting RF probabilities.'),
          '', '## Highest-utility oracle solution (diagnostic only)']
    if winner is not None:
        rep.append(f'- β = ({winner.beta:.4f}, {winner.beta2:.4f}); predicted conductivity = {winner.conductivity:.2f} S/cm; coverage = {winner.coverage:.2f}%; c = {winner.c:.6f}; residual = {winner.residual:.3g}.')
        for f in FEATURES:rep.append(f'- {f}: {winner[f]}')
    rep+=['','## Limitations and validity',
          '- Only DMSO and EG are optimized continuously; other settings are fixed to observed templates.',
          '- NBI equality is verified numerically, but multistart SLSQP does not certify global max c.',
          '- Φ and n are built from predicted finite candidate anchors; extrapolations outside this design are not validated.',
          '- Property GP was fitted on ALL available historical records; no prospective independent test.',
          '- GP_U/GP_Q/RF/KDE are lightweight analogues; exact implementation may differ from original alloy paper.',
          '- Utility and all performance numbers are predicted by fixed surrogates; ablation is not real-world BO performance evidence.',
          '- A single global best candidate may dominate both objectives; check Pareto size before interpreting PFS improvements.',
          '- Solvent labels are integer IDs; resolve the chemistry before any wet-lab run.',
          '', '## Output contract', '- dashboard.png: one multipanel figure (a–f).',
          '- all_results.csv: historical/oracle/iteration rows, distinguished by `record_type`.',
          '- REPORT.md: this report.']
    (out/'REPORT.md').write_text('\n'.join(rep)+'\n',encoding='utf-8')
    print('Wrote',out.resolve(), 'files:',sorted(p.name for p in out.iterdir()),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Full-feature PFS educational ablation on Polybot PEDOT:PSS')
    ap.add_argument('--csv',default='polybot-PEDOT_experiment.csv')
    ap.add_argument('--out',default='results_v5')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--candidates',type=int,default=350)
    ap.add_argument('--beta-min',type=float,default=-.25)
    ap.add_argument('--beta-max',type=float,default=1.25)
    ap.add_argument('--beta-steps',type=int,default=11)
    ap.add_argument('--budget',type=int,default=7)
    ap.add_argument('--seeds',type=int,default=5)
    ap.add_argument('--templates',type=int,default=5)
    ap.add_argument('--starts',type=int,default=1)
    ap.add_argument('--maxiter',type=int,default=80)
    ap.add_argument('--c-min',type=float,default=-1.0)
    ap.add_argument('--c-max',type=float,default=1.0)
    ap.add_argument('--eq-tol',type=float,default=1e-4)
    run(ap.parse_args())
