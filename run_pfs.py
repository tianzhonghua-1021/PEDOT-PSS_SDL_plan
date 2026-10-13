"""Polybot PEDOT:PSS: dual-objective PFS / surrogate NBI exploration.
Runs offline; exports exactly one PNG, one MD and one CSV in results/.
"""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FIELDS = ['DMSO concentration (Vol%)','EG concentration (Vol%)','Coating speed (mm/sec)',
          'Coating temperature (Celsius)','Post-processing solvent No.',
          'Post coating speed (mm/sec)','Post coating temperature (Celsius)']
CONDUCT='Average conductivity (S/cm)'; COVER='Average coverage (%)'
NUM=[0,1,2,3,5,6]; CAT=4

def model_prep(df):
    frame=df[FIELDS].copy()
    for col in FIELDS: frame[col] = pd.to_numeric(frame[col],errors='coerce')
    good=frame.notna().all(axis=1)&df[CONDUCT].notna()&df[COVER].notna()& (df[CONDUCT]>0)&df[COVER].between(0,100)
    return frame.loc[good],df.loc[good].copy()

def build_encoding(x):
    solvent=sorted(x.iloc[:,CAT].astype(int).unique().tolist())
    enc=ColumnTransformer([('numerical',StandardScaler(),NUM),('solvent',OneHotEncoder(categories=[solvent],handle_unknown='ignore',sparse_output=False),[CAT])],remainder='drop')
    return enc,solvent

def summarize_csv(df):
    allrows=[]
    for _,r in df.iterrows():
        row={'record_type':'historical','source_id':str(r.get('Sample ID','')), 'dataset_split':str(r.get('Dataset','')),
             'conductivity_s_cm':r[CONDUCT], 'coverage_pct':r[COVER]}
        row.update({k:r[k] for k in FIELDS});allrows.append(row)
    return allrows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='polybot-PEDOT_experiment.csv')
    ap.add_argument('--out',default='results')
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--beta-steps',type=int,default=15)
    ap.add_argument('--candidates',type=int,default=1600)
    a=ap.parse_args();rng=np.random.default_rng(a.seed)
    input_path=Path(a.csv);input_path=input_path if input_path.is_absolute() else Path(__file__).resolve().parent/input_path
    out=Path(a.out);out=out if out.is_absolute() else Path(__file__).resolve().parent/out;out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(input_path)
    X,df=model_prep(raw);X=X.reset_index(drop=True);df=df.reset_index(drop=True)
    if len(df)<12: raise ValueError('Too few valid records')
    enc,solvents=build_encoding(X)
    XX=enc.fit_transform(X)
    yc=np.log10(df[CONDUCT].to_numpy(dtype=float)); yv=df[COVER].to_numpy(dtype=float)/100
    # Independent surrogate models, units transformed for numerical stability
    gp=[]
    for y in (yc,yv):
        kernel=ConstantKernel(1.0,constant_value_bounds='fixed')*Matern(length_scale=np.ones(XX.shape[1]),length_scale_bounds='fixed',nu=2.5)+WhiteKernel(noise_level=0.05,noise_level_bounds='fixed')
        m=GaussianProcessRegressor(kernel=kernel,normalize_y=True,optimizer=None,alpha=1e-5,random_state=a.seed)
        m.fit(XX,y);gp.append(m)
    c_low,c_high=float(np.min(yc)),float(np.max(yc)); coverage_min,coverage_max=0.,1.
    def predict(arr,sd=False):
        arr=np.atleast_2d(arr); z=enc.transform(pd.DataFrame(arr,columns=FIELDS))
        if sd:
            mc,sc=gp[0].predict(z,return_std=True);mv,sv=gp[1].predict(z,return_std=True)
            return mc,mv,sc,sv
        return gp[0].predict(z),gp[1].predict(z)
    def obj(arr):
        mc,mv=predict(arr)
        return np.column_stack([np.clip((c_high-mc)/max(c_high-c_low,1e-10),-0.3,1.3),np.clip(1-mv,-0.3,1.3)])
    # Keep post-treatment solvent categorical and never interpolate solvent identity.
    lo=X.min().to_numpy(dtype=float);hi=X.max().to_numpy(dtype=float)
    # Generate candidate pool using observed discrete settings, with interpolation only in concentrations.
    sample_ix=rng.integers(0,len(X),size=a.candidates)
    grid=X.to_numpy(dtype=float)[sample_ix].copy()
    grid[:,0]=rng.uniform(lo[0],hi[0],size=len(grid));grid[:,1]=rng.uniform(lo[1],hi[1],size=len(grid))
    # Include history for estimating CHIM anchors but exclude it for recommending next recipes.
    existing=X.to_numpy(dtype=float)
    pool=np.vstack([existing,grid]); q=obj(pool)
    anchor1=pool[np.argmin(q[:,0])].copy(); anchor2=pool[np.argmin(q[:,1])].copy()
    phi=np.stack([obj(anchor1)[0],obj(anchor2)[0]],axis=1)
    nvec=-np.sum(phi,axis=1); nvec=nvec/np.linalg.norm(nvec) if np.linalg.norm(nvec)>1e-12 else np.array([-1.,-1.])/np.sqrt(2)
    # Fast finite-pool normal-ray projection. Approximate NBI, NOT exact NBI.
    def solve_beta(beta):
        ref=phi @ np.array([beta,1-beta])
        diff=q-ref
        c=diff @ nvec
        residual=np.linalg.norm(diff-c[:,None]*nvec[None,:],axis=1)
        # Constrained approximate ray matching (lexicographic priority: feasible residual, then c)
        valid=np.where((residual <= 0.035) & (np.arange(len(pool)) >= len(existing)))[0]
        if len(valid)==0:return None
        index=valid[int(np.argmax(c[valid]))]
        return {'beta':float(beta),'solvent':int(pool[index,CAT]),'x':pool[index].copy(),
                'q':q[index].copy(),'c':float(c[index]),'residual':float(residual[index]),'solver_success':False}
    # GP_U-style BO over beta, where utility favors both conductivity and coverage.
    beta_grid=np.linspace(0,1,max(7,a.beta_steps));observed=[];trace=[]
    def utility(item):
        # monotonic preference: both objectives high; balanced using geometric mean.
        return float(np.sqrt(np.clip(1-item['q'][0],0,1)*np.clip(1-item['q'][1],0,1)))
    cache={}
    def get(b):
        key=round(float(b),8)
        if key not in cache:cache[key]=solve_beta(key)
        return cache[key]
    for i in range(len(beta_grid)):
        available=[float(b) for b in beta_grid if round(float(b),8) not in {round(float(t['beta']),8) for t in trace}]
        if i<3: b=available[len(available)//2] if i==0 else available[0 if i==1 else -1]
        else:
            bs=np.array([t['beta'] for t in trace if t['feasible']]);us=np.array([t['utility'] for t in trace if t['feasible']]);
            if len(bs)<3:b=available[len(available)//2]
            else:
                k=ConstantKernel(1,constant_value_bounds='fixed')*Matern(length_scale=0.3,length_scale_bounds='fixed',nu=2.5)+WhiteKernel(noise_level=1e-3,noise_level_bounds='fixed')
                g=GaussianProcessRegressor(kernel=k,normalize_y=True,optimizer=None,alpha=1e-5).fit(bs.reshape(-1,1),us)
                m,s=g.predict(np.array(available).reshape(-1,1),return_std=True); imp=m-max(us)-0.005
                scores=imp*norm.cdf(imp/np.maximum(s,1e-10))+s*norm.pdf(imp/np.maximum(s,1e-10))
                b=available[int(np.argmax(scores))]
        v=get(b);entry={'beta':float(b),'feasible':v is not None,'utility':utility(v) if v else np.nan,'nbi_c':v['c'] if v else np.nan,'nbi_residual':v['residual'] if v else np.nan,'solution':v};trace.append(entry)
    # Use candidates near feasible NBI solutions, plus strong Pareto candidates, excluding exact measured recipes.
    scores=np.sqrt(np.clip(1-q[len(existing):,0],0,1)*np.clip(1-q[len(existing):,1],0,1))
    m1,m2,s1,s2=predict(grid,sd=True)
    obs_log=np.max(yc);obs_cov=np.max(yv)
    # Ambitious but interpretable: two targets and epistemic exploration bonus.
    acq=scores+0.035*(s1/np.maximum(c_high-c_low,1e-10)+s2)
    # Experimental novelty tolerance relative to existing rows.
    same=np.zeros(len(grid),dtype=bool)
    for x in existing:
        exact=(grid[:,CAT]==x[CAT])&np.all(np.isclose(grid[:,NUM],x[NUM],atol=1e-5),axis=1)
        same |= exact
    acq[same]=-np.inf
    chosen=np.argsort(acq)[::-1][:10]
    # Treat NBI candidates as PFS-driven proposals; take feasible NBI proposal ranking first.
    feasible=[v for v in cache.values() if v is not None]
    feasible.sort(key=utility,reverse=True)
    if feasible:
        recommend=feasible[0]['x']; recommend_method='PFS GP_U + finite-pool approximate NBI'
    else:
        recommend=grid[chosen[0]]; recommend_method='Fallback: Pareto-style surrogate candidate ranking (NBI infeasible)'
    # Validation using fixed data splits: Train/Test is preserved, with RF comparator.
    def validation(y):
        if {'Train','Test'}.issubset(set(df.Dataset)):
            tr=np.where(df.Dataset=='Train')[0];te=np.where(df.Dataset=='Test')[0]
        else:tr,te=np.arange(len(df))[:int(len(df)*.7)],np.arange(len(df))[int(len(df)*.7):]
        rf=RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=a.seed)
        rf.fit(XX[tr],y[tr]); pred=rf.predict(XX[te]);return {'mae':mean_absolute_error(y[te],pred),'r2':r2_score(y[te],pred),'test_n':len(te)}
    metrics_c=validation(yc); metrics_v=validation(yv)
    # One CSV: long-form records for historical, candidate, beta and final recommendation.
    records=summarize_csv(df)
    def add_recipe(record_type,x,rank=None,extra=None):
        a1,a2,b1,b2=predict(x,sd=True); r={'record_type':record_type,'rank':rank,**{k:float(x[j]) for j,k in enumerate(FIELDS)},'predicted_conductivity_s_cm':10**float(a1[0]),'predicted_coverage_pct':float(a2[0])*100,'predicted_log10_conductivity_sd':float(b1[0]),'predicted_coverage_sd_pct':float(b2[0])*100}
        if extra:r.update(extra)
        records.append(r)
    for rank,idx in enumerate(chosen,1):add_recipe('candidate',grid[idx],rank,{'acquisition':float(acq[idx])})
    for iteration,v in enumerate(trace,1):
        item=v['solution']
        if item:add_recipe('pfs_beta',item['x'],iteration,{'beta':v['beta'],'utility':v['utility'],'nbi_c':v['nbi_c'],'nbi_residual':v['nbi_residual'],'feasible':True})
        else:records.append({'record_type':'pfs_beta','rank':iteration,'beta':v['beta'],'feasible':False})
    add_recipe('next_experiment',recommend,1,{'recommend_method':recommend_method})
    pd.DataFrame.from_records(records).to_csv(out/'all_results.csv',index=False)
    # Multi-panel only-one figure.
    fig,ax=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    ax[0,0].scatter(df[COVER],df[CONDUCT],c=df['DMSO concentration (Vol%)'],cmap='viridis',s=35,alpha=.8);ax[0,0].set_yscale('log');ax[0,0].set(xlabel='Coverage (%)',ylabel='Conductivity (S/cm)',title='A | Observed performance')
    ax[0,1].scatter(df['DMSO concentration (Vol%)'],np.log10(df[CONDUCT]),c=df['EG concentration (Vol%)'],cmap='plasma',s=32);ax[0,1].set(xlabel='DMSO (vol%)',ylabel='log10 conductivity (S/cm)',title='B | Additives vs conductivity')
    ax[0,2].scatter(df['EG concentration (Vol%)'],df[COVER],c=df['DMSO concentration (Vol%)'],cmap='viridis',s=32);ax[0,2].set(xlabel='EG (vol%)',ylabel='Coverage (%)',title='C | Additives vs coverage')
    usable=[v for v in trace if v['feasible']];ax[1,0].scatter([v['beta'] for v in usable],[v['utility'] for v in usable],c=[v['nbi_residual'] for v in usable],cmap='viridis',s=55);ax[1,0].set(xlabel='beta_1  (beta_2 = 1-beta_1)',ylabel='Predicted joint utility',title='D | PFS beta search')
    ax[1,1].scatter(df[COVER],np.log10(df[CONDUCT]),alpha=.4,label='Measured');ax[1,1].scatter(m2[chosen]*100,m1[chosen],marker='x',s=70,label='Top unseen candidates');ax[1,1].set(xlabel='Coverage (%)',ylabel='log10 conductivity',title='E | Candidate selection');ax[1,1].legend(fontsize=8)
    ax[1,2].scatter([v['beta'] for v in usable],[v['nbi_residual'] for v in usable],s=45);ax[1,2].axhline(.035,ls='--',color='gray',label='Approximation tolerance');ax[1,2].set(xlabel='beta_1',ylabel='NBI equality residual',title='F | Constraint diagnostics');ax[1,2].legend(fontsize=8)
    fig.suptitle('Polybot PEDOT:PSS | Offline PFS / NBI feasibility & next-experiment discovery',fontsize=15)
    fig.savefig(out/'dashboard.png',dpi=180);plt.close(fig)
    cm,cv,cs,vs=predict(recommend,sd=True)
    lines=['# Polybot PEDOT:PSS — PFS toy experiment report','',f'- Input: {len(raw)} records ({len(df)} valid); split: { {str(k):int(v) for k,v in raw.Dataset.value_counts().items()} }. ',f'- Target 1: maximize conductivity (S/cm); target 2: maximize coverage (%).','- DMSO and EG are additive concentrations; the post-processing solvent is supplied only as a numeric category ID.','- Conductivity is modeled in log10 space because values range across orders of magnitude.','', '## Next experiment (model recommendation)',f'- Selection: **{recommend_method}**']
    for j,k in enumerate(FIELDS):lines.append(f'- **{k}**: {recommend[j]:.3f}')
    lines += [f'- Predicted conductivity: **{10**cm[0]:.3f} S/cm** (model log10 SD {cs[0]:.3f}).',f'- Predicted coverage: **{cv[0]*100:.2f}%** (model SD {vs[0]*100:.2f} percentage points).','', '## Offline validation (Random Forest sanity-check on original Train/Test split)',f'- Conductivity log10 MAE: {metrics_c["mae"]:.3f}; R²: {metrics_c["r2"]:.3f}; test size: {metrics_c["test_n"]}.',f'- Coverage fraction MAE: {metrics_v["mae"]:.3f}; R²: {metrics_v["r2"]:.3f}; test size: {metrics_v["test_n"]}.','', '## PFS / NBI',f'- Evaluated β values: {len(trace)}; feasible NBI solutions: {len(usable)}.', '- β = (β1, 1−β1) specifies the NBI CHIM reference, not raw property percentages.', '- The inner method searches a finite candidate pool: sampled DMSO/EG concentrations and observed combinations of other settings; the solvent ID is categorical.', '- Finite-pool NBI uses orthogonal projection and residual tolerance 0.035 in normalized objective space. It is NOT an exact equality-constrained NBI solver.', '- GP_U-style Expected Improvement searches β using predicted balanced utility from the property surrogate. GP_Q, RF feasibility and KDE are not implemented in this simplified proof of concept.', '', '## Critical limitations','- All NBI results use surrogate predictions, not fresh measurements; this is NOT a validated autonomous wet-lab closed loop.', '- The original Test rows are used in exploratory surrogate fitting and recommendation; the RF Train/Test diagnostic is separate and must NOT be treated as independent validation of the final GP.', '- Solvent category numbers cannot be translated into solvent chemical identities without a lookup table.', '- Recommendations interpolate DMSO and EG only within observed ranges; prediction SDs are not experimentally calibrated.', '- Very low conductivity and high measured replicate SD in some rows warrant inspection before experimental execution.', '- Geometric-mean joint utility is an assumed preference; change it if experimental priorities differ.',
           '- If original Train/Test baseline R² is negative, treat recommendation as exploratory only; inspect distribution shift and validate with new data.', '', '## Output schema','- all_results.csv: record_type historical / candidate / pfs_beta / next_experiment.','- dashboard.png: six-panel summary.','- REPORT.md: this file.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Generated',out.resolve());print('Feasible beta:',len(usable),'/',len(trace));print('Recommendation',dict(zip(FIELDS,np.round(recommend,3))));print('Objectives: conductivity / coverage, predicted',10**cm[0],cv[0]*100)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=UserWarning)
    main()
