#!/usr/bin/env python3
"""Continuous overlap balancing with weights refitted in every sky resample."""
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import healpy as hp
from photdis.stage1 import require_compute
from halo_common import *
from sv18_common import cached,index_join,interval
from sv18_populations import cells


def weighted_cdf_gap(x,y,w):
    order=np.argsort(x,kind='stable');xx=x[order];yy=y[order];ww=w[order]
    sums=[ww[yy==i].sum() for i in (0,1)]
    if min(sums)<=0:return np.nan
    a=np.cumsum(np.where(yy==0,ww/sums[0],0));b=np.cumsum(np.where(yy==1,ww/sums[1],0))
    at=np.r_[np.flatnonzero(np.diff(xx)>0),len(xx)-1]
    return float(np.max(abs(a[at]-b[at])))


def balance(x,y,mult):
    use=mult>0
    if len(np.unique(y[use]))<2:return np.zeros(len(y))
    mean=np.average(x[use],axis=0,weights=mult[use]);var=np.average((x[use]-mean)**2,axis=0,weights=mult[use])
    z=np.clip((x-mean)/np.maximum(np.sqrt(var),1e-5),-8,8)
    design=np.column_stack((z,z*z,z*z*z,z[:,1]*z[:,2]))
    model=LogisticRegression(C=10.,max_iter=1000,tol=1e-7,solver='lbfgs')
    model.fit(design[use],y[use],sample_weight=mult[use])
    p=expit(model.decision_function(design))
    common=(p>=.05)&(p<=.95)
    return mult*np.where(y==1,1-p,p)*common


def contrast(delta,y,w):
    if any(w[y==i].sum()==0 for i in (0,1)):return np.nan
    return np.average(delta[y==1],weights=w[y==1])-np.average(delta[y==0],weights=w[y==0])


def moments(x,y,w):
    out=[]
    for i,name in enumerate(('r','gr','rz','max_error','EBV','reference_distance','north')):
        a=[np.average(x[y==j,i],weights=w[y==j]) for j in (0,1)]
        v=[np.average((x[y==j,i]-a[j])**2,weights=w[y==j]) for j in (0,1)]
        den=np.sqrt(sum(v)/2)
        out.append(dict(variable=name,ordinary_mean=a[0],metal_poor_mean=a[1],mean_difference=a[1]-a[0],
             SMD=(a[1]-a[0])/den if den>1e-12 else 0.,CDF_gap=weighted_cdf_gap(x[:,i],y,w)))
    return out


def main():
    require_compute();d=folder('balance');inputs={}
    t=cached(inputs,('TYPE','ls_id','region','m','observed_r','err','EBV','pop','heldout','state','q','RA','DEC',
        'gaia_stage','gaia_d','specdis_stage','specdis_d','spdist_stage','spdist_d'))
    role=read(OUT/'roles',('reference_roles.npz','TYPE','ls_id','direct_fit_or_calibration'),inputs)
    j=index_join(t['ls_id'],role['ls_id']);assert (j>=0).all()
    use=t['heldout']&(t['state']==3)&np.isin(t['pop'],(1,2))&~role['direct_fit_or_calibration'][j]
    t={k:v[use] for k,v in t.items()}
    block=hp.ang2pix(C['balance']['sky_nside'],t['RA'],t['DEC'],nest=True,lonlat=True)
    summary=[];audit=[];drawtable=[];sensitivity=[];private=[]
    for ref in ('specdis','spdist','gaia'):
        for lo,hi in ((2,5),(5,10)):
            use=(t[ref+'_stage']==3)&(t[ref+'_d']>=lo)&(t[ref+'_d']<hi)
            ii=np.flatnonzero(use)
            if not len(ii):continue
            y=(t['pop'][ii]==2).astype(int);bb=block[ii]
            x=np.column_stack((t['observed_r'][ii],t['m'][ii,0]-t['m'][ii,1],t['m'][ii,1]-t['m'][ii,2],
                t['err'][ii].max(axis=1),t['EBV'][ii],t[ref+'_d'][ii],t['region'][ii]==3))
            delta=5*np.log10(t['q'][ii,2]/t[ref+'_d'][ii])
            finite=np.isfinite(x).all(axis=1)&np.isfinite(delta)
            ii=ii[finite];x=x[finite];y=y[finite];bb=bb[finite];delta=delta[finite]
            if min(np.bincount(y,minlength=2))<5:continue
            w=balance(x,y,np.ones(len(y)));point=contrast(delta,y,w)
            for stage,ww in (('before',np.ones(len(y))),('after',w)):
                if any(ww[y==i].sum()==0 for i in (0,1)):continue
                audit.extend(dict(reference=ref,lo=lo,stage=stage,**r) for r in moments(x,y,ww))
            keys,inv=np.unique(bb,return_inverse=True);draws=[]
            for draw in range(C['balance']['draws']):
                mult=np.bincount(RNG.integers(len(keys),size=len(keys)),minlength=len(keys))[inv]
                ww=balance(x,y,mult);val=contrast(delta,y,ww);draws.append(val)
                drawtable.append(dict(reference=ref,lo=lo,draw=draw,shift_mu=val,weight=ww.sum()))
            coarse=cells(t,'original',t[ref+'_d'])[ii];_,ci=np.unique(coarse,axis=0,return_inverse=True)
            cellweight=np.bincount(ci,weights=w);skyweight=np.bincount(inv,weights=w)
            highcell=int(np.argmax(cellweight))
            for kind,labels in (('sky',keys),('highest_weight_cell',[highcell])):
                for key in labels:
                    keep=(bb!=key) if kind=='sky' else ci!=key
                    ww=balance(x,y,keep.astype(float));val=contrast(delta,y,ww)
                    sensitivity.append(dict(reference=ref,lo=lo,kind=kind,omitted=int(key),shift_mu=val))
            smds=[r['SMD'] for r in audit if r['reference']==ref and r['lo']==lo and r['stage']=='after']
            summary.append(dict(reference=ref,lo=lo,hi=hi,n=len(y),ordinary=int((y==0).sum()),metal_poor=int(y.sum()),
                common_fraction=float((w>0).mean()),shift_mu=point,distance_ratio=10**(point/5),
                effective_n=w.sum()**2/(w@w) if w.any() else 0,
                ordinary_ESS=w[y==0].sum()**2/(w[y==0]@w[y==0]) if w[y==0].any() else 0,
                metal_poor_ESS=w[y==1].sum()**2/(w[y==1]@w[y==1]) if w[y==1].any() else 0,
                max_cell_weight_fraction=cellweight.max()/w.sum() if w.any() else None,
                max_sky_weight_fraction=skyweight.max()/w.sum() if w.any() else None,
                maximum_abs_SMD=max(abs(np.array(smds))) if smds else None,
                balance_pass=bool(smds and max(abs(np.array(smds)))<.1),**interval(draws)))
            private.append(dict(TYPE=t['TYPE'][ii],ls_id=t['ls_id'][ii],reference=np.full(len(ii),ref),lo=np.full(len(ii),lo),weight=w))
            print(summary[-1],flush=True)
    csv_rows(d/'balanced_metallicity.csv',summary);csv_rows(d/'observable_balance.csv',audit)
    csv_rows(d/'sky_refit_draws.csv',drawtable);csv_rows(d/'leave_one_sensitivity.csv',sensitivity)
    atomic_npz(d/'balanced_ids.npz',**{k:np.concatenate([p[k] for p in private]) for k in private[0]})
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
        for ax,lo in zip(axes,(2,5)):
            for ref,col in zip(('specdis','spdist','gaia'),(BLUE,ORANGE,GREEN)):
                a=[r for r in audit if r['reference']==ref and r['lo']==lo and r['stage']=='after']
                if a:ax.plot(range(len(a)),[r['SMD'] for r in a],'o-',ms=3,color=col,label=ref)
            ax.axhspan(-.1,.1,color='.85',alpha=.45);ax.axhline(0,color='.5',ls=':',lw=.6)
            ax.set(xticks=range(7),xticklabels=['r','g−r','r−z','σ','E(B−V)','d','North'],title=f'{lo}–{5 if lo==2 else 10} kpc',ylabel='Standardized mean difference')
        axes[0].legend(frameon=False,fontsize=7)
        figure(d,'continuous_observable_balance',f,audit)
    complete(d,inputs,estimand='metal-poor minus ordinary overlap-population weighted mean modulus residual; differs from old coarse-cell median contrast',
        design='standardized linear, squared, cubic terms and gr*rz; L2 logistic C=10; common propensity [.05,.95]; independently refit each sky bootstrap and omission',
        inference='reference-scale agreement, not causal chemical identification; no new blind validation')


if __name__=='__main__':main()
