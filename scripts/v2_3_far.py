#!/usr/bin/env python3
"""Post-freeze branch diagnostics and explicitly selected nonstellar controls."""
import json
import joblib
import numpy as np
from scipy.special import ndtr
from photdis.baseline_parallax import normal_interval
from v2_3_prepare import ROOT, DEST, TABLES
from v2_3_diagnostics import references
from v2_1_near import load, subset, csv_rows, save_json, sha
from photdis.stage1 import require_compute
from photdis.baseline_v0 import metrics, distance_quantiles
from photdis.baseline_v1 import quantiles_interpolated
from photdis.probability_v2_1 import JointReferenceForest
from baseline_v1_4_experiments import arrays


def main():
    require_compute()
    t=load(ROOT/'data/v2_1/development.npz');train=load(DEST/'training.npz')
    keep=np.zeros(len(t['identity']),bool)
    for kind in ('chosen','Gaia','specdis','spdist'):
        ref,_=references(t,kind);d=10**((ref-10)/5)
        keep|=(t['class_label']==1)&(d>=10)&(d<50)
    rows=np.flatnonzero(keep);q=subset(t,rows)
    control=joblib.load(DEST/'control.joblib');candidate=joblib.load(DEST/'candidate.joblib')
    m,_,_=arrays(train,np.arange(len(train['identity'])))
    joint=JointReferenceForest(control,m,train['Mg'],train['class_label'],train['label_source'])
    values={n:dict(mu=np.full((len(rows),5),np.nan),mass=np.zeros(len(rows))) for n in ('control','candidate')}
    for start in range(0,len(rows),128):
        rr=np.arange(start,min(start+128,len(rows)));m,e,ids=arrays(q,rr)
        pp=joint.predict(m,e,ids);p=pp['joint'][:,1].astype(float);mass=p.sum(axis=1);good=mass>0
        values['control']['mass'][rr]=mass
        values['control']['mu'][rr[good]]=distance_quantiles(m[good,0],quantiles_interpolated(p[good]/mass[good,None],control.edges))[0]
        w,mean,sigma=candidate.conditional(m,e);w*=candidate.class_frequency[None,:,1];mass=w.sum(axis=1)
        values['candidate']['mass'][rr]=mass;good=mass>0;w=w[good]/mass[good,None]
        edges=np.arange(-30,40.0001,.1);pdf=np.zeros((good.sum(),len(edges)-1))
        for k in range(candidate.components):
            z=(edges[None,:]-mean[good,k,None])/sigma[good,k,None]
            pdf+=w[:,k,None]*normal_interval(z[:,:-1],z[:,1:])
        pdf/=pdf.sum(axis=1)[:,None]
        values['candidate']['mu'][rr[good]]=distance_quantiles(m[good,0],quantiles_interpolated(pdf,edges))[0]
    results=[]
    for name,p in values.items():
        for split,role in [(1,'cal'),(2,'development_test')]:
            for kind in ('chosen','Gaia','specdis','spdist'):
                ref,sigma=references(q,kind);d=10**((ref-10)/5)
                for lo,hi in [(10,20),(20,50)]:
                    eligible=(q['split']==split)&(d>=lo)&(d<hi);use=eligible&(p['mass']>0)
                    results.append(dict(model=name,role=role,reference=kind,low_kpc=lo,high_kpc=hi,eligible=int(eligible.sum()),
                        mean_branch_mass=float(p['mass'][eligible].mean()) if eligible.any() else None,
                        **metrics(p['mu'][use],ref[use])))
    csv_rows(TABLES/'far_conditional.csv',results)
    # Independent of class oracle: same-object reference disagreement retained.
    a,sa=references(q,'specdis');b,sb=references(q,'spdist');r=[]
    for lo,hi in [(10,20),(20,50)]:
        d=10**((a-10)/5);u=(q['split']==2)&np.isfinite(a)&np.isfinite(b)&(d>=lo)&(d<hi)
        delta=a[u]-b[u]
        r.append(dict(low_kpc=lo,high_kpc=hi,n=int(u.sum()),median_specdis_minus_spdist_mu=float(np.median(delta)) if len(delta) else None,
            nmad_mu=float(1.4826*np.median(abs(delta-np.median(delta)))) if len(delta) else None))
    csv_rows(TABLES/'far_reference_disagreement.csv',r)
    controls=[]
    for role,t in [('development',t),('reserved',load(DEST/'reserved_inputs.npz'))]:
        for name in ('control','candidate'):
            p=load(DEST/f'{role}_{name}.npz')
            for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25)]:
                u=t['nonstellar_risk']&(t['r']>=lo)&(t['r']<hi)
                controls.append(dict(role=role,model=name,r_low=lo,r_high=hi,flagged_n=int(u.sum()),near_scope_selected=int((u&p['near_scope']).sum()),
                    broad_q95_r20_selected=int((u&(p['qnear']>.95)&(t['r']<20)).sum()),
                    caveat='Risk-flagged PSF controls, not confirmed nonstellar truth or representative contamination estimate'))
    csv_rows(TABLES/'psf_nonstellar_controls.csv',controls)
    save_json(DEST/'far_complete.json',dict(models=json.loads((DEST/'models_frozen.json').read_text())['models'],
        source_sha256=sha(ROOT/'scripts/v2_3_far.py'),oracle_rows=len(rows),class_conditioning='Diagnostic only; missing label is not a fourth physical class',
        independent_scale=False,source='Existing chosen/Gaia/SpecDis/SPDist only'))
    print('FAR COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
