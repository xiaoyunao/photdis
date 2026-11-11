#!/usr/bin/env python3
"""Outcome-blind matched red controls and signed-parallax metal-pair audit."""
import json
import os
import numpy as np
from scipy.spatial import cKDTree
from v2_1_near import ROOT,load,subset,sha,save_json,csv_rows,atomic_npz
from v2_4_audit import pdf_rows
from v2_3_diagnostics import references
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.science_v2_4 import cluster_mean_interval
from photdis.baseline_parallax import parallax_check
from photdis.baseline_v1_2 import full_parallax_log_density
D=ROOT/'data/v2_5';T=D/'tables'


def weighted_interval(v,w,cluster):
    v=np.asarray(v);w=np.asarray(w);_,j=np.unique(cluster,return_inverse=True)
    a=np.bincount(j,weights=v*w);b=np.bincount(j,weights=w);rng=np.random.default_rng(251111)
    if len(a)<2:return dict(value=float(a.sum()/b.sum()),low=None,high=None,clusters=len(a))
    ix=rng.integers(len(a),size=(1000,len(a)));boot=a[ix].sum(axis=1)/b[ix].sum(axis=1)
    lo,hi=np.quantile(boot,[.025,.975]);return dict(value=float(a.sum()/b.sum()),low=lo,high=hi,clusters=len(a))


def red():
    old=ROOT/'data/v2_3';app=load(ROOT/'data/v2_2/application_inputs.npz');joint=load(ROOT/'data/v2_4/application_joint_summary.npz');fit=load(old/'training.npz')['identity']
    h=load(old/'reserved_inputs.npz');hp=load(old/'reserved_control.npz');hs=load(old/'reserved_control_selection.npz');cond=load(old/'selection_conditions.npz')
    hr=np.flatnonzero(hp['near_scope']&cond['supported']);assert len(hr)==711
    rr=joint['rows'];par=np.full(len(rr),np.nan);err=par.copy();risk=np.zeros(len(rr),bool)
    for field in np.unique(app['field'][rr]):
        u=app['field'][rr]==field;rows=app['parent_row'][rr[u]]
        with np.load(ROOT/f'data/baseline_v0/parent_{field:02}.npz') as z:
            assert np.array_equal(z['identity'][rows],app['identity'][rr[u]])
            par[u]=z['p_corr'][rows];err[u]=z['gaia_parallax_error'][rows];risk[u]=z['nonstellar_risk'][rows]
    eligible=~np.isin(app['identity'][rr],fit)&np.isfinite(joint['gaia_pit'])&~risk&(app['TYPE'][rr]=='PSF')
    def features(t,r,ee):return np.column_stack((t['g0'][r]-t['r0'][r],t['r0'][r]-t['z0'][r],t['r'][r],np.log10(np.maximum(t['optical_error'][r],1e-8)),np.log10(ee)))
    scale=np.array([.2,.2,1,.5,.5]);hx=features(h,hr,h['gaia_parallax_error'][hr]);ax=features(app,rr,err)
    eligible &=np.isfinite(ax).all(axis=1);pool=np.flatnonzero(eligible)
    dist,k=cKDTree(ax[pool]/scale).query(hx/scale);selected=pool[k];matched=dist<=2
    records=[]
    for i,r in enumerate(hr):
        a=selected[i];records.append(dict(historical_identity=int(h['identity'][r]),TYPE='PSF',red=bool(hx[i,1]>=.8),matched=bool(matched[i]),match_distance=dist[i],
            application_identity=int(app['identity'][rr[a]]),field=int(app['field'][rr[a]]),weight=float(app['sample_weight'][rr[a]]),
            historical_raw_pit=hs['raw_pit32'][r],historical_conditional_pit=hs['conditional_pit'][r],application_raw_pit=joint['gaia_pit'][a],
            historical_parallax=h['p_corr'][r],application_parallax=par[a],historical_error=h['gaia_parallax_error'][r],application_error=err[a],
            historical_gr=hx[i,0],application_gr=ax[a,0],historical_rz=hx[i,1],application_rz=ax[a,1],historical_r=hx[i,2],application_r=ax[a,2]))
    csv_rows(T/'red_matches.csv',records);summary=[]
    for group,g in [('all',np.ones(len(hr),bool)),('red',hx[:,1]>=.8),('blue',hx[:,1]<.8)]:
        use=g&matched;a=selected[use];r=hr[use]
        for name,v in [('historical_raw',hs['raw_pit32'][r]),('historical_conditional',hs['conditional_pit'][r]),('application_raw',joint['gaia_pit'][a])]:
            for weighting,w in [('matched_identity',np.ones(len(a))),('parent_sampling',app['sample_weight'][rr[a]])]:
                for tail,event in [('low',v<.05),('high',v>.95)]:
                    for cluster,c in [('reused_control',app['identity'][rr[a]]),('field',app['field'][rr[a]])]:
                        if len(a):summary.append(dict(group=group,method=name,weighting=weighting,tail=tail,cluster=cluster,denominator=int(g.sum()),matched=len(a),unique_controls=len(np.unique(a)),**weighted_interval(event,w,c)))
    csv_rows(T/'red_migration.csv',summary)
    save_json(D/'red_complete.json',dict(denominator=711,matched=int(matched.sum()),unmatched=int((~matched).sum()),application_pool=len(pool),unique_controls=int(len(np.unique(selected[matched]))),selection_uses_PIT=False,nonPSF_excluded=int((app['TYPE'][rr]!='PSF').sum()),missingTYPE_excluded=int((app['TYPE'][rr]=='').sum()),
        interpretation='Controls matched to full711, not low147. Parent-sampling and matched-identity estimands differ; historical conditional vs application raw PIT are NOT same selection.'))


def metal():
    old=ROOT/'data/v2_3';t=load(ROOT/'data/v2_1/development.npz');pairs=load(old/'metal_poor_pairs.npz');fit=load(old/'training.npz')['identity']
    a=unique_join(pairs['poor_identity'],t['identity']);b=unique_join(pairs['control_identity'],t['identity']);assert (a>=0).all() and (b>=0).all() and len(a)==2055
    eligible=t['parallax_eligible']&np.isfinite(t['p_corr'])&np.isfinite(t['gaia_parallax_error'])&(t['gaia_parallax_error']>0)&(t['TYPE']=='PSF')&~t['nonstellar_risk']
    eligible &=~t.get('identity_conflict',np.zeros(len(eligible),bool))
    union=np.unique(np.r_[a[eligible[a]],b[eligible[b]]]);objects=[];summaries=[];availability=[]
    for side,r in [('poor',a),('ordinary',b)]:availability.append(dict(side=side,original_pairs=2055,eligible=int(eligible[r].sum()),unique_eligible=len(np.unique(r[eligible[r]])),negative=int(np.sum(eligible[r]&(t['p_corr'][r]<0))),weak=int(np.sum(eligible[r]&(t['p_corr'][r]/t['gaia_parallax_error'][r]<5))),fit_overlap=int(np.sum(eligible[r]&np.isin(t['identity'][r],fit)))))
    for name in ('control','candidate'):
        pdf,edges=pdf_rows('development',name,union);args=(pdf,edges,t['g0'][union],t['p_corr'][union],t['gaia_parallax_error'][union])
        pit=parallax_check(*args,np.full(len(union),-np.inf),np.full(len(union),np.inf),order=32)['pit'];log=full_parallax_log_density(*args,order=32)
        pp=np.full(len(t['identity']),np.nan);ll=pp.copy();pp[union]=pit;ll[union]=log
        for i,(aa,bb) in enumerate(zip(a,b)):
            for side,r in [('poor',aa),('ordinary',bb)]:
                if eligible[r]:objects.append(dict(pair=i,model=name,side=side,identity=int(t['identity'][r]),control_identity=int(t['identity'][bb]),TYPE='PSF',cell=int(t['cell'][r]),fitted=bool(np.isin(t['identity'][r],fit)),label_source=int(t['label_source'][r]),parallax=t['p_corr'][r],error=t['gaia_parallax_error'][r],pit=pp[r],log_score=ll[r],both_eligible=bool(eligible[aa]&eligible[bb])))
        both=eligible[a]&eligible[b]
        for source in ('all','specdis','spdist','both_VAC'):
            use=both.copy()
            for s in (['specdis','spdist'] if source=='both_VAC' else [source]):
                if s!='all':ref,_=references(t,s);use &=np.isfinite(ref[a])&np.isfinite(ref[b])
            for role in ('all','neither_fitted'):
                uu=use.copy()
                if role=='neither_fitted':uu &=~np.isin(t['identity'][a],fit)&~np.isin(t['identity'][b],fit)
                aa=a[uu];bb=b[uu]
                for metric,delta in [('log_score',ll[aa]-ll[bb]),('low_tail',(pp[aa]<.05).astype(float)-(pp[bb]<.05)),('high_tail',(pp[aa]>.95).astype(float)-(pp[bb]>.95))]:
                    for cluster,c in [('control',t['identity'][bb]),('sky_pair',np.char.add(np.char.add(t['cell'][aa].astype(str),':'),t['cell'][bb].astype(str)))]:
                        summaries.append(dict(model=name,source=source,role=role,metric=metric,cluster=cluster,n=len(aa),unique_controls=len(np.unique(bb)),**cluster_mean_interval(delta,c)))
    csv_rows(T/'metal_gaia_objects.csv',objects);csv_rows(T/'metal_gaia_pairs.csv',summaries);csv_rows(T/'metal_gaia_availability.csv',availability)


def information():
    t=load(ROOT/'data/v2_1/information_eval.npz');p=load(ROOT/'data/v2_1/information_probabilities.npz');assert np.array_equal(t['identity'],p['identity'])
    rows=[]
    for left,right in [('grz','griz'),('grz','grzW1'),('grzW1','grzW1W2')]:
        for method in ('frozen_partitions_same_labels','fixed_forest'):
            for c in range(3):
                use=(t['split']==2)&(t['class_label']==c)&(t['TYPE']=='PSF');y=t['class_label'][use].astype(int)
                val=np.log(np.maximum(p[right+'_'+method][use,y],1e-12))-np.log(np.maximum(p[left+'_'+method][use,y],1e-12))
                rows.append(dict(left=left,right=right,method=method,group=c,n=int(use.sum()),faint_r20=int(np.sum(use&(t['r']>=20))),**cluster_mean_interval(val,t['cell'][use])))
    csv_rows(T/'band_gain_by_class.csv',rows)


def main():
    require_compute();T.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'scripts/v2_5_diagnostics.py',ROOT/'docs/v2_5_protocol.md',ROOT/'data/v2_3/control.joblib',ROOT/'data/v2_3/candidate.joblib',ROOT/'data/v2_4/prospective_identity_reservation.npz']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in paths};marker=D/'diagnostic_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    red();print('RED COMPLETE',flush=True);metal();print('METAL COMPLETE',flush=True);information()
    save_json(D/'diagnostics_complete.json',dict(job=os.getenv('SLURM_JOB_ID'),reservation_unopened=True,all_PSF=True));print('DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':main()
