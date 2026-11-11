#!/usr/bin/env python3
"""Independent empirical response and fresh-catalog population recovery."""
import os
import json
import concurrent.futures
import multiprocessing as mp
import numpy as np
from scipy.spatial.distance import cdist
from v2_1_near import ROOT,load,subset,sha,save_json,csv_rows,atomic_npz
from v2_4_population import joint_model,DISTANCE_EDGES
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.audit import rvs_metallicity
from photdis.population_v2_5 import (DISTANCES,DISTANCE_PRIOR,NCELL,COLOR_EDGES,observation_cell,
    response_from_draws,posterior_grid,summarize_posterior)
from photdis.science_v2_4 import distance_bin_mass
D=ROOT/'data/v2_5';T=D/'tables';_J=_CAT=None


def templates():
    t=load(ROOT/'data/v2_1/development.npz');fit=load(ROOT/'data/v2_3/training.npz')['identity']
    test=load(ROOT/'data/v2_4/recovery_design.npz');u=(test['replica']==-1)&(test['distance_bin']==0)
    testids=test['identity'][u];ix=np.array([np.flatnonzero(t['identity']==i)[0] for i in testids]);test=subset(t,ix)
    feh,valid=rvs_metallicity(t['rvs_feh'],t['rvs_teff'],t['rvs_logg'])
    quality=valid&t['rvs_parameter_quality']&np.isfinite(t['rvs_feh_err'])&(t['rvs_feh_err']>0)
    base=(t['split']==1)&(t['TYPE']=='PSF')&np.isfinite(t['ref_chosen'])&(t['optical_error']<.02)&~np.isin(t['identity'],np.r_[fit,testids])
    masks=[base&(t['class_label']==0)&quality&(feh>=-.5),base&(t['class_label']==0)&quality&(feh< -1.5),base&(t['class_label']==1),base&(t['class_label']==2),base&(t['class_label']<0)]
    chosen=[];labels=[]
    for k,(m,limit) in enumerate(zip(masks,[64,64,48,48,32])):
        rows=np.flatnonzero(m)
        with np.errstate(over='ignore'):rank=t['identity'][rows].astype('u8')*np.uint64(11400714819323198485)
        rows=rows[np.argsort(rank,kind='stable')[:limit]]
        assert len(rows)>=8,(k,len(rows))
        chosen.extend(rows);labels.extend([k]*len(rows))
    cal=subset(t,np.array(chosen));labels=np.array(labels)
    def convert(q):
        m,e,ids=arrays(q,np.arange(len(q['identity'])))
        return dict(identity=ids,TYPE=q['TYPE'],absolute=m[:,:3]-q['ref_chosen'][:,None],error=e[:,:3],sigma=q['sigma_mu'])
    c=convert(cal);e=convert(test)
    old=load(ROOT/'data/v2_4/recovery_design.npz');e['group']=old['group'][u]
    member=np.eye(4)[np.minimum(labels,3)];known=labels<4
    feature=np.column_stack((c['absolute'][:,0]-c['absolute'][:,1],c['absolute'][:,1]-c['absolute'][:,2],c['absolute'][:,0]))
    kernel=np.exp(-.5*cdist(feature[~known]/[.2,.2,1.],feature[known]/[.2,.2,1.])**2)
    kernel+=1e-20;soft=kernel@member[known];soft/=soft.sum(axis=1)[:,None];member[~known]=soft
    c['member']=member;c['label_available']=known
    # Reference-like baseline uses the eligible cal pool, not the quota-selected set.
    count=np.array([m.sum() for m in masks[:4]],float)+.5;basefrac=count/count.sum()
    app=load(ROOT/'data/v2_2/application_inputs.npz');m,err,_=arrays(app,np.arange(len(app['identity'])))
    bins=np.arange(10,27.);curve=[];support=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        take=(app['r']>=lo)&(app['r']<hi)&np.isfinite(err[:,:3]).all(axis=1)&(app['TYPE']=='PSF')
        support.append(int(take.sum()));curve.append(np.nanmedian(err[take,:3],axis=0) if take.any() else [np.nan]*3)
    curve=np.array(curve)
    for k in range(3):
        good=np.isfinite(curve[:,k]);curve[:,k]=np.interp(np.arange(len(curve)),np.flatnonzero(good),curve[good,k])
    curve=np.maximum(curve,.005)
    assert not np.intersect1d(c['identity'],e['identity']).size and not np.intersect1d(c['identity'],fit).size
    assert (c['TYPE']=='PSF').all() and (e['TYPE']=='PSF').all()
    atomic_npz(D/'cal_templates.npz',**c);atomic_npz(D/'test_templates.npz',**e)
    save_json(D/'templates.json',dict(cal=len(chosen),test=len(testids),missing_labels=int((~known).sum()),cal_counts=count.tolist(),base=basefrac.tolist(),native_PSF=True,
        eligible_nonPSF=int(((t['split']==1)&(t['TYPE']!='PSF')).sum()),missing_TYPE=int(np.sum(t['TYPE']=='')),fit_overlap=0,test_overlap=0,
        noise_bin_counts=support,noise_bins=bins.tolist(),noise_curve=curve.tolist(),label_model='Missing label marginalized with fixed cal color/Mg kernel; physical calibration unproven'))
    return c,e,basefrac,bins,curve


def draw(c,group,dist,rng,bins,curve,weights=None,teacher=0):
    member=c.get('member',np.eye(4)[c.get('group',np.zeros(len(c['identity']),int))])
    ix=np.empty(len(group),int)
    for k in range(4):
        u=group==k;w=member[:,k].copy()
        if weights is not None:w*=weights
        assert w.sum()>0;ix[u]=rng.choice(len(w),u.sum(),p=w/w.sum())
    absolute=c['absolute'][ix]-teacher*c['sigma'][ix,None]
    clean=absolute+(10+5*np.log10(DISTANCES[dist]))[:,None]
    slot=np.clip(np.searchsorted(bins,clean[:,1])-1,0,len(curve)-1)
    error=np.maximum(curve[slot],c['error'][ix]);m=clean+rng.normal(size=clean.shape)*error
    selected=(m[:,1]>=14)&(m[:,1]<20)&(error.max(axis=1)<.1)
    return m,error,selected,ix


def response(c,bins,curve,seed=20261111,weights=None,teacher=0):
    comp=np.repeat(np.arange(20),2048);group=comp//5;dist=comp%5
    m,e,sel,ix=draw(c,group,dist,np.random.default_rng(seed),bins,curve,weights,teacher)
    cell=observation_cell(m)
    rindex=np.unravel_index(np.arange(NCELL),tuple(len(e)-1 for e in COLOR_EDGES))[-1]
    allowed=(COLOR_EDGES[-1][rindex]>=14)&(COLOR_EDGES[-1][rindex+1]<=20)
    return [response_from_draws(cell,comp,sel,allowed_cells=allowed),response_from_draws(cell,comp,~sel)]


def populations(base):
    old={'ordinary_dominated':([.72,.08,.05,.15],0),'giant_rich':([.45,.05,.35,.15],0),
         'metal_poor_rich':([.32,.48,.05,.15],0),'balanced':([.7/3,.3/3,1/3,1/3],0)}
    giant=base.copy();giant*=.65/(1-base[2]);giant[2]=.35
    poor=base.copy();poor[0:2]=base[:2].sum()*np.array([.4,.6])
    return {'reference_like':(base,0),'giant_only':(giant,0),'poor_only':(poor,0),'distance_only':(base,.6),**old}


def catalogs(test,pops,bins,curve):
    rows=[];mags=[];errors=[];rng=np.random.default_rng(251111)
    for design,(fraction,tilt) in pops.items():
        dp=DISTANCE_PRIOR*(DISTANCES/5)**tilt;dp/=dp.sum()
        for regime in range(2):
            for repeat in range(12):
                need=128;attempt=0
                while need:
                    g=rng.choice(4,512,p=fraction);d=rng.choice(5,512,p=dp)
                    m,e,sel,ix=draw(test,g,d,rng,bins,curve);keep=np.flatnonzero(sel if regime==0 else ~sel)[:need]
                    for j in keep:rows.append(dict(catalog=len(rows)//128,design=design,regime=regime,repeat=repeat,identity=int(test['identity'][ix[j]]),TYPE='PSF',group=int(g[j]),distance_bin=int(d[j])))
                    mags.extend(m[keep]);errors.extend(e[keep]);need-=len(keep);attempt+=1;assert attempt<100
    out={k:np.asarray([r[k] for r in rows]) for k in rows[0]};out['mags']=np.array(mags);out['errors']=np.array(errors)
    atomic_npz(D/'catalogs.npz',**out);return out


def qrf_chunk(rows):
    path=D/'qrf_chunks'/f'{rows[0]:07}.npz'
    if path.exists():
        assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256'];return str(path)
    m=np.full((len(rows),5),np.nan);e=m.copy();m[:,:3]=_CAT['mags'][rows];e[:,:3]=_CAT['errors'][rows]
    p=_J.predict(m,e,_CAT['identity'][rows]);mass=distance_bin_mass(p['joint'],_J.model.edges,m[:,0],DISTANCE_EDGES)
    atomic_npz(path,rows=rows,identity=_CAT['identity'][rows],TYPE=_CAT['TYPE'][rows],joint_mass=mass,mu=p['mu'])
    save_json(path.with_suffix('.json'),dict(sha256=sha(path)));return str(path)


def evaluate(c,test,base,bins,curve,cat,p):
    responses=response(c,bins,curve);results=[];diagnostics=[];raw=[];posteriors=[];sensitivity=[]
    ids=np.unique(cat['catalog']);counts=np.array([np.bincount(observation_cell(cat['mags'][cat['catalog']==i]),minlength=NCELL) for i in ids])
    for regime in range(2):
        r=responses[regime];chosen=[i for i in ids if cat['regime'][cat['catalog']==i][0]==regime];cc=counts[chosen]
        theta,selected,weights=posterior_grid(cc,r,base,power=14)
        _,small,sw=posterior_grid(cc,r,base,power=13)
        atomic_npz(D/f'posterior_regime{regime}.npz',catalog=np.array(chosen),theta=theta,selected=selected,weights=weights,response=r)
        # Independent test response quantifies the population target; no test
        # response enters the likelihood used to fit these same catalogs.
        tr=response(test,bins,curve,seed=251114)[regime]
        for k,i in enumerate(chosen):
            u=cat['catalog']==i;design=str(cat['design'][u][0]);fraction,tilt=populations(base)[design];fraction=np.asarray(fraction)
            dp=DISTANCE_PRIOR*(DISTANCES/5)**tilt;dp/=dp.sum();eta=(fraction[:,None]*dp).ravel()
            truth=eta*tr.sum(axis=1);truth/=truth.sum();truth=truth.reshape(4,5)
            mean,interval,diag=summarize_posterior(theta,selected,weights[k]);mean=mean.reshape(4,5)
            convergence=float(np.max(abs(weights[k]@selected-sw[k]@small)))
            diagnostics.append(dict(catalog=int(i),design=design,regime=regime,repeat=int(cat['repeat'][u][0]),ess=diag['ess'],boundary=diag['boundary'],max_correlation=diag['max_abs_correlation'],grid_max_delta=convergence,
                min_alpha=float(r.sum(axis=1).min()),max_alpha=float(r.sum(axis=1).max()),missing_label_fraction=float((~c['label_available']).mean())))
            point=np.clip(np.searchsorted(DISTANCE_EDGES,10**((p['mu'][u,2]-10)/5),side='right')-1,0,5)
            qpdf=p['joint_mass'][u,:4].sum(axis=1).sum(axis=0)/128
            qpoint=np.bincount(point,minlength=6)/128
            realized=np.bincount(cat['distance_bin'][u],minlength=5)/128
            for j in range(5):
                # Posterior predictive count interval adds multinomial variation.
                rng=np.random.default_rng(25000+i*10+j);drawidx=rng.choice(len(weights[k]),4096,p=weights[k]);v=selected[drawidx].reshape(-1,4,5).sum(axis=1)[:,j]
                pred=np.quantile(rng.binomial(128,v)/128,[.025,.975])
                results.append(dict(catalog=int(i),design=design,regime=regime,bin=j,truth=float(truth[:,j].sum()),realized=float(realized[j]),
                    point=float(qpoint[j]),pdf=float(qpdf[j]),population=float(mean[:,j].sum()),low=float(interval[4+j,0]),high=float(interval[4+j,2]),predictive_low=float(pred[0]),predictive_high=float(pred[1])))
            for j in range(4):
                raw.append(dict(catalog=int(i),design=design,regime=regime,group=j,truth=float(truth[j].sum()),realized=float(np.mean(cat['group'][u]==j)),population=float(mean[j].sum()),
                    low=float(interval[j,0]),high=float(interval[j,2]),qrf_reference=float(p['joint_mass'][u,j if j<2 else j-1].sum()/128) if j>=2 else None))
        # Finite-template bootstrap and teacher-scale/imputation sensitivity,
        # evaluated on the FIRST reference-like catalog only (never tune settings).
        first=next(i for i in chosen if cat['design'][cat['catalog']==i][0]=='reference_like');count=counts[first]
        variants=[('teacher_minus',None,-1),('teacher_plus',None,1),('labeled_only',c['label_available'].astype(float),0)]
        rng=np.random.default_rng(251116)
        variants += [(f'template_bootstrap_{j}',rng.exponential(size=len(c['identity'])),0) for j in range(16)]
        for label,w,shift in variants:
            rr=response(c,bins,curve,weights=w,teacher=shift)[regime]
            _,ss,ww=posterior_grid(count,rr,base,power=12)
            value=(ww@ss).reshape(4,5)
            for j in range(5):sensitivity.append(dict(regime=regime,variant=label,bin=j,estimate=float(value[:,j].sum())))
    csv_rows(T/'recovery_distance.csv',results);csv_rows(T/'recovery_group.csv',raw);csv_rows(T/'identifiability.csv',diagnostics);csv_rows(T/'uncertainty_sources.csv',sensitivity)
    save_json(D/'population_complete.json',dict(catalogs=len(ids),rows=len(cat['identity']),cal_templates=len(c['identity']),test_templates=len(test['identity']),job=os.getenv('SLURM_JOB_ID'),
        template_overlap=0,fit_overlap=0,PSF=True,selection='14<=observed dereddened r<20, maximum assigned error<.1; complementary stress separate',
        caveats='Discrete distances; factorized four-parameter population; teacher-scaled SEDs; empirical noise law; no detector selection;12independent catalogs per scenario; finite test-library MC truth. No adoption or physical application counts.'))


def main():
    global _J,_CAT
    require_compute();T.mkdir(parents=True,exist_ok=True);(D/'qrf_chunks').mkdir(exist_ok=True)
    files=[ROOT/'scripts/v2_5_population.py',ROOT/'src/photdis/population_v2_5.py',ROOT/'docs/v2_5_protocol.md',ROOT/'docs/v2_5_response_support_amendment.md',ROOT/'data/v2_3/control.joblib',ROOT/'data/v2_4/recovery_design.npz',ROOT/'data/v2_1/development.npz']
    freeze={str(x.relative_to(ROOT)):sha(x) for x in files};marker=D/'population_freeze.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old!=freeze:
            key='scripts/v2_5_population.py'
            assert old[key]=='4ad00fe196184056a885a4739117059def6604a23aed71df7fd30ea13f62c4bb'
            assert all(old[k]==v for k,v in freeze.items() if k!=key)
            assert not (D/'population_complete.json').exists()
            save_json(D/'population_writer_migration.json',dict(previous=old,replacement=freeze,
                reason='Convert scenario list to ndarray in truth-summary writer; reuse completed templates/catalogs/QRF. No prediction or model change.',
                catalogs_sha256=sha(D/'catalogs.npz'),cal_sha256=sha(D/'cal_templates.npz'),test_sha256=sha(D/'test_templates.npz')))
            save_json(marker,freeze)
    else:save_json(marker,freeze)
    if (D/'templates.json').exists():
        meta=json.loads((D/'templates.json').read_text());c=load(D/'cal_templates.npz');test=load(D/'test_templates.npz')
        base=np.array(meta['base']);bins=np.array(meta['noise_bins']);curve=np.array(meta['noise_curve'])
    else:c,test,base,bins,curve=templates()
    _CAT=load(D/'catalogs.npz') if (D/'catalogs.npz').exists() else catalogs(test,populations(base),bins,curve)
    _J=joint_model();tasks=[np.arange(i,min(i+64,len(_CAT['identity']))) for i in range(0,len(_CAT['identity']),64)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:paths=list(pool.map(qrf_chunk,tasks))
    parts=[load(x) for x in paths];p={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    assert np.array_equal(p['identity'],_CAT['identity']);np.testing.assert_allclose(p['joint_mass'][:,:4].sum(axis=(1,2)),1,atol=2e-6)
    print('FRESH QRF CATALOGS COMPLETE',len(_CAT['identity']),flush=True)
    evaluate(c,test,base,bins,curve,_CAT,p);print('POPULATION COMPLETE',flush=True)


if __name__=='__main__':main()
