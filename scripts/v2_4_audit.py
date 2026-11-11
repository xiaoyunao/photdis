#!/usr/bin/env python3
"""Allocated same-identity failure, teacher-scale and information diagnostics."""
import argparse
import json
import os
import numpy as np
from v2_1_near import ROOT,load,subset,sha,save_json,csv_rows,atomic_npz
from v2_3_diagnostics import references,masks
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.science_v2_4 import pdf_reference_shape,cluster_mean_interval,conditional_prior,weighted_scores
OLD=ROOT/'data/v2_3';DEST=ROOT/'data/v2_4';TABLES=DEST/'tables'


def setup():
    require_compute();TABLES.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/'docs/v2_4_protocol.md',ROOT/'scripts/v2_4_audit.py',ROOT/'src/photdis/science_v2_4.py',
            OLD/'models_frozen.json',OLD/'control.joblib',OLD/'candidate.joblib']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    path=DEST/'audit_freeze.json'
    if path.exists():assert json.loads(path.read_text())==freeze,'Explicit audit migration required'
    else:save_json(path,freeze)


def pdf_rows(role,name,rows):
    rows=np.asarray(rows,int);grid=load(OLD/(name+'_grid.npz'))['edges']
    out=np.empty((len(rows),len(grid)-1));found=np.zeros(len(rows),bool)
    for start in np.unique(rows//256*256):
        path=OLD/'chunks'/f'{role}_{name}_{start:07}.npz'
        meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['sha256']
        p=load(path);u=(rows>=start)&(rows<start+256)
        pos=rows[u]-start;out[u]=p['pdf'][pos];found[u]=True
    assert found.all();np.testing.assert_allclose(out.sum(axis=1),1,atol=2e-6)
    return out,grid


def anomalies():
    t=load(OLD/'reserved_inputs.npz');p={n:load(OLD/f'reserved_{n}.npz') for n in ('control','candidate')}
    sel={n:load(OLD/f'reserved_{n}_selection.npz') for n in p};cond=load(OLD/'selection_conditions.npz')
    base=p['control']['near_scope']&cond['supported'];bad=base&(sel['control']['conditional_pit']<.05)
    sp,sp_sigma=references(t,'spdist');spnear=p['control']['near_scope']&np.isfinite(sp)
    assert base.sum()==711 and bad.sum()==147 and spnear.sum()==399
    rows=np.flatnonzero(base);q=subset(t,rows);assert (q['TYPE']=='PSF').all()
    rowrecords=[];shape={};pdfs={}
    for name in p:
        pdf,edges=pdf_rows('reserved',name,rows);pdfs[name]=(pdf,edges)
        shape[name]=pdf_reference_shape(pdf,edges,q['g0'],sp[rows])
    # Recover exact original Gaia observation rows, not a nearest-sky substitute.
    import fitsio
    config=json.loads((ROOT/'configs/baseline_v1_4.json').read_text())
    columns=['SOURCE_ID','RUWE','VISIBILITY_PERIODS_USED','IPD_FRAC_MULTI_PEAK','IPD_GOF_HARMONIC_AMPLITUDE','DUPLICATED_SOURCE','ASTROMETRIC_PARAMS_SOLVED']
    quality={k:np.full(len(rows),np.nan) for k in columns[1:]};valid=q['source_row']>=0
    with fitsio.FITS(config['mws_path']) as f:
        ga=f['GAIA'].read(rows=q['source_row'][valid],columns=columns)
    assert np.array_equal(ga['SOURCE_ID'],q['source_id'][valid])
    for k in quality:quality[k][valid]=ga[k]
    for i,r in enumerate(rows):
        rec=dict(identity=int(t['identity'][r]),TYPE=str(t['TYPE'][r]),low_pit=bool(bad[r]),W1=bool(t['B'][r]),cell=int(t['cell'][r]),
            ra=t['RA'][r],dec=t['DEC'][r],r=t['r'][r],gr=t['g0'][r]-t['r0'][r],rz=t['r0'][r]-t['z0'][r],optical_error=t['optical_error'][r],
            parallax=t['p_corr'][r],parallax_error=t['gaia_parallax_error'][r],snr=t['p_corr'][r]/t['gaia_parallax_error'][r],
            SPDist_qualified=bool(spnear[r]),SPDist_mu=sp[r],SPDist_sigma=sp_sigma[r],label_source=int(t['label_source'][r]),
            specdis_valid=bool(t['specdis_valid'][r]),spdist_source_only=bool(t['spdist_source_only_join'][r]),
            **{k:float(v[i]) for k,v in quality.items()})
        for name in p:
            rec.update({name+'_'+k:float(sel[name][k][r]) for k in ('raw_log32','raw_pit32','conditional_log','conditional_pit')})
            rec.update({name+'_mu50':float(p[name]['mu'][r,2]),name+'_qnear':float(p[name]['qnear'][r]),name+'_near_scope':bool(p[name]['near_scope'][r])})
            rec.update({name+'_'+k:float(v[i]) for k,v in shape[name].items()})
        if spnear[r]:
            expected=10**((10-sp[r])/5);error=np.hypot(t['gaia_parallax_error'][r],expected*np.log(10)/5*sp_sigma[r])
            rec['Gaia_minus_SPDist_sigma']=(t['p_corr'][r]-expected)/error
        rowrecords.append(rec)
    csv_rows(TABLES/'anomaly_objects.csv',rowrecords)
    groups={'all':np.ones(len(rows),bool),'no_W1':~q['B'],'W1':q['B'],'negative':q['p_corr']<0,'nonnegative':q['p_corr']>=0,
            'SPDist':spnear[rows],'no_SPDist':~spnear[rows]}
    variables=[('r',q['r'],[-np.inf,18,19,20,np.inf]),('gr',q['g0']-q['r0'],[-np.inf,.3,.8,1.3,np.inf]),
        ('rz',q['r0']-q['z0'],[-np.inf,.2,.5,.8,np.inf]),('optical_error',q['optical_error'],[0,.01,.03,.1,np.inf]),
        ('gaia_error',q['gaia_parallax_error'],[0,.03,.1,.3,np.inf]),('RUWE',quality['RUWE'],[0,1,1.2,1.4,np.inf]),
        ('visibility',quality['VISIBILITY_PERIODS_USED'],[0,10,15,20,np.inf]),('ipd_multi',quality['IPD_FRAC_MULTI_PEAK'],[-1,1,5,100,np.inf])]
    for key,val,edges in variables:
        for lo,hi in zip(edges[:-1],edges[1:]):groups[f'{key}_{lo}_{hi}']=(val>=lo)&(val<hi)
        groups[key+'_missing']=~np.isfinite(val)
    for c in np.unique(q['cell']):groups['cell_'+str(c)]=q['cell']==c
    for c in np.unique(q['rvs_program']):groups['program_'+str(c)]=q['rvs_program']==c
    summary=[]
    for label,u in groups.items():
        for name in p:
            rr=rows[u];v=sel[name]['conditional_pit'][rr]
            summary.append(dict(group=label,model=name,n=len(rr),low=int((v<.05).sum()),negative=int((q['p_corr'][u]<0).sum()),
                low_fraction=float((v<.05).mean()) if len(v) else None,mean_log=float(sel[name]['conditional_log'][rr].mean()) if len(rr) else None,
                qrf_low=int(bad[rr].sum()),same_direction_low=int((bad[rr]&(v<.05)).sum())))
    csv_rows(TABLES/'anomaly_strata.csv',summary)
    overlap=[]
    for label,u in [('all711',base),('low147',bad),('remaining564',base&~bad)]:
        rr=np.flatnonzero(u&spnear)
        for name in p:
            delta=sp[rr]-p[name]['mu'][rr,2]
            overlap.append(dict(group=label,model=name,denominator=int(u.sum()),SPDist_intersection=len(rr),
                SPDist_farther=int((delta>0).sum()),SPDist_farther1mag=int((delta>1).sum()),median_SPDist_minus_phot=float(np.median(delta)) if len(rr) else None,
                negative=int((t['p_corr'][rr]<0).sum())))
    csv_rows(TABLES/'anomaly_overlap.csv',overlap)
    # Deterministic examples: lowest identities in each existing subgroup.
    examples=[]
    for group,u in [('low_SPDist',bad[rows]&spnear[rows]),('low_no_SPDist',bad[rows]&~spnear[rows]),('not_low_SPDist',~bad[rows]&spnear[rows])]:
        ix=np.flatnonzero(u);ix=ix[np.argsort(q['identity'][ix])[:4]]
        for i in ix:
            for name,(pdf,edges) in pdfs.items():
                # Rebin exactly to0.1mag for display; scientific scores use native grid.
                centers=(edges[:-1]+edges[1:])/2;mu=q['g0'][i]-centers
                hist,bins=np.histogram(mu,bins=np.arange(-5,30.01,.1),weights=pdf[i])
                for j,val in enumerate(hist):examples.append(dict(group=group,identity=int(q['identity'][i]),model=name,mu=(bins[j]+bins[j+1])/2,density=val/.1,SPDist_mu=sp[rows[i]],p_corr=q['p_corr'][i],p_error=q['gaia_parallax_error'][i]))
    csv_rows(TABLES/'anomaly_pdf_examples.csv',examples)
    save_json(DEST/'anomalies_complete.json',dict(n=711,low=147,SPDist=399,intersection=int((base&spnear).sum()),low_intersection=int((bad&spnear).sum()),
        identical_QRF_XD_rows=True,full_denominator=True,Gaia_quality_rows=int(valid.sum()),not_independent_distance_truth=True,job=os.getenv('SLURM_JOB_ID')))
    print('ANOMALIES COMPLETE',int((bad&spnear).sum()),flush=True)


def metal():
    t=load(ROOT/'data/v2_1/development.npz');pairs=load(OLD/'metal_poor_pairs.npz')
    a=unique_join(pairs['poor_identity'],t['identity']);b=unique_join(pairs['control_identity'],t['identity']);assert (a>=0).all() and (b>=0).all()
    assert len(a)==2055 and (t['TYPE'][np.r_[a,b]]=='PSF').all()
    union=np.unique(np.r_[a,b]);j={int(v):i for i,v in enumerate(union)};ia=np.array([j[v] for v in a]);ib=np.array([j[v] for v in b]);rows=[];objects=[]
    for name in ('control','candidate'):
        p=load(OLD/f'development_{name}.npz');pdf,edges=pdf_rows('development',name,union)
        for refname in ('Gaia','specdis','spdist'):
            ref,sigma=references(t,refname);d=10**((ref-10)/5);sh=pdf_reference_shape(pdf,edges,t['g0'][union],ref[union])
            for lo,hi in [(0,2),(2,5),(5,10),(0,np.inf)]:
                use=np.isfinite(ref[a])&np.isfinite(ref[b])&(d[a]>=lo)&(d[a]<hi)&(d[b]>=lo)&(d[b]<hi)
                aa=a[use];bb=b[use];delta=(p['mu'][aa,2]-ref[aa])-(p['mu'][bb,2]-ref[bb])
                mg=(t['g0'][aa]-ref[aa])-(t['g0'][bb]-ref[bb]);cell=np.char.add(np.char.add(t['cell'][aa].astype(str),':'),t['cell'][bb].astype(str))
                row=dict(model=name,reference=refname,low_kpc=lo,high_kpc=hi,n=len(aa),unique_controls=len(np.unique(bb)),unique_poor=len(np.unique(aa)),
                    median_reference_Mg_delta=float(np.median(mg)) if len(mg) else None,median_delta_mu=float(np.median(delta)) if len(delta) else None)
                for cluster,ids in [('control',t['identity'][bb]),('sky_pair',cell)]:
                    row.update({cluster+'_'+k:v for k,v in cluster_mean_interval(delta,ids).items()})
                for side,rr,index in [('poor',aa,ia[use]),('ordinary',bb,ib[use])]:
                    x=10**((p['mu'][rr,2]-ref[rr])/5)-1
                    row.update({side+'_median_relative':float(np.median(x)) if len(x) else None,side+'_mean_mass1':float(np.mean(sh['mass_within1'][index])) if len(x) else None,
                        side+'_reference_peak_fraction':float(np.mean(sh['peak_near_reference'][index])) if len(x) else None})
                rows.append(row)
            good=np.isfinite(ref[a])&np.isfinite(ref[b])
            for k in np.flatnonzero(good):objects.append(dict(model=name,reference=refname,poor_identity=int(t['identity'][a[k]]),ordinary_identity=int(t['identity'][b[k]]),
                poor_cell=int(t['cell'][a[k]]),ordinary_cell=int(t['cell'][b[k]]),poor_ref_Mg=t['g0'][a[k]]-ref[a[k]],ordinary_ref_Mg=t['g0'][b[k]]-ref[b[k]],
                poor_ref_mu=ref[a[k]],ordinary_ref_mu=ref[b[k]],poor_mass1=sh['mass_within1'][ia[k]],ordinary_mass1=sh['mass_within1'][ib[k]],
                poor_mode_delta=sh['mode_mu'][ia[k]]-ref[a[k]],ordinary_mode_delta=sh['mode_mu'][ib[k]]-ref[b[k]],
                poor_peak=bool(sh['peak_near_reference'][ia[k]]),ordinary_peak=bool(sh['peak_near_reference'][ib[k]])))
    csv_rows(TABLES/'metal_source_pairs.csv',rows);csv_rows(TABLES/'metal_pair_objects.csv',objects)
    save_json(DEST/'metal_complete.json',dict(original_pairs=2055,unique_controls=int(len(np.unique(b))),source_rows=len(rows),bootstrap=1000,
        uncertainty='Separate repeated-control and sky-pair cluster bootstrap intervals; no IID independence or causal attribution.',job=os.getenv('SLURM_JOB_ID')))
    print('METAL COMPLETE',flush=True)


def information():
    t=load(ROOT/'data/v2_1/information_eval.npz');p=load(ROOT/'data/v2_1/information_probabilities.npz')
    assert np.array_equal(t['identity'],p['identity']) and (t['TYPE']=='PSF').all()
    cal=t['split']==1;dev=t['split']==2;y=t['class_label'].astype(int)
    cells=t['cell']*10+np.digitize(t['r'],[18,20,22,25]);prior,rsky=conditional_prior(y[cal],cells[cal],cells)
    result=[];reliability=[];gains=[]
    groups={'all':np.ones(len(y),bool),'r_lt18':t['r']<18,'r18_20':(t['r']>=18)&(t['r']<20),'r20plus':t['r']>=20}
    for name in ('grz','griz','grzW1','grzW1W2'):
        methods={'cal_global_prior':np.tile(prior,(len(y),1)),'cal_r_sky_prior':rsky,
            'frozen_partitions':p[name+'_frozen_partitions_same_labels'],'fixed_color_probe':p[name+'_fixed_forest']}
        for group,mask in groups.items():
            u=dev&mask
            if not u.any():continue
            for population in ('natural_reference','equal_class'):
                count=np.bincount(y[u],minlength=3);w=np.ones(u.sum()) if population=='natural_reference' else 1/np.maximum(count[y[u]],1)
                for method,prob in methods.items():
                    stats=weighted_scores(prob[u],y[u],w);base=weighted_scores(rsky[u],y[u],w)
                    result.append(dict(bands=name,group=group,population=population,method=method,n=int(u.sum()),**stats,gain_r_sky=stats['log_score']-base['log_score'],
                        dwarf=int(count[0]),evolved=int(count[1]),other=int(count[2])))
                    if population=='natural_reference':
                        val=np.log(np.maximum(prob[u,y[u]],1e-12))-np.log(np.maximum(rsky[u,y[u]],1e-12))
                        gains.append(dict(bands=name,group=group,method=method,**cluster_mean_interval(val,t['cell'][u])))
                        for c in range(3):
                            bins=np.minimum((prob[u,c]*10).astype(int),9)
                            for k in range(10):
                                v=bins==k
                                reliability.append(dict(bands=name,group=group,method=method,class_label=c,bin=k,n=int(v.sum()),mean_probability=float(prob[u,c][v].mean()) if v.any() else None,frequency=float((y[u][v]==c).mean()) if v.any() else None))
    csv_rows(TABLES/'information_population.csv',result);csv_rows(TABLES/'information_uncertainty.csv',gains);csv_rows(TABLES/'information_reliability.csv',reliability)
    pairs=[]
    for left,right in [('grz','griz'),('grz','grzW1'),('grzW1','grzW1W2')]:
        for method in ('fixed_forest','frozen_partitions_same_labels'):
            a=p[left+'_'+method][dev];b=p[right+'_'+method][dev];yy=y[dev]
            delta=np.log(np.maximum(b[np.arange(len(yy)),yy],1e-12))-np.log(np.maximum(a[np.arange(len(yy)),yy],1e-12))
            pairs.append(dict(left=left,right=right,method=method,n=len(yy),**cluster_mean_interval(delta,t['cell'][dev])))
    csv_rows(TABLES/'information_paired_bands.csv',pairs)
    save_json(DEST/'information_complete.json',dict(cal_n=int(cal.sum()),dev_n=int(dev.sum()),global_prior=prior.tolist(),matched_identity=True,
        no_fit=True,cal_only_population_prior=True,not_information_bound=True,
        limitation='Frozen probe caches use their original error/calibration treatment; all arms share identities and population. Narrow matched i+WISE reference subset is not optical-parent composition.',
        inputs={str(x.relative_to(ROOT)):sha(x) for x in [ROOT/'data/v2_1/information_eval.npz',ROOT/'data/v2_1/information_probabilities.npz']},job=os.getenv('SLURM_JOB_ID')))
    print('INFORMATION COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['anomalies','metal','information','all']);args=parser.parse_args();setup()
    for name,fn in [('anomalies',anomalies),('metal',metal),('information',information)]:
        if args.task in (name,'all'):fn()
