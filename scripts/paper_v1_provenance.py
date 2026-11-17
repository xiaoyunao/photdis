#!/usr/bin/env python3
"""Audit the frozen training identities and conditional luminosity-function checks."""
import csv
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT as NATIVE
from final_catalogue_benchmark import load_parent, parent, published, masks
from final_catalogue_population import OUT as LF, phase
from release_readiness_compare import available
from v3_10_redrock import DEST as RR
from paper_v1_scan import OUT


def roster(t, headline, inputs, dest):
    side = ROOT/'data/v3_0'
    m = side/'controls/complete.json'
    r = json.loads(m.read_text())
    a = read_columns(verified(m.with_name('training.npz'),r['outputs']['training.npz']),
                     ('TYPE','rows','identity','primary_mu'))
    b = read_columns(verified(side/'references.npz',r['inputs']['data/v3_0/references.npz']),
                     ('TYPE','ls_id','source_id','identity','primary_channel','RA','DEC','role'))
    ii = a['rows']
    assert (a['TYPE']=='PSF').all() and (b['TYPE'][ii]=='PSF').all()
    np.testing.assert_array_equal(a['identity'],b['identity'][ii])
    assert np.all(b['role'][ii]==1)
    inputs['south_train'] = sha(m)
    m = NATIVE/'north/native_model/complete.json'
    r = json.loads(m.read_text())
    n = read_columns(verified(m.with_name('roles.npz'),r['outputs']['roles.npz']),
                     ('TYPE','ls_id','source_id','role'))
    assert (n['TYPE']=='PSF').all()
    inputs['north_roles'] = sha(m)
    rows = []
    for side,label,k,src,target in (
        ('DECam','qrf_fit_specdis',ii[b['primary_channel'][ii]==1],b,'SpecDis calibrated primary posterior mean mu'),
        ('DECam','qrf_fit_spdist',ii[b['primary_channel'][ii]==2],b,'SPDist calibrated primary posterior mean mu'),
        ('DECam','qrf_fit_gaia',ii[b['primary_channel'][ii]==3],b,'signed Gaia primary posterior mean mu'),
        *(( 'BASS_MzLS',name,np.flatnonzero(n['role']==name),n,desc) for name,desc in
          (('gaia_fit','high-S/N Gaia geometry'),('specdis_fit','quality-qualified SpecDis'),
           ('calibration','independent Gaia geometry zero-point fit'),('test','Gaia geometry holdout'),
           ('class_fit','spectroscopic class label')))):
        keys = src['ls_id'][k]
        g = src['source_id'][k]
        same = np.isin(t['ls_id'],keys[keys>0])
        same_gaia = (t['source_id']>0)&np.isin(t['source_id'],g[g>0])
        on_sky = (t['region']=='north')==(side=='BASS_MzLS')
        def overlap(w):
            return dict(bench_ls=int((w&same).sum()),bench_gaia=int((w&same_gaia).sum()),
                        bench_either=int((w&(same|same_gaia)).sum()))
        for name,h in (('parent',np.ones(len(same),bool)),*headline.items()):
            for where,w in (('all',h),('same_native',h&on_sky)):
                rows.append(dict(training_native=side,role=label,target=target,n=len(k),
                    training_gaia_keys=int((g>0).sum()),test_frame=name,test_domain=where,
                    test_size=int(w.sum()),**overlap(w)))
    assert sum(x['n'] for x in rows if x['training_native']=='DECam' and
               x['test_frame']=='parent' and x['test_domain']=='all')==32768
    csv_rows(dest/'training_overlap.csv',rows)
    def block(a,d):
        return (a//10).astype(int)*18+np.minimum(((d+90)//10).astype(int),17)
    k=block(b['RA'][ii],b['DEC'][ii])
    z=block(t['RA'],t['DEC'])
    csv_rows(dest/'south_training_sky.csv',[dict(test_frame=name,n=int(use.sum()),
        southern_fit_blocks=len(np.unique(k)),test_blocks=len(np.unique(z[use])),
        common_blocks=len(np.intersect1d(k,z[use])),test_rows_in_south_fit_blocks=int((use&np.isin(z,k)).sum()))
        for name,j in (('parent',np.ones(len(z),bool)),*headline.items())
        for use in (j&(t['region']=='south_low'),)])


def lf_summary(inputs,dest):
    mark = LF/'complete.json'
    m = json.loads(mark.read_text())
    p = verified(mark.with_name('predictive_counts.csv'),m['outputs']['predictive_counts.csv'])
    with p.open() as f:
        x = list(csv.DictReader(f))
    inputs['lf'] = sha(mark)
    rows=[]
    for region in sorted({r['region'] for r in x}):
        for ref in ('specdis','spdist'):
            for scenario in ('empirical','disk_rich','halo_rich'):
                subset = [r for r in x if (r['region'],r['reference'],r['scenario'])==(region,ref,scenario)]
                for event in ('all_core','giant','intermediate_gravity','severe_core','all_events'):
                    rows_event = [r for r in subset if r['event']==event or event=='all_events']
                    nonzero = [r for r in rows_event if int(r['observed'])>0]
                    rows.append(dict(region=region,reference=ref,scenario=scenario,event=event,
                         cells=len(rows_event),covered=sum(r['covered']=='True' for r in rows_event),
                         observed_nonzero_cells=len(nonzero),covered_nonzero=sum(r['covered']=='True' for r in nonzero),
                         observed_total=sum(int(r['observed']) for r in rows_event) if event!='all_events' else None,
                         sum_cell_medians=sum(float(r['expected']) for r in rows_event) if event!='all_events' else None))
    csv_rows(dest/'lf_positive_counts.csv',rows)


def nonstellar_control(inputs,dest):
    p=RR/'complete.json'
    m=json.loads(p.read_text())
    a=read_columns(verified(p.with_name('qualified.npz'),m['outputs']['qualified.npz']),
                   ('ls_id','TYPE','qualified'))
    rows=[]
    for domain in ('north','south'):
        t=parent(domain,inputs)
        labels=phase(t,inputs)
        j=unique_join(t['ls_id'],a['ls_id'])
        matched=j>=0
        np.testing.assert_array_equal(a['TYPE'][j[matched]],t['TYPE'][matched])
        rr=np.where(t['redrock_quality_assessed'],t['redrock_SPECTYPE'],'').astype('U8')
        matched[matched] = a['qualified'][j[matched]]!=''
        rr[matched]=a['qualified'][j[matched]]
        d=t['qrf_quantiles_kpc'][:,2]
        obs=(t['r']>=18)&(t['r']<20)&(t['sky_role']==4)&t['evaluation_row']&(t['row_state']==7)
        for stage,w in (('supported',t['state']>=2),('core_2_5',
                        (t['state']==4)&(d>=2)&(d<5))):
            z=obs&w
            rows.append(dict(region=t['region'][0],stage=stage,n_heldout=int(z.sum()),
                n_redrock_class=int((z&np.isin(rr,('STAR','QSO','GALAXY'))).sum()),
                n_nonstellar=int((z&np.isin(rr,('QSO','GALAXY'))).sum()),
                nonstellar_unconflicted=int((z&(labels=='qualified_nonstellar')).sum()),
                class_conflicts=int((z&(labels=='class_conflict')).sum())))
    csv_rows(dest/'redrock_nonstellar.csv',rows)


def main():
    require_compute()
    d=OUT/'provenance'
    d.mkdir(exist_ok=True,parents=True)
    t,receipt=load_parent()
    q,_,inputs=published(t,receipt['outputs']['ids.npy'])
    use=masks(t)['weak_gaia_dwarfs']&available(q['Photdis'])&available(q['BJ_photogeometric'])
    headline={s:use&(t[s.lower()+'_kpc']>=2)&(t[s.lower()+'_kpc']<5) for s in ('SpecDis','SPDist')}
    roster(t,headline,inputs,d)
    lf_summary(inputs,d)
    nonstellar_control(inputs,d)
    save_json(d/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        outputs={p.name:sha(p) for p in d.glob('*.csv')}))


if __name__=='__main__':
    main()
