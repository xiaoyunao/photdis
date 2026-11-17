#!/usr/bin/env python3
"""Compare M5 and M68 with the same frozen CMD template and quality selection."""
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, plt, save
from v3_9_references import subset
from v3_10_figures import check_templates
from v3_11_layers import read_receipt
from v3_13_clusters import DISTANCE, SYSTEMS, photos
from v3_13_cmd import ABS, APP, C, MU, M5_CENTER, M5_D, fit_template
from v3_14_audit import DEST as AUDIT
from v3_14_structure import D68, DEST as M68_FOLDER, M68

DEST = ROOT/'data/v3_15/transfer'
DRAW = 512
SEED = 31509
D = 10**((MU-10)/5)


def cohort(name,inputs):
    root = AUDIT/name if name=='M5' else M68_FOLDER
    required = ('TYPE','ls_id','target','member','cluster_mask','low_crowding',
                'qrf_mass','training_or_calibration')
    if name=='M5':
        t = read_receipt(root,'complete.json','native_join.npz',(*required,'g0','r0'),inputs)
        a = AUDIT/'M5'
    else:
        t = read_receipt(root,'complete.json','field.npz',required,inputs)
        SYSTEMS['M68'] = M68,D68
        p,_ = photos('M68',inputs)
        j = unique_join(t['ls_id'],p['ls_id'])
        if np.any(j<0) or not np.all(p['TYPE'][j]=='PSF'):
            raise ValueError('M68 frozen IDs do not reproduce the native roster')
        np.testing.assert_array_equal(t['target'],p['target'][j])
        t.update(g0=p['g0'][j],r0=p['r0'][j])
        a = root
    if not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id']):
        raise ValueError('Invalid native test identity')
    with (a/'area.csv').open() as f:
        from csv import DictReader
        areas = list(DictReader(f))
    metadata = json.loads((a/'complete.json').read_text())
    assert sha(a/'area.csv')==metadata['outputs']['area.csv']
    dist = M5_D if name=='M5' else D68
    gr = t['g0']-t['r0']
    valid = (gr>=C[0])&(gr<C[-1])&(t['r0']>=APP[0])&(t['r0']<APP[-1])
    p = t['qrf_mass']
    valid &= (p.sum(axis=1,dtype=float)>0)&~t['training_or_calibration']
    views = {}
    for mode in ('strict','all'):
        outside = mode=='strict'
        admitted = valid&~t['cluster_mask']&t['low_crowding'] if outside else valid
        x,y = admitted&t['target'],admitted&~t['target']
        area = np.array([sum(float(k['area_deg2']) for k in areas if k['zone']==zone
                       and k['limiting_r']=='0' and (not outside or k['masked']=='False'))
                       for zone in ('target','background')])
        if min(x.sum(),y.sum())<30 or (area<=0).any():
            raise ValueError(f'{name} lacks common support for {mode}')
        hist = [np.histogram2d(gr[k],t['r0'][k],(C,APP))[0] for k in (x,y)]
        pdf = p[:,1:-1]
        net = pdf[x].sum(axis=0,dtype=float)/area[0]-pdf[y].sum(axis=0,dtype=float)/area[1]
        mid = .5*(DISTANCE[1:]+DISTANCE[:-1])
        peak = mid[np.argmax(net)]
        window = np.clip((np.minimum(DISTANCE[1:],dist+1)-np.maximum(DISTANCE[:-1],dist-1))/.25,0,1)
        var = sum(np.square((pdf[k]@window)/a).sum() for k,a in zip((x,y),area))
        row = dict(cluster=name,sample=mode,target=int(x.sum()),background=int(y.sum()),
              members=int((x&t['member']).sum()),reference_kpc=dist,target_deg2=area[0],background_deg2=area[1],
              qrf_peak=peak,qrf_relative_bias=peak/dist-1,qrf_window_z=float(net@window/np.sqrt(var)))
        views[mode] = dict(row=row,hist=hist,ratio=area[0]/area[1],qrf=net,used=admitted)
    return t,views


def kernel(c,m):
    h = np.histogram2d(c,m,(C,ABS))[0]
    h = gaussian_filter(h,(1.,1.))
    return h/h.sum()


def sampled_kernels(c,m):
    draw = np.random.default_rng(SEED).integers(len(c),size=(DRAW,len(c)))
    return np.array([kernel(c[k],m[k]) for k in draw]),draw


def boot_scores(kernels,view):
    """Vectorized interpolation of the same template grid used by fit_template."""
    x = .5*(ABS[1:]+ABS[:-1])
    y = .5*(APP[1:]+APP[:-1])[None,:]-MU[:,None]
    j = np.clip(np.searchsorted(x,y)-1,0,len(x)-2)
    f = (y-x[j])/(x[j+1]-x[j])
    keep = (y>=x[0])&(y<=x[-1])
    on,off = view['hist']
    w = 1/np.maximum(on+off*view['ratio']**2,5.)
    value = (on-view['ratio']*off)*w
    blocks = []
    for chunk in np.array_split(kernels,16):
        u = ((1-f)*chunk[...,j]+f*chunk[...,j+1])*keep
        norm = np.einsum('bcma,ca->bm',u**2,w)
        blocks.append(np.einsum('bcma,ca->bm',u,value)/np.sqrt(norm))
    return np.concatenate(blocks)


def run():
    inputs = {'style':check_templates(),'seed':SEED,'resamples':DRAW}
    base = read_receipt(AUDIT/'M30','complete.json','native_join.npz',
        ('TYPE','ls_id','g0','r0','r','target','member','cluster_mask','low_crowding',
         'training_or_calibration'),inputs)
    assert np.all(base['TYPE']=='PSF')
    fit = base['target']&base['member']&~base['training_or_calibration']&(base['r']<21)
    strict = fit&~base['cluster_mask']&base['low_crowding']
    assert strict.sum()==99 and fit.sum()==1077
    color = base['g0']-base['r0']
    mag = base['r0']-10-5*np.log10(SYSTEMS['M30'][1])
    models = {'strict':kernel(color[strict],mag[strict]),'all':kernel(color[fit],mag[fit])}
    resamples,index = sampled_kernels(color[strict],mag[strict])
    rows,signals,draws,parts = [],{},{},[]
    for name in ('M5','M68'):
        t,views = cohort(name,inputs)
        parts.append((name,t,views['strict']['used']))
        for field,view in views.items():
            row = view['row']
            dist = row['reference_kpc']
            for mode,k in models.items():
                score,_,_ = fit_template(k,*view['hist'],view['ratio'])
                peak = D[np.argmax(score)]
                z = np.max(score[abs(D-dist)<=1])
                rows.append(dict(**row,template=mode,cmd_peak=peak,cmd_relative_bias=peak/dist-1,
                     cmd_window_score=z,cmd_pass=bool(abs(peak/dist-1)<=.10 and z>=3)))
                signals[f'{name}_{field}_{mode}'] = score
            s = boot_scores(resamples,view)
            draws[f'{name}_{field}'] = D[np.argmax(s,axis=1)]
            signals[f'{name}_{field}_qrf'] = view['qrf']
            np.testing.assert_allclose(boot_scores(models['strict'][None],view)[0],
                                       signals[f'{name}_{field}_strict'],atol=1e-9)
    summaries = []
    for key,p in draws.items():
        dist = M5_D if key.startswith('M5') else D68
        lo,med,hi = np.quantile(p,[.025,.5,.975])
        summaries.append(dict(field=key,n=len(p),peak_p025=lo,peak_p50=med,peak_p975=hi,
                      within_ten_pct=(abs(p/dist-1)<=.10).mean()))
    diff = draws['M68_strict']/D68-draws['M5_strict']/M5_D
    lo,med,hi = np.quantile(diff,[.025,.5,.975])
    summaries.append(dict(field='paired M68 minus M5 relative bias',n=len(diff),
                         peak_p025=lo,peak_p50=med,peak_p975=hi))
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'comparison.csv',rows)
    csv_rows(DEST/'bootstrap.csv',summaries)
    atomic_npz(DEST/'rosters.npz',template_id=base['ls_id'][strict],draw_index=index,
         **{name+'_'+key:v[used] for name,t,used in parts
            for key,v in t.items() if key in ('TYPE','ls_id','target')})
    with plt.rc_context(STYLE):
        f,axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
        for i,(name,dist) in enumerate((('M5',M5_D),('M68',D68))):
            ax,bx = axes[i]
            ax.plot(D,signals[f'{name}_strict_strict'],color=BLUE)
            ax.axvline(dist,color=GRAY,ls='--')
            ax.set(xlim=(5,14),title=f'{name} · fixed M30 template',xlabel='Distance (kpc)',
                   ylabel='Conditional matched-filter score')
            bx.hist(draws[f'{name}_strict'],bins=np.linspace(5,14,61),color=ORANGE)
            bx.axvline(dist,color=GRAY,ls='--')
            bx.set(xlim=(5,14),title=f'{name} · paired template resampling',
                   xlabel='Distance peak (kpc)',ylabel='Resamples per bin')
        save(f,DEST,'matched_transfer',rows,dict(distance_kpc=D,**signals,**draws))
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
