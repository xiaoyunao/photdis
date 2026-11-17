#!/usr/bin/env python3
"""Measure coverage and compare distances on the identical frozen test stars."""
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import DIST, ORANGE, STYLE, density, plt, save
from v3_10_figures import check_templates
from release_readiness_archive import PRODUCTS
from release_readiness_sample import DEST, FIELDS

SHELLS = ((0.,np.inf),(0.,2.),(2.,5.),(5.,10.),(10.,np.inf))
NAMES = {'Photdis':'Photdis','BJ_geometric':'Bailer-Jones geometric',
         'BJ_photogeometric':'Bailer-Jones photogeometric','GSP_Phot':'Gaia GSP-Phot',
         'StarHorse':'StarHorse 2022'}


def available(x):
    return np.isfinite(x[:,1])&(x[:,1]>0)


def interval_ok(x):
    return np.isfinite(x).all(axis=1)&(x[:,0]>0)&(x[:,0]<=x[:,1])&(x[:,1]<=x[:,2])


def stats(q, ref):
    ratio = q[:,1]/ref
    if not len(ratio):
        return dict(n=0,n_interval=0,bias=None,nmad=None,tail=None,coverage68=None)
    delta=ratio-1
    center = np.median(delta)
    w=interval_ok(q)
    covers=(q[w,0]<=ref[w])&(ref[w]<=q[w,2])
    return dict(n=len(ratio),n_interval=int(w.sum()),bias=float(center),
                nmad=float(1.4826*np.median(abs(delta-center))),
                tail=float((abs(5*np.log10(ratio))>1).mean()),
                coverage68=float(covers.mean()) if len(covers) else None)


def load_products(t, expected, inputs):
    ids = t['source_id']
    datasets = {'Photdis':t['q'][:,1:4]}
    found = {'Photdis':np.ones(len(ids),bool)}
    for source, (_, _, _, fields) in PRODUCTS.items():
        path = DEST/'archives'/source/'complete.json'
        r = json.loads(path.read_text())
        if r['rows']!=len(ids) or r['inputs']['ids_sha256']!=expected:
            raise ValueError(f'Archive query does not use the requested Gaia cohort: {source}')
        archive_path = verified(path.with_name('matched.npz'),r['outputs']['matched.npz'])
        with np.load(archive_path,allow_pickle=False) as a:
            j = np.searchsorted(ids,a['source_id'])
            if np.any(j>=len(ids)) or len(np.unique(j))!=len(j):
                raise ValueError(f'Invalid {source} rows')
            np.testing.assert_array_equal(ids[j],a['source_id'])
            if len(j)!=r['matched']:
                raise ValueError('Receipt loses matched sources')
            for name in fields:
                datasets[name] = np.full((len(ids),3),np.nan)
                datasets[name][j] = a[name]
                found[name] = np.isin(np.arange(len(ids)),j)
        inputs[source] = sha(path)
    return datasets,found


def strata(t):
    n = len(t['ls_id'])
    lab = t['rvs_label']
    f,df = t['rvs_feh'],t['rvs_feh_err']
    feh = np.isfinite(f)&(df>0)&(df<=.3)
    c = t['color']
    return (('all',np.ones(n,bool)),
            ('r_lt_18',t['r']<18),('r_18_20',(t['r']>=18)&(t['r']<20)),
            ('color_blue',c<.5),('color_0p5_0p8',(c>=.5)&(c<.8)),('color_ge_0p8',c>=.8),
            ('faint_blue',(t['r']>=18)&(c<.5)),('spectroscopic_dwarf',lab==0),
            ('dwarf_feh_ge_minus1p5',(lab==0)&feh&(f>=-1.5)),
            ('dwarf_feh_lt_minus1p5',(lab==0)&feh&(f<-1.5)),
            ('non_dwarf_stellar',np.isin(lab,(1,2))),('spectrally_unclassified',lab<0))


def profiles(t,measurements,found,dest):
    coverage,pairs,overlap=[],[],[]
    valid={k:available(v) for k,v in measurements.items()}
    common=np.logical_and.reduce(list(valid.values()))
    masks=(('all',np.ones(len(common),bool)),('north',t['region']==3),('south_low',t['region']==2))
    populations=strata(t)
    competitors=list(measurements)[1:]
    bit=np.zeros(len(common),'u1')
    for i,k in enumerate(competitors):
        bit |= valid[k].astype('u1')<<i
    for region,r in masks:
        for group,g in populations:
            j=r&g
            if not j.any():
                continue
            none = bit[j]==0
            for name,q in measurements.items():
                coverage.append(dict(region=region,group=group,method=name,n=int(j.sum()),
                    archive_row=int((j&found[name]).sum()),valid_median=int((j&valid[name]).sum()),
                    valid68=int((j&interval_ok(q)).sum()),not_queried=0,
                    all_products=int((j&common).sum()),no_external_median=int(none.sum())))
            for b,n in enumerate(np.bincount(bit[j],minlength=16)):
                overlap.append(dict(region=region,group=group,pattern=b,n=int(n),
                                    products='+'.join(k for i,k in enumerate(competitors) if (b>>i)&1)))
            for source in ('SpecDis','SPDist'):
                ref = t[source.lower()+'_kpc']
                base=np.flatnonzero(j&t['heldout']&np.isfinite(ref)&(ref>0))
                if not len(base):
                    continue
                for lo,hi in SHELLS:
                    use=base[(ref[base]>=lo)&(ref[base]<hi)]
                    comparisons=[('photdis_reference_all','Photdis',use)]
                    for other in competitors:
                        pair=use[valid[other][use]]
                        comparisons.extend((('pair_'+other,'Photdis',pair),('pair_'+other,other,pair)))
                    comparisons.extend(('all_methods_common',k,use[common[use]]) for k in measurements)
                    for match,name,k in comparisons:
                        pairs.append(dict(region=region,group=group,reference=source,ref_lo=lo,
                            ref_hi=hi,match=match,method=name,ref_n=len(use),
                            **stats(measurements[name][k],ref[k])))
    csv_rows(dest/'coverage.csv',coverage)
    csv_rows(dest/'method_overlap.csv',overlap)
    csv_rows(dest/'paired_distances.csv',pairs)
    return common


def figures(t,q,common,dest):
    fig,axes=plt.subplots(2,3,figsize=(18,11),layout='constrained')
    rows,counts=[],{}
    panels=[('SpecDis',name) for name in q]+[('SPDist','Photdis')]
    for i,(ax,(source,name)) in enumerate(zip(axes.ravel(),panels)):
        ref=t[source.lower()+'_kpc']
        j=common&t['heldout']&(ref>=2)&(ref<5)&(t['r']>=18)&(t['rvs_label']==0)
        d=ref[j]
        z=q[name][j]
        edges=DIST[(DIST>=1)&(DIST<=20)]
        h=density(fig,ax,d,z[:,1],edges,edges)
        ax.plot(edges[[0,-1]],edges[[0,-1]],ls='--',color=ORANGE,lw=1.5)
        ax.set(title=f'{NAMES[name]} · {source}',xlabel='Spectroscopic reference (kpc)',
               ylabel='Distance estimate (kpc)',xlim=(1,12),ylim=(1,12))
        summary=stats(z,d)
        caption = (f"N = {len(d):,} · bias {100*summary['bias']:+.1f}%"
                   if len(d) else 'No common reference')
        ax.text(.04,.94,caption,transform=ax.transAxes,va='top',
                bbox={'facecolor':'white','edgecolor':'none','alpha':.83})
        rows.append(dict(panel=i,reference=source,method=name,**summary))
        counts[f'panel_{i}']=h
    fig.suptitle('Same stars across published products · faint spectroscopic dwarfs · 2–5 kpc reference')
    save(fig,dest,'F12_published_distances',rows,dict(distance_bins=edges,**counts))


def main():
    require_compute()
    dest=DEST/'comparison'
    dest.mkdir(exist_ok=True,parents=True)
    marker=DEST/'cohort_receipt.json'
    meta=json.loads(marker.read_text())
    t=read_columns(verified(DEST/'cohort.npz',meta['outputs']['cohort.npz']),FIELDS)
    if len(t['ls_id'])!=meta['rows'] or (t['TYPE']!='PSF').any():
        raise ValueError('Comparison needs the unchanged native PSF cohort')
    inputs=dict(cohort=sha(marker),style=check_templates(),script=sha(ROOT/'scripts/release_readiness_compare.py'))
    data,found=load_products(t,meta['outputs']['gaia_ids.npy'],inputs)
    mask=profiles(t,data,found,dest)
    with plt.rc_context(STYLE):
        figures(t,data,mask,dest)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,rows=meta['rows'],
              outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
