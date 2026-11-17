#!/usr/bin/env python3
"""Score frozen GD-1 outcomes against independent CFHT photometric distances."""
import csv
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_8_direct_figures import BLUE, GRAY, GREEN, ORANGE, STYLE, plt, save
from v3_10_figures import check_templates
from v3_13_delivery import figure_check
from v3_15_delivery import validate
from v3_16_fit import DEPTH, FIT, METHODS
from v3_16_gd1 import CONFIG, DEST, EDGES, P

REFERENCE = ROOT/'manifests/v3_16/cfht_reference.csv'
CFHT = 'https://doi.org/10.1093/mnras/sty677'


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def slope(x,y):
    a = x-x.mean()
    return a@y/np.square(a).sum()


def summary(x):
    q = np.nanquantile(x,(.025,.5,.975))
    return dict(p025=float(q[0]),median=float(q[1]),p975=float(q[2]))


def comparison(rows,ref,curves):
    by = {(x['method'],x['depth'],float(x['phi1'])):x for x in rows}
    lon = np.array(list(ref))
    results,changes = [],[]
    arrays = {}
    rng = np.random.default_rng(P['seed']+1)
    jitters = rng.normal(size=(len(lon),P['spatial_draws']))
    for k,name in enumerate(DEPTH):
        valid = np.array([f's{s}_k{k}_peak' in curves for s in range(len(lon))])
        loc = lon[valid]
        if len(loc)<2:
            continue
        d = np.array([ref[x][0] for x in loc])
        e = np.array([ref[x][1] for x in loc])
        ref_slope = slope(loc,d)
        reference_draws = d[:,None]+e[:,None]*jitters[valid]
        for m,method in enumerate(METHODS):
            rows_for = [by[method,name,x] for x in loc]
            recovered = np.array([float(r['peak']) for r in rows_for])
            replicates = np.array([curves[f's{s}_k{k}_peak'][m,1:] for s in np.flatnonzero(valid)])
            on = np.array([r['signal']=='True' for r in rows_for])
            accuracy = abs(recovered/d-1)<=P['reference_tolerance_fraction']
            errors = abs(replicates/reference_draws-1).mean(axis=0)
            arrays[name,method] = valid,errors,np.asarray([float(r['peak_z']) for r in rows_for]),on
            errors_slope = slope(loc,replicates-reference_draws)
            qs = summary(slope(loc,replicates))
            ss = summary(errors_slope)
            results.append(dict(depth=name,method=method,segments=len(loc),detected=int(on.sum()),
               within_tolerance=int(accuracy.sum()),both=int((accuracy&on).sum()),
               mean_signed_bias=float((recovered/d-1).mean()),mean_abs_bias=float(abs(recovered/d-1).mean()),
               ref_slope_kpc_deg=ref_slope,slope_kpc_deg=slope(loc,recovered),slope_p025=qs['p025'],
               slope_p975=qs['p975'],delta_slope_p025=ss['p025'],delta_slope_p975=ss['p975']))
    for depth,base in [('r < 21','r < 20'),('r < 22','r < 21')]:
        for method in METHODS:
            if (depth,method) not in arrays or (base,method) not in arrays:
                continue
            a,b = arrays[base,method],arrays[depth,method]
            if not np.array_equal(a[0],b[0]):
                continue
            changes.append(dict(contrast=f'{depth} minus {base}',method=method,segments=int(a[0].sum()),
                               peak_z_mean_change=float((b[2]-a[2]).mean()),
                               change_in_detections=int(b[3].sum()-a[3].sum()),**summary(b[1]-a[1])))
    for depth in DEPTH:
        if (depth,METHODS[1]) not in arrays:
            continue
        a,b = arrays[depth,METHODS[1]],arrays[depth,METHODS[2]]
        changes.append(dict(contrast='M92 deep CMD minus M13 CMD',method=depth,segments=int(a[0].sum()),
                            peak_z_mean_change=float((b[2]-a[2]).mean()),
                            change_in_detections=int(b[3].sum()-a[3].sum()),**summary(b[1]-a[1])))
    return results,changes


def figure(out,rows,external):
    colors = (BLUE,ORANGE,GREEN)
    x = np.array(list(external))
    r = np.array(list(external.values()))
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,3,figsize=(15,5),layout='constrained',sharex=True,sharey=True)
        for ax,method,color in zip(axes,METHODS,colors):
            ax.errorbar(x,r[:,0],yerr=r[:,1],marker='o',color=GRAY,ls='--',label='CFHT reference')
            for depth,mark,alpha in [('r < 20','s',.40),('r < 21','^',.65),('r < 22','o',1.)]:
                a = [k for k in rows if (k['method'],k['depth'])==(method,depth)]
                xx,y,lo,hi = (np.array([float(k[key]) for k in a]) for key in ('phi1','peak','p16','p84'))
                ax.vlines(xx,lo,hi,color=color,alpha=alpha,lw=1.)
                ax.scatter(xx,y,marker=mark,c=color,alpha=alpha,label=depth)
            ax.set(title=method,xlabel=r'Stream longitude $\phi_1$ (deg)',ylim=(5,14))
            ax.legend(loc='upper left',fontsize=9)
        axes[0].set_ylabel('Heliocentric distance (kpc)')
        save(fig,out,'gd1_distance',rows)
    figure_check(out/'gd1_distance.png')


def run():
    lock = check_templates()
    inputs = {'config':sha(CONFIG),'reference':CFHT,'reference_csv':sha(REFERENCE)}
    for stage in (DEST/'footprint',DEST/'sample',FIT):
        validate(stage,inputs,lock)
    out = DEST/'score_block'
    out.mkdir(parents=True,exist_ok=True)
    r = read(REFERENCE)
    ref = {float(a['phi1']):(float(a['distance_kpc']),float(a['error_kpc'])) for a in r}
    np.testing.assert_array_equal(list(ref),.5*(EDGES[1:]+EDGES[:-1]))
    rows = read(FIT/'segments.csv')
    for x in rows:
        d,e = ref[float(x['phi1'])]
        delta = float(x['peak'])/d-1
        x.update(reference_kpc=d,reference_error_kpc=e,relative_error=delta,
                 within_ten_percent=bool(abs(delta)<=P['reference_tolerance_fraction']))
    with np.load(FIT/'curves.npz',allow_pickle=False) as curves:
        result,depth = comparison(rows,ref,curves)
    csv_rows(out/'distance.csv',rows)
    csv_rows(out/'gradient.csv',result)
    csv_rows(out/'depth.csv',depth)
    figure(out,rows,ref)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),style=lock,inputs=inputs,
       outputs={f.name:sha(f) for f in out.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
