#!/usr/bin/env python3
"""Use M68 to test the native DECam, metal-poor CMD transfer outside CLUSTER."""
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, plt, save
from v3_9_references import subset
from v3_10_figures import check_templates
from v3_11_layers import read_receipt
from v3_13_clusters import DISTANCE, SYSTEMS, forecasts, photos
from v3_13_cmd import ABS, APP, C, MU, fit_template
from v3_14_audit import DEST as AUDIT, annotate, area, read_flags

DEST = ROOT/'data/v3_14/structure'
SOURCE = ROOT/'data/v3_14/input/m68_members.txt'
DIGEST = '2abad48f2d3ea23952f60b7a36069f1d385175be6c5fdd69ea398b33a61f3fcd'
M68 = (189.86658,-26.74406)
D68 = 10.404


def perform(t,eligible,area,base,fit):
    good = np.isfinite(t['qrf_kpc'][:,2])&eligible
    on,off = good&t['target'],good&~t['target']
    gr = t['g0']-t['r0']
    stats = dict(target=int(on.sum()),background=int(off.sum()),target_members=int((on&t['member']).sum()),
                  template_members=int(fit.sum()),target_area=float(area[0]),background_area=float(area[1]))
    if (area<=0).any() or min(on.sum(),off.sum())<30 or stats['target_members']<30 or fit.sum()<30:
        return stats,{}
    kernel = np.histogram2d(base['g0'][fit]-base['r0'][fit],
                            base['r0'][fit]-10-5*np.log10(SYSTEMS['M30'][1]),(C,ABS))[0]
    kernel = gaussian_filter(kernel,(1.,1.))
    kernel /= kernel.sum()
    h,k = [np.histogram2d(gr[j],t['r0'][j],(C,APP))[0] for j in (on,off)]
    score,_,amp = fit_template(kernel,h,k,area[0]/area[1])
    d = 10**((MU-10)/5)
    j = int(np.argmax(score))
    hist = t['qrf_mass'][:,1:-1]
    p,q = [hist[g].sum(axis=0,dtype=np.float64)/a for g,a in zip((on,off),area)]
    mid = (DISTANCE[:-1]+DISTANCE[1:])/2
    net = p-q
    peak = mid[np.argmax(net)]
    w = np.clip((np.minimum(DISTANCE[1:],D68+1)-np.maximum(DISTANCE[:-1],D68-1))/.25,0,1)
    denom = np.sqrt(np.sum(((hist[on]@w)/area[0])**2)+np.sum(((hist[off]@w)/area[1])**2))
    cmd_window_z = np.max(score[abs(d-D68)<=1])
    qrf_window_z = float(net@w/denom) if denom>0 else np.nan
    stats.update(cmd_peak=d[j],cmd_score=score[j],cmd_peak_bias=d[j]/D68-1,
                 cmd_window_score=cmd_window_z,cmd_pass=bool(abs(d[j]/D68-1)<=.1 and cmd_window_z>=3),
                 qrf_peak=peak,qrf_peak_bias=peak/D68-1,qrf_window_z=qrf_window_z,
                 qrf_pass=bool(abs(peak/D68-1)<=.1 and qrf_window_z>=3))
    return stats,dict(cmd=score,qrf=net,amplitude=amp)


def run():
    inputs = {'style':check_templates(),'membership':DIGEST,'distance_source':'https://academic.oup.com/mnras/article/505/4/5957/6283731'}
    SYSTEMS['M68'] = M68,D68
    base = read_receipt(AUDIT/'M30','complete.json','native_join.npz',
             ('TYPE','g0','r0','r','target','member','cluster_mask','low_crowding','training_or_calibration'),inputs)
    assert np.all(base['TYPE']=='PSF')
    p,roster = photos('M68',inputs)
    gr = p['g0']-p['r0']
    p = subset(p,(gr>=C[0])&(gr<C[-1])&(p['r0']>=APP[0])&(p['r0']<APP[-1]))
    p.update(read_flags('M68',p,inputs))
    annotate(p,'M68')
    footprints = area('M68',inputs)
    forecasts('south',p,inputs)
    members = np.loadtxt(verified(SOURCE,DIGEST),usecols=(0,16),dtype=[('id','i8'),('P','f8')])
    p['member'] = np.isin(p['source_id'],members['id'][members['P']>.9])
    p['member'] &= unique_join(p['source_id'],p['source_id'])>=0
    fit = base['target']&base['member']&~base['training_or_calibration']&(base['r']<21.)
    eligible = np.ones(len(p['TYPE']),bool)
    out,signals = [],{}
    for label in ('outside CLUSTER, low crowding','all admitted diagnostic'):
        strict = label.startswith('outside')
        chosen = eligible&~p['cluster_mask']&p['low_crowding'] if strict else eligible
        train = fit&~base['cluster_mask']&base['low_crowding'] if strict else fit
        a = np.array([sum(x['area_deg2'] for x in footprints if x['zone']==zone and x['limiting_r']==0
                          and (not strict or not x['masked'])) for zone in ('target','background')])
        row,curves = perform(p,chosen,a,base,train)
        out.append(dict(selection=label,reference_kpc=D68,**row))
        if curves:
            signals[label] = curves
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'admission.csv',roster)
    csv_rows(DEST/'area.csv',footprints)
    csv_rows(DEST/'comparison.csv',out)
    atomic_npz(DEST/'field.npz',TYPE=p['TYPE'],ls_id=p['ls_id'],target=p['target'],member=p['member'],
               cluster_mask=p['cluster_mask'],low_crowding=p['low_crowding'],qrf_mass=p['qrf_mass'],
               training_or_calibration=p['training_or_calibration'])
    if signals:
        with plt.rc_context(STYLE):
            f,axes = plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
            for label,data in signals.items():
                axes[0].plot(10**((MU-10)/5),data['cmd'],label=label,color=BLUE if label.startswith('outside') else ORANGE)
                axes[1].stairs(data['qrf'],DISTANCE,label=label,color=BLUE if label.startswith('outside') else ORANGE)
            for ax,y in zip(axes,('Conditional matched-filter score','Excess stars per cell per sq. deg.')):
                ax.axvline(D68,color=GRAY,ls='--')
                ax.set(xlabel='Distance (kpc)',ylabel=y,xlim=(5,14))
                ax.legend()
            f.suptitle('M68 · independent DECam template transfer')
            save(f,DEST,'distance',out,{'mu':MU,'qrf_distance_bins':DISTANCE,
                **{f'{label}_{k}':z for label,v in signals.items() for k,z in v.items()}})
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        rows=len(p['TYPE']),demographics=out,training_exclusions=int(p['training_or_calibration'].sum()),
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
