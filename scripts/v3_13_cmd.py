#!/usr/bin/env python3
"""Evaluate a locked M30 CMD template on the independent native M5 field."""
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, plt, save
from v3_9_references import subset
from v3_10_figures import check_templates
from v3_13_clusters import DEST as CLUSTERS, DISTANCE, SYSTEMS, exposure, forecasts, photos

DEST = ROOT/'data/v3_13/cmd'
MEMBERS = ROOT/'data/v3_13/input/m5_members.txt'
MEMBER_SHA = '74889d8e421ce1a74a4377653f1a48982f2763d7ac6d12e9b539311da6743185'
REFERENCE_URL = 'https://academic.oup.com/view-large/353698133'
M5_CENTER = (229.638,2.081)
M5_D = 7.48
C = np.linspace(.12,1.20,28)
APP = np.linspace(16.4,20.8,45)
ABS = np.linspace(-.3,8.3,87)
MU = np.arange(10+5*np.log10(5.),10+5*np.log10(14.)+.001,.025)


def fit_template(template,on,off,alpha):
    weights = 1/np.maximum(on+off*alpha**2,5.)
    delta = on-alpha*off
    absolute = (ABS[1:]+ABS[:-1])/2
    apparent = (APP[1:]+APP[:-1])/2
    significance,norm,amplitudes = [],[],[]
    for mu in MU:
        u = np.stack([np.interp(apparent-mu,absolute,row,left=0.,right=0.) for row in template])
        n = np.sum(weights*u*u)
        response = np.sum(weights*u*delta)
        norm.append(n)
        significance.append(response/np.sqrt(n))
        amplitudes.append(max(0.,response/n))
    return np.asarray(significance),np.asarray(norm),np.asarray(amplitudes)


def run():
    inputs = {'styles':check_templates(),'membership_source':'https://zenodo.org/records/4891252',
              'm5_member_hash':MEMBER_SHA,'external_scale':REFERENCE_URL}
    # The M5 reference distance does not enter masks, QRF predictions, or CMD fitting.
    SYSTEMS['M5'] = (M5_CENTER,M5_D)
    base = CLUSTERS/'M30'
    marker = base/'complete.json'
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    r = json.loads(marker.read_text())
    p = read_columns(verified(base/'sources.npz',r['outputs']['sources.npz']),
        ('TYPE','target','member','g0','r0','r','Mr_reference','training_or_calibration'))
    keep = p['target']&p['member']&~p['training_or_calibration']&(p['r']<21.)
    if not np.all(p['TYPE'][keep]=='PSF') or keep.sum()<50:
        raise ValueError('Insufficient eligible M30 template members')
    kernel = np.histogram2d(p['g0'][keep]-p['r0'][keep],p['Mr_reference'][keep],(C,ABS))[0]
    kernel = gaussian_filter(kernel,(1.,1.))
    kernel /= kernel.sum()
    t,ledger = photos('M5',inputs)
    gr = t['g0']-t['r0']
    t = subset(t,(gr>=C[0])&(gr<C[-1])&(t['r0']>=APP[0])&(t['r0']<APP[-1]))
    forecasts('south',t,inputs)
    area = exposure('M5',inputs)[0]
    valid = np.isfinite(t['qrf_kpc'][:,2])
    on,off = t['target']&valid,~t['target']&valid
    if not (on.any() and off.any()):
        raise ValueError('Missing common target and control rows')
    gr = t['g0']-t['r0']
    h,k = [np.histogram2d(gr[m],t['r0'][m],(C,APP))[0] for m in (on,off)]
    significance,norm,amplitude = fit_template(kernel,h,k,area[0]/area[1])
    d = 10**((MU-10)/5)
    j = int(np.argmax(significance))
    q1,q2 = [t['qrf_mass'][m,1:-1].sum(axis=0)/s for m,s in zip((on,off),area)]
    net = q1-q2
    km = (DISTANCE[1:]+DISTANCE[:-1])/2
    im = int(np.argmax(net))
    pub = np.loadtxt(verified(MEMBERS,MEMBER_SHA),usecols=(0,16),dtype=[('id','i8'),('pmember','f8')])
    claimed = np.isin(t['source_id'],pub['id'][pub['pmember']>.9])
    claimed &= unique_join(t['source_id'],t['source_id'])>=0
    rows = [dict(method='M30 CMD matched filter',peak_kpc=d[j],reference_kpc=M5_D,
        relative_bias=d[j]/M5_D-1,detection_score=significance[j],peak_at_scan_boundary=(j in (0,len(d)-1))),
        dict(method='Frozen QRF PDF sum',peak_kpc=km[im],reference_kpc=M5_D,
             relative_bias=km[im]/M5_D-1,peak_at_scan_boundary=(im in (0,len(km)-1)))]
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'admission.csv',ledger)
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        axes[0].plot(d,significance,color=BLUE)
        axes[0].set(title='M30 CMD template',ylabel='Conditional matched-filter score')
        axes[1].stairs(net,DISTANCE,color=ORANGE)
        axes[1].set(title='Frozen QRF',ylabel='Excess stars per cell per sq. deg.')
        for ax in axes:
            ax.axvline(M5_D,color=GRAY,ls='--')
            ax.set(xlim=(5,14),xlabel='Distance (kpc)')
        fig.suptitle('M5 · independent group distance comparison')
        save(fig,DEST,'m5',rows,dict(distance_kpc=d,score=significance,amplitude=amplitude,
           norm=norm,distance_bins=DISTANCE,qrf_excess=net,color_bins=C,apparent_bins=APP,
           abs_bins=ABS,target_cmd=h,control_cmd=k,template_cmd=kernel))
    atomic_npz(DEST/'m5_rows.npz',TYPE=t['TYPE'],ls_id=t['ls_id'],source_id=t['source_id'],
               common=on|off,target=t['target'],confirmed_member=claimed)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        native_target=int(on.sum()),native_control=int(off.sum()),target_members=int((on&claimed).sum()),
        excluded_distance_training=int(t['training_or_calibration'].sum()),area=area,template_members=int(keep.sum()),
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
