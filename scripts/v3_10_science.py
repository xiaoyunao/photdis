#!/usr/bin/env python3
"""Verify the working indices and test M92 distance information against a control."""
import csv
import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, plt, save, spatial
from v3_9_figures import CHANNELS, select_channel
from v3_9_references import join, subset
from v3_10_coverage import DEST as COVERAGE, m92_zones
from v3_10_evidence import DEST as EVIDENCE
from v3_10_figures import check_templates, sky_coord
from v3_10_redrock import DEST as REDROCK
from v3_10_working_sample import DEST as WORK, DOMAINS, shell_mass

DEST = ROOT/'data/v3_10/science'
M92_DISTANCE = 8.50
DISTANCE = np.arange(1.,20.1,.25)
COLOR = (-np.inf,.50,.80,1.40,np.inf)


def working_index(side, inputs):
    root = WORK/side
    parts, ids = [], []
    for shard in range(48):
        path = root/f'{shard:02d}.json'
        r = json.loads(path.read_text())
        if r['side']!=side or r['group']!=shard:
            raise ValueError('Incomplete working sample')
        inputs[str(path.relative_to(ROOT))] = sha(path)
        if not r['rows']:
            continue
        name = f'{shard:02d}.npz'
        part = verified(root/name,r['outputs'][name])
        rows = read_columns(part,('TYPE','RA','DEC','g0','r0','state','ls_id','qrf_median_kpc','radial_mass'))
        if not np.all(rows['TYPE']=='PSF') or r['selected']!=int((rows['state']==4).sum()):
            raise ValueError('Working sample reported a changed membership')
        ids.append(rows['ls_id'])
        parts.append((part,rows))
    idx = np.concatenate(ids)
    if len(np.unique(idx))!=len(idx):
        raise ValueError('Duplicate native source in working index')
    return parts,join([r for _,r in parts])


def validation(side, selection, inputs):
    marker = EVIDENCE/DOMAINS[side]/'complete.json'
    r = json.loads(marker.read_text())
    path = verified(marker.parent/'references.npz',r['outputs']['references.npz'])
    t = read_columns(path,('TYPE','ls_id','r','g0','r0','RA','DEC','evaluation_row','sky_role','class_label',
        'gaia_state','reference_kpc','specdis_quality','specdis_kpc','spdist_quality','spdist_kpc',
        'qrf_quantiles_kpc','redrock_quality_assessed','redrock_SPECTYPE','rvs_logg','rvs_teff','rvs_feh'))
    j = unique_join(t['ls_id'],selection['ls_id'])
    keep = (j>=0)&t['evaluation_row']&(t['sky_role']==4)
    t = subset(t,keep)
    s = selection['state'][j[keep]]
    labels = t['class_label']
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    summaries = []
    stats = []
    ref = np.full(len(s),np.nan)
    rr_marker = REDROCK/'complete.json'
    rr_meta = json.loads(rr_marker.read_text())
    z = read_columns(verified(REDROCK/'qualified.npz',rr_meta['outputs']['qualified.npz']),
                     ('ls_id','TYPE','qualified'))
    jj = unique_join(t['ls_id'],z['ls_id'])
    found = np.flatnonzero(jj>=0)
    np.testing.assert_array_equal(t['TYPE'][found],z['TYPE'][jj[found]])
    assessed = found[z['qualified'][jj[found]]!='']
    rr = np.where(t['redrock_quality_assessed'],t['redrock_SPECTYPE'].astype('U8'),'')
    conflict = assessed[(rr[assessed]!='')&(rr[assessed]!=z['qualified'][jj[assessed]])]
    rr[assessed] = z['qualified'][jj[assessed]]
    rr[conflict] = ''
    inputs[str(rr_marker.relative_to(ROOT))] = sha(rr_marker)
    for channel in reversed(CHANNELS):
        d,q = select_channel(t,channel)
        ref[q&(d>0)] = d[q&(d>0)]
    c = t['g0']-t['r0']
    for a,b in ((-np.inf,np.inf),*zip(COLOR[:-1],COLOR[1:])):
        use = (c>=a)&(c<b)
        for name,k in (('dwarf',labels==0),('giant',labels==1),('other',labels==2),
                       ('qualified_qso',rr=='QSO'),('qualified_galaxy',rr=='GALAXY')):
            samples = s[use&k]
            summaries.append(dict(side=side,gr_min=a,gr_max=b,class_label=name,
                supported=int((samples>=2).sum()),selected=int((samples==4).sum())))
        for name in CHANNELS:
            d,q = select_channel(t,name)
            for lo,hi in ((0,2),(2,5),(5,10),(10,np.inf)):
                for label in (-2,0,1,2):
                    cls = ((labels==label) if label!=-2 else np.ones(len(s),bool))
                    j = use&cls&q&(d>=lo)&(d<hi)&(d>0)
                    for stage,k in (('supported',s>=2),('selected',s==4)):
                        v = j&k
                        stats.append(dict(side=side,gr_min=a,gr_max=b,class_label=label,
                            source=name,d_min=lo,d_max=hi,stage=stage,**distance_summary(
                            10+5*np.log10(t['qrf_quantiles_kpc'][v]),10+5*np.log10(d[v]))))
    populations = []
    los = (-np.inf,3.2,4.2,np.inf)
    temps = (-np.inf,4500.,5500.,7000.,np.inf)
    q = t['qrf_quantiles_kpc']
    for name,d,good in (('SpecDis',t['specdis_kpc'],t['specdis_quality']),
                        ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
        frame = (labels==2)&(s>=2)&good&(d>=5)&(d<10)
        for a,b in zip(los[:-1],los[1:]):
            for c,e in zip(temps[:-1],temps[1:]):
                k = frame&(t['rvs_logg']>=a)&(t['rvs_logg']<b)&(t['rvs_teff']>=c)&(t['rvs_teff']<e)
                if not k.any():
                    continue
                feh = t['rvs_feh'][k]
                feh = feh[np.isfinite(feh)]
                populations.append(dict(side=side,source=name,logg_min=a,logg_max=b,teff_min=c,teff_max=e,
                    selected=int((k&(s==4)).sum()),feh_measured=len(feh),
                    median_feh=float(np.median(feh)) if len(feh) else None,
                    **distance_summary(10+5*np.log10(q[k]),10+5*np.log10(d[k]))))
    use = (s==4)&(labels==0)&np.isfinite(ref)
    return summaries,stats,populations,dict(RA=t['RA'][use],DEC=t['DEC'][use],distance=ref[use])


def cluster(parts):
    surveys = []
    cuts = 10+5*np.log10(DISTANCE)
    for path,p in parts:
        on,off = m92_zones(p['RA'],p['DEC'])
        keep = (on|off)&(p['state']>=2)
        if not keep.any():
            continue
        t = read_columns(path,('TYPE','ls_id','part_index','pdf_rows'))
        t.update(**{k:p[k] for k in ('state','g0','r0','radial_mass')},target=on)
        t = subset(t,keep)
        with path.with_suffix('.csv').open() as f:
            pointers = list(csv.DictReader(f))
        mass = np.full((len(t['ls_id']),len(cuts)-1),np.nan)
        for i in np.unique(t['part_index']):
            j = np.flatnonzero(t['part_index']==i)
            point = pointers[i]
            a = t['pdf_rows'][j]
            z = read_columns(verified(ROOT/point['file'],point['pdf_sha256']),
                             ('TYPE','ls_id','qrf_mu_edges','qrf_mu_mass'))
            np.testing.assert_array_equal(z['ls_id'][a],t['ls_id'][j])
            np.testing.assert_array_equal(z['TYPE'][a],t['TYPE'][j])
            mass[j] = shell_mass(z['qrf_mu_edges'][a],z['qrf_mu_mass'][a],cuts)
        if not np.isfinite(mass).all():
            raise ValueError('Missing full PDF for a bright supported M92-field object')
        surveys.append(dict(**t,pdf_mass=mass))
    return join(surveys)


def cluster_report(t, areas):
    f,axs = plt.subplots(2,3,figsize=(18,9),layout='constrained',sharex=True)
    results,counts = {},[]
    c = t['g0']-t['r0']
    target = t['target']
    for n,(a,b) in enumerate(((-np.inf,np.inf),(-np.inf,.50),(.50,.80),(.80,1.40),(1.40,np.inf))):
        k = (c>=a)&(c<b)
        for label,g in (('supported',t['state']>=2),('dwarf score',t['state']>=3),('working sample',t['state']==4)):
            signal,control = k&g&target,k&g&~target
            x = np.stack([t['pdf_mass'][signal].sum(axis=0)/areas[0],t['pdf_mass'][control].sum(axis=0)/areas[1]])
            net = x[0]-x[1]
            centers = .5*(DISTANCE[1:]+DISTANCE[:-1])
            shell = (centers>=5)&(centers<10)
            peak = np.flatnonzero(shell)[np.argmax(net[shell])]
            around = (centers>=7.5)&(centers<9.5)
            counts.append(dict(color_min=a,color_max=b,selection=label,target_n=int(signal.sum()),control_n=int(control.sum()),
                target_area_deg2=float(areas[0]),control_area_deg2=float(areas[1]),
                target_all_per_deg2=float(signal.sum()/areas[0]),control_all_per_deg2=float(control.sum()/areas[1]),
                target_7_5_9_5_per_deg2=float(x[0,around].sum()),control_7_5_9_5_per_deg2=float(x[1,around].sum()),
                peak_between_5_and_10_kpc=float(centers[peak]) if net[peak]>0 else None,
                peak_excess_per_deg2=float(max(net[peak],0.))))
            results[f'{n}_{label}'] = x
            if label=='working sample':
                ax = axs.flat[n]
                ax.stairs(x[0],DISTANCE,color=BLUE,label='Cluster annulus')
                ax.stairs(x[1],DISTANCE,color=ORANGE,label='Control annulus')
                ax.axvline(M92_DISTANCE,color=GRAY,ls='--',lw=1.25)
                ax.set(xlim=(2,12),xlabel='Distance (kpc)',ylabel='Stars per cell per sq. deg.',
                       title='All colors' if n==0 else f'Dereddened g − r · {a:g} to {b:g}')
    axs.flat[-1].axis('off')
    axs[0,0].legend()
    f.suptitle('M92 · photometry-only working sample · reference distance marked in gray')
    save(f,DEST,'m92_distance',counts,dict(distance_bins=DISTANCE,**results))


def main():
    require_compute()
    lock = check_templates()
    inputs = {}
    DEST.mkdir(parents=True,exist_ok=True)
    counts,selection,stats,rows,refs,populations = [],[],[],[],[],[]
    for side in DOMAINS:
        parts,p = working_index(side,inputs)
        a,b,c,ref = validation(side,p,inputs)
        selection.extend(a)
        stats.extend(b)
        populations.extend(c)
        refs.append(ref)
        l,lat = sky_coord(p['RA'],p['DEC'])
        good = p['state']==4
        rows.append(dict(l=l[good],b=lat[good],d=p['qrf_median_kpc'][good]))
        counts.append(dict(side=side,**{f'state_{i}':int((p['state']==i).sum()) for i in range(5)}))
        if side=='north':
            field = cluster(parts)
    csv_rows(DEST/'working_sample_counts.csv',counts)
    csv_rows(DEST/'working_reference_selection.csv',selection)
    csv_rows(DEST/'working_reference_distances.csv',stats)
    csv_rows(DEST/'intermediate_star_parameters.csv',populations)
    p,t = join(rows),join(refs)
    l,b = sky_coord(t['RA'],t['DEC'])
    n,m = len(p['d']),len(l)
    with plt.rc_context(STYLE):
        spatial(DEST,dict(primary=np.ones(n+m,bool)),np.r_[p['d'],np.full(m,np.nan)],
                np.r_[p['l'],l],np.r_[p['b'],b],np.r_[np.full(n,np.nan),t['distance']],
                np.r_[np.zeros(n,bool),np.ones(m,bool)])
        marker = COVERAGE/'north/complete.json'
        a = json.loads(marker.read_text())
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        area = read_columns(verified(marker.parent/'area.npz',a['outputs']['area.npz']),
                            ('m92_area_deg2',))['m92_area_deg2'][3]
        if not field or not np.all(area>0):
            raise ValueError('M92 and the control annulus require nonzero three-band-depth areas')
        atomic_npz(DEST/'m92_sources.npz',**field)
        cluster_report(field,area)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        style_lock_sha256=lock,working_rows=n,qualified_dwarf_reference_rows=m,
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    main()
