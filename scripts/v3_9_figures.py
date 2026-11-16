#!/usr/bin/env python3
"""Direct distance, color–magnitude and individual-source atlas for v3.9."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from photdis.stage1 import galactic, require_compute
from photdis.v3.metrics import distance_summary
from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_candidate import read_cache
from v3_9_catalog import OUT
from v3_9_references import OLDER, join, matched, sorted_key, subset

REGIONS = {'south_faint':'DECam 20–23 mag','south_low':'DECam low latitude','north':'BASS/MzLS native'}
CHANNELS = ('Gaia','SpecDis','SPDist')
COLORS = (BLUE,GREEN,ORANGE)
SHELLS = ((0,1),(1,5),(5,10),(0,10))


def read_inventory(side):
    root = OUT/'validation'/side
    groups = json.loads((OUT/side/'inventory.json').read_text())['groups']
    records = [root/f'{i:02d}.json' for i in range(len(groups))]
    metadata = [json.loads(p.read_text()) for p in records]
    tables,sky,refs = [],[],[]
    for path,receipt in zip(records,metadata):
        for name,digest in receipt['outputs'].items():
            data = verified(path.parent/name,digest)
            if name.endswith('_count.csv'):
                tables.append(data)
            elif name.endswith('_sky.npz'):
                sky.append(data)
            elif name.endswith('_reference.npz'):
                refs.append(data)
    return tables,sky,refs,{p.name:sha(p) for p in records}


def normalize_projection(l,b):
    return np.deg2rad((180-l)%360-180),np.deg2rad(b)


def save(fig, root, name):
    for ext in ('png','pdf'):
        fig.savefig(root/f'{name}.{ext}',dpi=300)
    plt.close(fig)


def select_channel(t, channel):
    if channel=='Gaia':
        return t['reference_kpc'],t['gaia_state']==6
    label = channel.lower()
    return t[label+'_kpc'],t[label+'_quality']


def evaluation_rows(side, reference, south_training):
    if side=='north':
        # Match the untouched sky role in v3_9_references.northern_fit.
        a = np.floor(reference['RA']/10).astype(int)
        d = np.floor((reference['DEC']+90)/10).astype(int)
        return (a+3*d)%5==4
    sid = reference['source_id']
    identity = np.where(sid>0,sid,-reference['ls_id'])
    return ~np.isin(identity,south_training)


def attach_raw_spectral_labels(reference, inputs):
    """Recover RVS class evidence outside the historical photometric window."""
    archive = OLDER.parent
    roster = archive/'sources.json'
    audit = archive/'source_evidence.json'
    declared_roster = json.loads(roster.read_text())['source_sha256']
    evidence = json.loads(audit.read_text())
    if evidence['input_sha256']['sources.npz'] != declared_roster:
        raise ValueError('RVS labels and source positions use different raw inventories')
    s = read_columns(verified(archive/'sources.npz',declared_roster),('source_id','ra','dec'))
    e = read_columns(verified(archive/'source_evidence.npz',evidence['output_sha256']),
                     ('source_id','original_row','class_label','class_identity_conflict','target_identity_conflict'))
    np.testing.assert_array_equal(e['original_row'],np.arange(len(s['source_id'])))
    np.testing.assert_array_equal(e['source_id'],s['source_id'])
    eligible = ((s['source_id']>0)&(e['class_label']>=0)
                &~e['class_identity_conflict']&~e['target_identity_conflict'])
    raw = dict(reference_ra=s['ra'],reference_dec=s['dec'],
               _gaia=sorted_key(s['source_id'],np.flatnonzero(eligible)))
    j = matched(reference,raw)
    label = np.full(len(j),-1,'i1')
    label[j>=0] = e['class_label'][j[j>=0]]
    old = reference['class_label']
    conflict = (label>=0)&(old>=0)&(old!=label)
    nonstellar = np.isin(reference['spectype'],('QSO','GALAXY'))
    supplement = (label>=0)&(old<0)&~nonstellar
    reference['class_label_historical'] = old
    reference['class_label_raw_rvs'] = label
    reference['class_label'] = np.where(conflict,-1,np.where(supplement,label,old)).astype('i1')
    reference['class_label_conflict'] = conflict
    inputs[str(roster.relative_to(ROOT))] = sha(roster)
    inputs[str(audit.relative_to(ROOT))] = sha(audit)
    return dict(raw_rvs_only=int(supplement.sum()),class_conflicts=int(conflict.sum()))


def distance_plots(t, root):
    q = t['qrf_quantiles_kpc']
    metrics = []
    fig,axes = plt.subplots(3,3,figsize=(15,10),sharex='col',
                            gridspec_kw={'height_ratios':[3,1,1]},layout='constrained')
    cmd,cax = plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True,layout='constrained')
    gr = t['g0']-t['r0']
    for col,(channel,color) in enumerate(zip(CHANNELS,COLORS)):
        d,quality = select_channel(t,channel)
        use = quality&np.isfinite(q[:,2])&(q[:,2]>0)&(d>0)
        dist,pred = d[use],q[use,2]
        if len(dist)>1200:
            axes[0,col].hexbin(dist,pred,xscale='log',yscale='log',mincnt=1,gridsize=50,
                              cmap='Blues',norm=LogNorm())
        else:
            axes[0,col].scatter(dist,pred,s=9,c=color,alpha=.72)
        limits = (.02,40)
        axes[0,col].plot(limits,limits,'--',color=GRAY,lw=1)
        axes[0,col].set(xlim=limits,ylim=limits,xscale='log',yscale='log',title=f'{channel} · N = {len(dist):,}')
        residual = pred/dist-1
        axes[1,col].scatter(dist,residual,s=5,c=color,alpha=.4,rasterized=True)
        axes[1,col].axhline(0,color=GRAY,ls='--',lw=1)
        axes[1,col].set(ylim=(-1,1),xscale='log')
        axes[1,col].text(.02,.05,f'Above panel: {(residual>1).sum():,}',
                         transform=axes[1,col].transAxes,fontsize=8)
        # The third panel shows the distance-matched reference distribution.
        axes[2,col].hist(dist,bins=np.geomspace(*limits,32),histtype='step',color=color,lw=1.3)
        axes[2,col].set(xscale='log',xlabel='Reference distance (kpc)')
        for axis,di,title in zip(cax[:,col],(dist,pred),(f'{channel} distance','Photometric distance, same stars')):
            mr = t['r0'][use]-10-5*np.log10(di)
            if len(mr)>1200:
                axis.hexbin(gr[use],mr,mincnt=1,gridsize=50,cmap='Blues',norm=LogNorm())
            else:
                axis.scatter(gr[use],mr,s=7,c=color,alpha=.55)
            axis.set(title=title)
        cax[1,col].set(xlabel=r'$(g-r)_0$ (mag)')
        for floor,ceil in ((0,2),(2,5),(5,10),(10,np.inf)):
            for lab in (-2,-1,0,1,2):
                for tier in (6,7):
                    for a,b in ((-np.inf,20),(20,21),(21,22),(22,23)):
                        mask = use&(d>=floor)&(d<ceil)&(t['row_state']==tier)&(t['r']>=a)&(t['r']<b)
                        if lab!=-2:
                            mask &= t['class_label']==lab
                        metrics.append(dict(source=channel,distance_min=floor,distance_max=ceil,
                            observed_r_min=a,observed_r_max=b,photometry_level=tier,class_label=lab,
                            **distance_summary(10+5*np.log10(q[mask]),10+5*np.log10(d[mask]))))
    axes[0,0].set_ylabel('Photometric distance (kpc)')
    axes[1,0].set_ylabel(r'$d_{\mathrm{phot}}/d_{\mathrm{ref}}-1$')
    axes[2,0].set_ylabel('Reference count')
    cax[0,0].set(ylim=(15,-6),xlim=(-.6,2.5),ylabel=r'$M_r$ (mag)')
    cax[1,0].set_ylabel(r'$M_r$ (mag)')
    csv_rows(root/'distance_by_source.csv',metrics)
    save(fig,root,'distance_comparisons')
    save(cmd,root,'same_star_cmd')


def evidence_plots(t, root):
    fig,axs = plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    w = t['signed_gaia_observation_metrics']
    use = np.isfinite(w[:,0])
    obs = t['signed_parallax_mas'][use]
    pred = w[use,3]
    if use.any():
        xx = np.quantile(np.r_[obs,pred],[.005,.995])
        if xx[1]>xx[0]:
            bins = np.linspace(*xx,42)
            axs[0].hist(obs,bins=bins,histtype='step',lw=1.4,color=BLUE,label='Observed signed Gaia')
            axs[0].hist(pred,bins=bins,histtype='step',lw=1.4,color=ORANGE,label='Photo PDF predictive mean')
    axs[0].axvline(0,c=GRAY,ls=':',lw=1)
    axs[0].set(xlabel='Corrected parallax (mas)',ylabel='Count',title=f'Weak signed Gaia · N = {len(obs):,}')
    if use.any():
        axs[0].legend(fontsize=9)
    known = (t['class_label']>=0)|np.isin(t['spectype'],['QSO','GALAXY'])
    for label,mask,col in (('Dwarfs',t['class_label']==0,BLUE),('Giants',t['class_label']==1,ORANGE),
                           ('QSO',t['spectype']=='QSO',GREEN),('Galaxies',t['spectype']=='GALAXY',GRAY)):
        k = known&mask&np.isfinite(t['dwarf_score'])
        if k.any():
            axs[1].hist(t['dwarf_score'][k],bins=np.linspace(0,1,25),histtype='step',color=col,label=label)
    if axs[1].get_legend_handles_labels()[0]:
        axs[1].legend(fontsize=9)
    axs[1].set(xlabel='Dwarf score',ylabel='Control count',title='Spectral controls')
    truth = t['reference_kpc']
    good = np.isfinite(truth)&np.isfinite(t['qrf_quantiles_kpc'][:,2])
    residual = (5*np.log10(t['qrf_quantiles_kpc'][good,2]/truth[good]))
    axs[2].scatter((t['g0']-t['r0'])[good],t['error_grz'][good].max(axis=1),
                   c=np.clip(residual,-2,2),cmap='coolwarm',vmin=-2,vmax=2,s=9,rasterized=True)
    axs[2].set(xlabel=r'$(g-r)_0$ (mag)',ylabel='Largest grz error (mag)',title='Gaia distance residuals')
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=plt.Normalize(-2,2),cmap='coolwarm'),ax=axs[2],label='Modulus residual (mag)')
    table = []
    for tier in (6,7):
        for floor,ceil in ((-np.inf,20),(20,21),(21,22),(22,23)):
            mask = (t['r']>=floor)&(t['r']<ceil)&(t['row_state']==tier)
            for name,kind in (('all',mask),('dwarf',mask&(t['class_label']==0)),
                              ('giant',mask&(t['class_label']==1)),('QSO',mask&(t['spectype']=='QSO')),
                              ('GALAXY',mask&(t['spectype']=='GALAXY'))):
                dwarf = kind&(t['dwarf_score']>=.65)
                run = kind&t['nominal_evaluated']
                matched = run&~t['nominal_numerical_flag']
                weak = kind&np.isfinite(w[:,0])
                table.append(dict(control=name,photometry_level=tier,min_r=floor,max_r=ceil,
                    n=int(kind.sum()),qrf_supported=int((kind&np.isfinite(t['qrf_quantiles_kpc'][:,2])).sum()),
                    nominal_evaluated=int(run.sum()),with_valid_nominal=int(matched.sum()),
                    passing_dwarf_point65=int(dwarf.sum()),
                    both_gates=int((matched&dwarf&(t['nominal_qnear']>=.90)).sum()),
                    weak_n=int(weak.sum()),weak_negative=int((weak&(t['signed_parallax_mas']<0)).sum()),
                    weak_delta_log_score_nats=float(np.mean(w[weak,0]-w[weak,5])) if weak.any() else None))
    csv_rows(root/'signed_gaia_and_spectral_controls.csv',table)
    save(fig,root,'reference_controls')


def atlas(p,t,root):
    m = p['qrf_median_kpc']
    l,b = galactic(p['RA'],p['DEC'])
    x,y = normalize_projection(l,b)
    rl,rb = galactic(t['RA'],t['DEC'])
    rx,ry = normalize_projection(rl,rb)
    rd = t['reference_kpc']
    spdist = np.where(t['spdist_quality'],t['spdist_kpc'],np.nan)
    d21 = np.where(t['specdis_quality'],t['specdis_kpc'],spdist)
    rd = np.where(np.isfinite(rd),rd,d21)
    fig,ax = plt.subplots(4,4,figsize=(21,13),subplot_kw={'projection':'aitoff'},layout='constrained')
    labels = ('Precise survey predictions','Qualified distance references','Spectroscopic dwarfs','Spectroscopic giants')
    groups = (np.isfinite(rd),(t['class_label']==0)&np.isfinite(rd),(t['class_label']==1)&np.isfinite(rd))
    counts = []
    for col,(lo,hi) in enumerate(SHELLS):
        source = (m>=lo)&(m<hi)
        ax[0,col].scatter(x[source],y[source],color=BLUE,s=.38,alpha=.55,linewidths=0,rasterized=True)
        for row,k in enumerate(groups,1):
            plot = k&(rd>=lo)&(rd<hi)
            ax[row,col].scatter(rx[plot],ry[plot],color=(GRAY,GREEN,ORANGE)[row-1],s=1.8,
                                alpha=.65,linewidths=0,rasterized=True)
            if not plot.any():
                ax[row,col].text(.5,.12,'No qualified sources',ha='center',transform=ax[row,col].transAxes,
                                 color=GRAY,fontsize=8)
            counts.append(dict(category=labels[row],distance_min_kpc=lo,distance_max_kpc=hi,n=int(plot.sum())))
        counts.append(dict(category=labels[0],distance_min_kpc=lo,distance_max_kpc=hi,n=int(source.sum())))
        ax[0,col].set_title(f'{lo}–{hi} kpc')
    for row,name in enumerate(labels):
        ax[row,0].set_ylabel(name,fontsize=12)
    fig.suptitle('Galactic sky positions · one point per retained source')
    save(fig,root,'aitoff_source_positions')
    csv_rows(root/'aitoff_counts.csv',counts)
    k = (m>0)&(m<10)
    ell,lat = np.deg2rad(l[k]),np.deg2rad(b[k])
    xyz = (m[k]*np.cos(lat)*np.cos(ell),m[k]*np.cos(lat)*np.sin(ell),m[k]*np.sin(lat))
    f,axes = plt.subplots(1,3,figsize=(15,5.2),layout='constrained')
    for a,(i,j) in zip(axes,((0,1),(0,2),(1,2))):
        if k.any():
            a.hist2d(xyz[i],xyz[j],bins=120,range=[[-10,10],[-10,10]],norm=LogNorm(),cmap='Blues')
        a.set(xlabel='XYZ'[i]+' (kpc)',ylabel='XYZ'[j]+' (kpc)',aspect='equal')
    f.suptitle('Selected stars in heliocentric coordinates · distance median below 10 kpc')
    save(f,root,'heliocentric_xyz')
    f,axes = plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    gr = p['g0']-p['r0']
    if len(gr):
        axes[0].hexbin(gr,p['r0']-p['z0'],mincnt=1,gridsize=64,norm=LogNorm(),cmap='Blues')
        axes[1].hexbin(p['EBV'],p['FRACFLUX_R'],mincnt=1,gridsize=48,norm=LogNorm(),cmap='Blues')
        ok = np.isfinite(p['PSFDEPTH_R'])&(p['PSFDEPTH_R']>0)
        depth = 22.5-2.5*np.log10(5./np.sqrt(p['PSFDEPTH_R'][ok]))
        axes[2].hist(depth,bins=40,histtype='step',color=BLUE,lw=1.4)
    axes[0].set(xlabel=r'$(g-r)_0$ (mag)',ylabel=r'$(r-z)_0$ (mag)',title='Color–color diagram')
    axes[1].set(xlabel='SFD reddening (mag)',ylabel='r-band contamination fraction',title='Extinction and crowding')
    axes[2].set(xlabel='5σ r-band depth (mag)',ylabel='Source count',title='Depth at source locations')
    save(f,root,'photometry_environment')


def run(region, figure_set):
    side = 'north' if region=='north' else 'south'
    tables,sky,refs,inputs = read_inventory(side)
    out = OUT/figure_set/region
    out.mkdir(parents=True,exist_ok=True)
    counts = []
    for table in tables:
        with table.open() as stream:
            counts.extend(r for r in csv.DictReader(stream) if r.get('region')==region)
    csv_rows(out/'sweep_counts.csv',counts)
    p = join([subset(z,(z['region']==region)&z['primary_survey_owner']) for path in sky
              if (z:=read_columns(path,('region','ls_id','TYPE','RA','DEC','r','g0','r0','z0','EBV',
                'FRACFLUX_R','PSFDEPTH_R','qrf_median_kpc','primary_survey_owner')))])
    t = join([subset(z,z['region']==region) for path in refs if (z:=read_columns(path,(
        'region','ls_id','source_id','TYPE','RA','DEC','r','g0','r0','error_grz','row_state','reference_kpc','gaia_state',
        'specdis_kpc','spdist_kpc','specdis_quality','spdist_quality','class_label','spectype',
        'qrf_quantiles_kpc','dwarf_score','signed_parallax_mas','signed_gaia_observation_metrics',
        'nominal_qnear','nominal_evaluated','nominal_numerical_flag')))])
    if not p or not t or not np.all(p['TYPE']=='PSF') or not np.all(t['TYPE']=='PSF'):
        raise ValueError(f'Real native PSF catalog and reference data are required for {region}')
    label_audit = attach_raw_spectral_labels(t,inputs)
    training = None
    if side=='south':
        roster = read_cache(ROOT/'data/v3_0/controls','training.npz')
        training = read_columns(roster,('identity',))['identity']
        inputs[str(roster.relative_to(ROOT))] = sha(roster)
    t['evaluation_row'] = evaluation_rows(side,t,training)
    reference = subset(t,t['evaluation_row'])
    with plt.rc_context(STYLE):
        distance_plots(reference,out)
        evidence_plots(reference,out)
        atlas(p,t,out)
    atomic_npz(out/'figure_reference_rows.npz',**t)
    save_json(out/'complete.json',dict(region=region,inputs=inputs,projected_rows=len(p['ls_id']),
              reference_rows=len(t['ls_id']),evaluation_reference_rows=len(reference['ls_id']),
              label_audit=label_audit,
              job=os.getenv('SLURM_JOB_ID'),
              outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('region',choices=tuple(REGIONS))
    args.add_argument('--figure-set',choices=('figures','figures_labeled'),default='figures')
    a = args.parse_args()
    run(a.region,a.figure_set)
