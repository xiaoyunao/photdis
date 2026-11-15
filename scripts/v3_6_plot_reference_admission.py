#!/usr/bin/env python3
"""Join the full DR11 and reference coverage counts for F01."""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from photdis.stage1 import require_compute
from photdis_figure_style import BLUE, GREEN, ORANGE, STYLE
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_6_catalog_observables import LABELS, OUT as CATALOG
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_reference_flow_audit import GATES, OUT as FLOWS, STAGES
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
from v3_6_vac_flow import OUT as VAC_MATCH


OUT = ROOT/'data/v3_6/figures/reference_admission'
SHELLS = ('0–2','2–5','5–10','10–20','20–50','50+','Unmeasured')
CLASSES = ('Unknown','Dwarf','Giant','Other')
HISTORY = ('Outside recorded history','Recorded history','QRF training')
KEYS = ('catalog_row','ls_id','TYPE','reference_kpc','observed_r','dereddened_g_r',
        'consensus_class','class_disagreement','baseline_v1_4_role','v3_0_role',
        'primary','strict','high_quality','qrf_training','nominal_sed_training',
        'selector_training','recorded_history')


def load_table(path,filename):
    m = path/'complete.json'
    receipt = json.loads(m.read_text())
    f = verified(path/filename,receipt['outputs'][filename])
    return m,f,receipt


def distance_cells(a,source,r_edges,c_edges):
    n = len(a['catalog_row'])
    if not np.all(a['TYPE']=='PSF') or np.any(a['strict']&~a['primary']):
        raise ValueError('F01 reference coverage requires native PSF and the nested cuts')
    d,r,c = (a[k] for k in ('reference_kpc','observed_r','dereddened_g_r'))
    finite = np.isfinite(d)&(d>0)
    di = np.where(finite,np.searchsorted(DISTANCE_EDGES,d,side='right')-1,len(SHELLS)-1)
    ri = np.searchsorted(r_edges,r,side='right')-1
    ci = np.searchsorted(c_edges,c,side='right')-1
    shape = (len(SHELLS),len(r_edges)-1,len(c_edges)-1,len(CLASSES),len(HISTORY))
    history = np.where(a['qrf_training'],2,a['recorded_history'].astype(int))
    group = np.ravel_multi_index((di,ri,ci,a['consensus_class']+1,history),shape)
    cell,inverse = np.unique(group,return_inverse=True)
    baseline = a['baseline_v1_4_role']
    r3,labels3 = np.unique(a['v3_0_role'],return_inverse=True)
    flags = {k:a[k] for k in ('primary','strict','high_quality','nominal_sed_training','selector_training',
                              'class_disagreement')}
    flags.update({f'v1_4_role_{i}':baseline==i for i in (-1,0,1,2,3)})
    flags.update({f'v3_0_role_{str(k) or "unassigned"}':labels3==i for i,k in enumerate(r3)})
    counts = np.bincount(inverse,minlength=len(cell))
    conditional = {k:np.bincount(inverse[v],minlength=len(cell)) for k,v in flags.items()}
    if counts.sum()!=n:
        raise ValueError(f'Unreconciled F01 reference rows: {source}')
    for p,k in enumerate(cell):
        i,j,l,cls,role = np.unravel_index(k,shape)
        yield dict(source=source,distance_kpc=SHELLS[i],r_lo=r_edges[j],r_hi=r_edges[j+1],
                   gr_lo=c_edges[l],gr_hi=c_edges[l+1],reference_class=CLASSES[cls],
                   qrf_history=HISTORY[role],reference_rows=counts[p],
                   **{name:values[p] for name,values in conditional.items()})


def unique_reference_cmd(parts,r_edges,c_edges):
    id_,r,c,d = (np.concatenate([p[k] for p in parts]) for k in ('catalog_row','observed_r',
                                                                 'dereddened_g_r','reference_kpc'))
    finite = np.isfinite(d)&(d>0)
    _,first = np.unique(id_[finite],return_index=True)
    unique = np.flatnonzero(finite)[first]
    hist = np.histogram2d(r[unique],c[unique],bins=(r_edges,c_edges))[0].astype('i8')
    if hist.sum()!=len(unique):
        raise ValueError('Observed CMD misses matched stars with a finite reference distance')
    return hist


def flow_steps(row):
    raw = int(row['raw_rows'])
    stage = {k:int(row[k]) for k in STAGES}
    if sum(stage.values())!=raw:
        raise ValueError('Exclusive terminal outcomes differ from the raw source count')
    supported = sum(stage[k] for k in GATES)
    pass_score = supported-sum(stage[k] for k in GATES[:2])
    return raw,supported,pass_score,stage['selected']


def export(fig,path):
    for ext in ('png','pdf'):
        fig.savefig(path.with_suffix(f'.{ext}'),dpi=300)
    plt.close(fig)


def interval(lo,hi):
    return f'<{hi:g}' if lo==-np.inf else f'≥{lo:g}' if hi==np.inf else f'{lo:g}–{hi:g}'


def render_cmd(parent,references,r_edges,c_edges,dest):
    series = (parent[0],references,parent[1],parent[2])
    high = max(2,*(x.max() for x in series))
    color_scale = LogNorm(1,high)
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(1,4,figsize=(16.9,4.75),layout='constrained')
        for ax,cell,title in zip(axs,series,('Supported PSFs','Unique distance references','Primary','Strict')):
            h = ax.imshow(np.ma.masked_less(cell,1),norm=color_scale,cmap='viridis',
                          origin='upper',aspect='auto')
            ax.set_title(f'{title}\nN = {cell.sum():,}')
            ax.set_xticks(range(len(c_edges)-1),[interval(a,b) for a,b in zip(c_edges[:-1],c_edges[1:])],
                          rotation=32)
            ax.set_yticks(range(len(r_edges)-1),[interval(a,b) for a,b in zip(r_edges[:-1],r_edges[1:])])
            ax.set_xlabel('Dereddened g − r [mag]')
            ax.grid(False)
        axs[0].set_ylabel('Observed r [mag]')
        fig.colorbar(h,ax=axs,label='Stars per observed bin',shrink=.82)
        export(fig,dest/'observed_cmd')


def render_flow(rows,dest):
    steps = ('Raw distance','Verified support','Dwarf-score pass','Selected')
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(2,2,figsize=(12.6,7.7),layout='constrained')
        for i,source in enumerate(('SpecDis','SPDist')):
            for j,shell in enumerate(('[2, 5)','[5, 10)')):
                ax = axs[i,j]
                for row in [r for r in rows if r['source']==source and r['distance_kpc']==shell]:
                    primary = row['cut']=='primary'
                    n = flow_steps(row)
                    ax.plot(np.arange(4)+(-.07 if primary else .07),n,marker='o',
                            color=ORANGE if primary else GREEN,label='Primary' if primary else 'Strict')
                ax.set(yscale='symlog',ylim=(-.3,None),title=f'{source} · {shell} kpc')
                ax.set_xticks(np.arange(len(steps)),steps)
                ax.set_ylabel('Source rows')
                ax.legend(frameon=False)
        export(fig,dest/'near_reference_flow')


def main():
    require_compute()
    cm,cp,catalog = load_table(CATALOG,'observables.npz')
    em = EVIDENCE/'complete.json'
    evidence = json.loads(em.read_text())
    data = load(cp)
    r_edges,c_edges = data['r_edges'],data['color_edges']
    parent = data['cell_counts'][:,0].sum(axis=-1)
    np.testing.assert_array_equal(parent.sum(axis=(1,2)),catalog['rows'])
    cells,parts = [],[]
    for source in evidence['rows']:
        filename = f'{source}.npz'
        a = read_columns(verified(EVIDENCE/filename,evidence['outputs'][filename]),KEYS)
        cells.extend(distance_cells(a,source,r_edges,c_edges))
        parts.append({k:a[k] for k in ('catalog_row','reference_kpc','observed_r','dereddened_g_r')})
    refs = unique_reference_cmd(parts,r_edges,c_edges)
    rows,inputs = [],[Path(__file__),cm,em]
    for source in ('SpecDis','SPDist'):
        fm,fp,flow = load_table(FLOWS/source.lower(),'distance.csv')
        match_name = str((VAC_MATCH/source.lower()/'complete.json').relative_to(ROOT))
        if evidence['inputs'][match_name]!=flow['inputs'][match_name]:
            raise ValueError(f'{source} flow and combined distance evidence use different matches')
        inputs.append(fm)
        with fp.open() as stream:
            subset = [dict(source=source,**r) for r in csv.DictReader(stream)]
        for cut in LABELS[1:]:
            cut_rows = [r for r in subset if r['cut']==cut]
            if sum(int(r['raw_rows']) for r in cut_rows)!=flow['raw_rows']:
                raise ValueError(f'Raw {source} denominator differs in {cut}')
            if sum(flow_steps(r)[-1] for r in cut_rows)!=flow['selected'][cut]:
                raise ValueError(f'Final {source} denominator differs in {cut}')
        rows.extend(subset)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    csv_rows(OUT/'reference_coverage_cells.csv',cells)
    csv_rows(OUT/'terminal_flow.csv',rows)
    csv_rows(OUT/'observed_cmd.csv',[
        dict(population=name,r_lo=r_edges[i],r_hi=r_edges[i+1],gr_lo=c_edges[j],gr_hi=c_edges[j+1],
             stars=int(a[i,j]))
        for name,a in zip((*LABELS,'unique_reference'),(*parent,refs))
        for i,j in np.ndindex(refs.shape)])
    render_cmd(parent,refs,r_edges,c_edges,OUT)
    render_flow(rows,OUT)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        caption=('F01 support and losses. The three catalog columns and the deduplicated '
                 'positive-distance reference column share observed bins. The raw VAC flow '
                 'uses each catalog’s own row denominator. The terminal table preserves all '
                 'exclusive association, native admission and selection outcomes, with separate '
                 'overlap flags. Source-specific coverage cells can contain the same star.'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
