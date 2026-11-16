#!/usr/bin/env python3
"""Paper plates from the qualified distance tables and registered experiments."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE

ROOT = Path(__file__).resolve().parents[1]
EARLIER = ROOT/'manifests/v3_6/distance_evidence/distance_cells.csv'
SHELLS = ('0–2','2–5','5–10','10–20','20–50','>50')
SOURCES = ('Gaia','SpecDis','SPDist')
COLORS = (BLUE,ORANGE,GREEN)
WORKPOINTS = ('primary','strict','lower_dwarf','lower_near','lower_both','lower_both_bright')
LABELS = ('Primary','Strict','Lower dwarf','Lower near','Lower both','Lower both, r < 18')


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def choose(rows,**filters):
    return [r for r in rows if all(r[field]==v for field,v in filters.items())]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def vals(rows,key):
    return np.asarray([float(r[key]) if r[key] else np.nan for r in rows])


def save(fig,name,rows,sources,figures,manifests):
    for suffix in ('png','pdf'):
        fig.savefig(figures/f'{name}.{suffix}',dpi=300)
    plt.close(fig)
    csv_path = manifests/f'{name}.csv'
    with csv_path.open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    files = {f'{name}.{suffix}':digest(figures/f'{name}.{suffix}') for suffix in ('png','pdf')}
    files[csv_path.name] = digest(csv_path)
    (manifests/f'{name}.json').write_text(json.dumps(
        dict(inputs={p.name:digest(p) for p in sources},outputs=files,dpi=300),indent=2)+'\n')


def performance():
    rows = choose(table(EARLIER),quality='high_quality',history='outside_qrf_train',observed='all')
    fig,axes = plt.subplots(2,3,figsize=(14.6,8.),layout='constrained',sharex=True)
    final = []
    for i,name in enumerate(('primary','strict')):
        for source,color in zip(SOURCES,COLORS):
            label,cat = ('Gaia (all classes)','all') if source=='Gaia' else (f'{source} (dwarfs)','dwarf')
            group = sorted(choose(rows,source=source,selection=name,reference_class=cat),
                           key=lambda r:float(r['reference_lo_kpc']))[:3]
            if len(group)!=3:
                raise ValueError('Missing qualified performance comparison')
            final.extend(group)
            for ax,key in zip(axes[i],('median_relative','nmad_relative','coverage68')):
                x = np.arange(3)+.08*(SOURCES.index(source)-1)
                y = 100*vals(group,key)
                # Keep small samples visible with a hollow symbol.
                dense = vals(group,'n')>=30
                ax.plot(x,y,lw=1.2,color=color,label=label)
                ax.scatter(x[dense],y[dense],color=color,s=38,zorder=3)
                ax.scatter(x[~dense],y[~dense],edgecolor=color,facecolor='white',s=38,zorder=3)
        for ax,title in zip(axes[i],('Median relative bias','Relative residual NMAD','68% interval agreement')):
            ax.set_title(title)
            ax.set_ylabel(f'{name.capitalize()} (%)')
            ax.set_xticks(range(3),SHELLS[:3])
        axes[i,0].axhline(0,color=GRAY,lw=.9)
        axes[i,2].axhline(68,color=GRAY,lw=.9,ls='--')
    axes[0,0].legend(loc='best',fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel('Reference distance (kpc)')
    fig.suptitle('Distance performance for qualified references outside QRF training\n'
                 'Open markers show fewer than 30 stars',fontsize=17)
    return fig,'F02_qualified_distances',final,[EARLIER]


def selection(root):
    source = root/'selection/reference.csv'
    a = choose(table(source),domain='all',history='all',reference_shell='5–10')
    fig,axes = plt.subplots(2,2,figsize=(13.4,8.4),layout='constrained',sharex=True)
    final = []
    for ref,color in zip(SOURCES[1:],COLORS[1:]):
        for title,ax,cls,key in (
            ('Known dwarfs retained',axes[0,0],'dwarf','retention'),
            ('Known giants retained',axes[0,1],'giant','retention'),
            ('Known dwarfs, distance bias',axes[1,0],'dwarf','median_relative'),
            ('Known dwarfs, |distance modulus residual| > 1 mag',axes[1,1],'dwarf','severe_fraction')):
            rows = [choose(a,source=ref,reference_class=cls,selection=w)[0] for w in WORKPOINTS]
            subset = vals(rows,'n')
            value = subset/vals(rows,'reference_denominator') if key=='retention' else vals(rows,key)
            ax.plot(range(6),100*value,'o-',color=color,label=ref)
            ax.set_title(title,fontsize=15)
            ax.set_ylabel('Percent')
            final.extend(dict(**r,metric=key,value=v) for r,v in zip(rows,value))
    axes[0,0].legend()
    axes[1,0].axhline(0,color=GRAY,lw=.9)
    for ax in axes[1]:
        ax.set_xticks(range(6),LABELS,rotation=22,ha='right')
    fig.suptitle('Selection trade-offs for stars at reference distances of 5–10 kpc')
    return fig,'F03_selection_tradeoffs',final,[source]


def blue_cmd():
    source = ROOT/'manifests/v3_6/figures/F03_blue_distance_failures.csv'
    rows = table(source)
    fig,axes = plt.subplots(1,2,figsize=(12.2,5.8),layout='constrained',sharey=True,sharex=True)
    gr = vals(rows,'dereddened_g_r')
    for color,sample,label in ((ORANGE,'faint_blue_near','Faint nearby (116)'),
                               (BLUE,'blue_2_to_5','Reference 2–5 kpc (189)')):
        keep = np.asarray([r['sample']==sample for r in rows])
        for ax,field,title in zip(axes,('M_G_reference','M_G_qrf'),
                                  ('Absolute magnitude from Gaia','Absolute magnitude from QRF')):
            ax.scatter(gr[keep],vals(rows,field)[keep],s=18,alpha=.48,color=color,label=label)
            ax.set_xlabel(r'Dereddened $g-r$ (mag)')
            ax.set_title(title,fontsize=15)
    axes[0].set_ylabel(r'Absolute Gaia $G$ magnitude')
    axes[0].invert_yaxis()
    axes[1].legend()
    fig.suptitle('QRF places two blue cohorts on a common luminosity branch')
    return fig,'F04_blue_color_magnitude',rows,[source]


def radial(root):
    source = root/'selection/catalog.csv'
    rows = table(source)
    fig,axes = plt.subplots(1,2,figsize=(12.6,5.3),layout='constrained',sharey=True)
    final = []
    for ax,name,title in zip(axes,('hard_count','pdf_count'),('Median distance bins','Summed QRF PDF mass')):
        for j,color in zip((0,1,2,4),(BLUE,GRAY,ORANGE,GREEN)):
            group = [choose(rows,selection=WORKPOINTS[j],shell=b)[0] for b in SHELLS[:3]]
            final.extend(dict(**s,estimand=name) for s in group)
            ax.plot(range(3),vals(group,name),'o-',color=color,label=LABELS[j])
        ax.set_yscale('log')
        ax.set_xticks(range(3),SHELLS[:3])
        ax.set_xlabel('Assigned distance (kpc)')
        ax.set_title(title)
    axes[0].set_ylabel('Estimated catalog count')
    axes[0].legend()
    fig.suptitle('Distance-layer counts from the weighted catalog draw')
    return fig,'F05_radial_selection',final,[source]


def weak_gaia(root):
    scores = root/'weak_gaia/scores.csv'
    stress = root/'weak_gaia/stress.csv'
    rows = choose(table(scores),history='outside_recorded_history')
    alt = choose(table(stress),history='outside_recorded_history',photometry='faint',far_odds='1.0',
                 offset_mas='0.0')
    fig,axes = plt.subplots(1,2,figsize=(13.5,5.2),layout='constrained')
    final = []
    for j,(key,color) in enumerate(zip(('supported','primary','strict'),COLORS)):
        group = [choose(rows,selection=key,photometry=k)[0] for k in ('all','faint','faint_blue')
                 if choose(rows,selection=key,photometry=k)]
        final.extend(group)
        indices = [('all','faint','faint_blue').index(r['photometry']) for r in group]
        offset = np.asarray(indices)+.13*(j-1)
        y = vals(group,'qrf_minus_broad')
        errors = np.maximum(0.,np.array([y-vals(group,'lo'),vals(group,'hi')-y]))
        axes[0].errorbar(offset,y,yerr=errors,fmt='o',color=color,label=key.capitalize(),capsize=3)
        a = sorted(choose(alt,selection=key),key=lambda r:float(r['parallax_scale']))
        final.extend(a)
        x,y = vals(a,'parallax_scale'),vals(a,'delta')
        axes[1].plot(x,y,'o-',color=color)
        axes[1].fill_between(x,vals(a,'lo'),vals(a,'hi'),color=color,alpha=.14)
    axes[0].set_xticks(range(3),['All','Faint','Faint blue'])
    axes[0].set_ylabel('QRF minus broad log score (nat)')
    axes[0].legend()
    axes[0].set_title('Comparison at the native scale')
    axes[1].set_xlabel('Parallax scale factor')
    axes[1].set_ylabel('Change in QRF log score (nat)')
    axes[1].set_title('Faint stars, scale sensitivity')
    for ax in axes:
        ax.axhline(0,color=GRAY,lw=.9)
    fig.suptitle('Signed Gaia parallaxes for QRF median distances of 5–10 kpc\n'
                 'Outside recorded history, 95% sky-block intervals')
    return fig,'F06_signed_gaia',final,[scores,stress]


def leakage(root):
    migration = root/'reference_comparison/qualified_migration.csv'
    odds = root/'lf_odds/odds_leakage.csv'
    a = choose(table(migration),source='SpecDis',selection='primary',history='all',reference_class='dwarf')
    fig,axes = plt.subplots(1,2,figsize=(13.6,5.6),layout='constrained')
    grid = np.full((6,6),np.nan)
    final = []
    for i,src in enumerate(SHELLS):
        for j,dst in enumerate(SHELLS):
            r = choose(a,reference_shell=src,predicted_shell=dst)[0]
            final.append(r)
            n = int(r['selected'])
            if not n:
                continue
            value = float(r['pdf_mass'])/n
            grid[i,j] = value
            axes[0].text(j,i,f"{int(r['hard_count']):,}",color='white' if value>.50 else '#252525',
                         fontsize=9,va='center',ha='center')
    image = axes[0].imshow(grid,cmap='Blues',norm=Normalize(0.,1.),aspect='equal')
    axes[0].set_xticks(range(6),SHELLS,rotation=20)
    axes[0].set_yticks(range(6),SHELLS)
    axes[0].set_xlabel('QRF distance (kpc)')
    axes[0].set_ylabel('SpecDis distance (kpc)')
    axes[0].set_title('Primary, known dwarfs\nNumbers show median-bin counts',fontsize=15)
    fig.colorbar(image,ax=axes[0],fraction=.05,label='PDF fraction of reference row')
    rows = choose(table(odds),shell='5–10')
    ax = axes[1]
    for i,(event,label) in enumerate((('far_giant','Far giant'),('foreground','Foreground'))):
        for j,(selection,assignment,color) in enumerate(
                [('primary','median_bin',BLUE),('primary','pdf_weighted',ORANGE),
                 ('strict','median_bin',GREEN),('strict','pdf_weighted',GRAY)]):
            group = choose(rows,event=event,selection=selection,assignment=assignment)
            final.extend(group)
            v = 100*vals(group,'fraction')
            pos = i+(j-1.5)*.15
            ax.plot([np.min(v),np.max(v)],[pos,pos],color=color,lw=3,
                    label=f"{selection.capitalize()}, {'median' if j%2==0 else 'PDF'}" if i==0 else None)
            ax.scatter(v,np.full(len(v),pos),color=color,s=8)
    ax.set_xscale('log')
    ax.set_yticks([0,1],['Far giant','Foreground'])
    ax.legend(fontsize=9,loc='best')
    ax.set_xlabel('Conditional fraction (%)')
    ax.set_title('LF and class-odds sensitivity\nAssigned layer 5–10 kpc',fontsize=15)
    fig.suptitle('Direct reference migration and conditional leakage')
    return fig,'F07_migration_leakage',final,[migration,odds]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--figures',type=Path,default=ROOT/'figures/v3_7')
    p.add_argument('--manifests',type=Path,default=ROOT/'manifests/v3_7/figures')
    p.add_argument('--plate',choices=('all','performance','selection','blue_cmd','radial','weak_gaia','leakage'),default='all')
    a = p.parse_args()
    for d in (a.figures,a.manifests):
        d.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update(STYLE)
    for plot in (performance,selection,blue_cmd,radial,weak_gaia,leakage):
        if a.plate in ('all',plot.__name__):
            result = plot() if plot in (performance,blue_cmd) else plot(a.inputs)
            save(*result,a.figures,a.manifests)


if __name__=='__main__':
    main()
