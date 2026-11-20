#!/usr/bin/env python3
"""Render the integrated science figures with the locked Pal 5/v3.8 style."""
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from science_closeout_catalogue import OUT, ROOT, digest, read, write

DEST = OUT / 'figures'
ARCHIVE = ROOT / 'data/science_closeout_20261120/sequence'
SYSTEMS = ('NGC_6362','NGC_6723','NGC_6809_M55','M5','M30','M13','M92')
NAMES = dict(zip(SYSTEMS,('NGC 6362','NGC 6723','M55','M5','M30','M13','M92')))
FIELDS = ('L219','L233','L240','L224','L032','M037','M240','M249','H240')


def finish(fig,name,rows=None):
    for ext in ('png','pdf'):
        fig.savefig(DEST / f'{name}.{ext}', dpi=300)
    plt.close(fig)
    if rows:
        write(DEST / f'{name}.csv',rows)


def compact_training():
    from photdis.stage1 import require_compute
    require_compute()
    colors = np.linspace(-.25,1.55,73)
    mag = np.linspace(1,10,73)
    rows=[]
    for side in ('south','north'):
        with np.load(ARCHIVE/f'{side}_training.npz') as t:
            m = t['m']
            hist = np.histogram2d(m[:,0]-m[:,1],t['y']-m[:,0]+m[:,1],bins=(colors,mag))[0]
        a,b=np.where(hist>0)
        rows.extend(dict(native=side,gr_lo=colors[i],gr_hi=colors[i+1],Mr_lo=mag[j],Mr_hi=mag[j+1],
                         labels=int(hist[i,j])) for i,j in zip(a,b))
    write(ARCHIVE/'training_hess.csv',rows)


def sequence():
    h=read(OUT/'training_hess.csv')
    s=read(OUT/'sequence.csv')
    joint=[x for x in s if x['axis']=='Mr' and int(x['n'])>=20]
    palette=[BLUE,ORANGE,GREEN, '#9467bd','#8c564b','#d62728','#4c78a8']
    fig,axes=plt.subplots(1,3,figsize=(16,5.1),layout='constrained')
    for k,side in enumerate(('south','north')):
        ux=np.linspace(-.25,1.55,73)
        uy=np.linspace(1,10,73)
        hist=np.zeros((72,72))
        for row in h:
            if row['native'] != side:
                continue
            i=np.abs(ux-float(row['gr_lo'])).argmin()
            j=np.abs(uy-float(row['Mr_lo'])).argmin()
            hist[i,j]=int(row['labels'])
        ax=axes[k]
        art=ax.pcolormesh(ux,uy,np.ma.masked_equal(hist.T,0),cmap='viridis',
                          norm=LogNorm(1,max(2,hist.max())),rasterized=True)
        for system,c in zip(SYSTEMS,palette):
            a=sorted([r for r in joint if (r['system'],r['side'])==(system,side)],
                     key=lambda r:float(r['Mr']))
            if a:
                ax.plot([float(r['gr']) for r in a],[float(r['Mr']) for r in a],'o-',color=c,lw=1.5,
                        ms=4,label=NAMES[system])
        ax.set(xlabel=r'$(g-r)_0$',ylabel=r'$M_r$',ylim=(9.5,1.5),
               xlim=(-.15,1.35),title={'south':'DECam', 'north':'BASS/MzLS'}[side])
        ax.legend(loc='lower left',fontsize=9)
        fig.colorbar(art,ax=ax,pad=.02,label='Training labels per cell')
    ax=axes[2]
    for system,c in zip(SYSTEMS,palette):
        r=sorted([r for r in joint if r['system']==system],key=lambda z:float(z['Mr']))
        if r:
            xs=[float(a['Mr']) for a in r]
            ax.plot(xs,[float(a['qrf_bias']) for a in r],'o-',ms=4,color=c,label=NAMES[system])
            ax.plot(xs,[float(a['leaf_delta_mu']) for a in r],ls='--',color=c,alpha=.78)
    ax.axhline(0,color=GRAY,lw=1)
    ax.set(xlabel=r'Reference $M_r$',ylabel=r'$\Delta\mu$ (mag)',
           title='Member distances and routed labels')
    ax.plot([],[],color='black',marker='o',label='Member QRF median')
    ax.plot([],[],color='black',ls='--',label='Leaf mean at measured color')
    ax.legend(fontsize=8,ncol=2)
    finish(fig,'training_and_cluster_scale',joint)


def signed():
    gates=[x for p in sorted((OUT/'signed').glob('*/gate.csv')) for x in read(p)]
    score=[x for p in sorted((OUT/'signed').glob('*/signed_summary.csv')) for x in read(p)]
    hist=[x for p in sorted((OUT/'signed').glob('*/signed_snr.csv')) for x in read(p)]
    raw=[x for p in sorted((OUT/'signed').glob('*/power.csv')) for x in read(p)]
    lookup={(x['field'],int(x['r_lo']),abs(float(x['shift_mag']))):x for x in raw
            if float(x['systematics_multiplier'])==1 and float(x['shift_mag'])>0}
    fig,axes=plt.subplots(1,3,figsize=(17,5.1),layout='constrained')
    cells=np.zeros((len(FIELDS),4))
    for j,field in enumerate(FIELDS):
        for i,(r,s) in enumerate(((20,.3),(20,.5),(21,.3),(21,.5))):
            pos=lookup[field,r,s]
            neg=next(q for q in raw if (q['field'],int(q['r_lo']),float(q['shift_mag']),
                       float(q['systematics_multiplier']))==(field,r,-s,1))
            cells[j,i]=min(float(pos['directional_power']),float(neg['directional_power']))
    im=axes[0].imshow(cells,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    axes[0].set_xticks(range(4),('20–21\n0.3','20–21\n0.5','21–22\n0.3','21–22\n0.5'))
    axes[0].set_yticks(range(len(FIELDS)),FIELDS)
    axes[0].set(xlabel=r'Injected $|\Delta\mu|$ (mag)',title='Signed-parallax power')
    for j,i in np.ndindex(cells.shape):
        axes[0].text(i,j,f'{cells[j,i]:.2f}',ha='center',va='center',
                     color='white' if cells[j,i]>.57 else 'black',fontsize=9)
    axes[0].grid(False)
    fig.colorbar(im,ax=axes[0],label='Directional power')
    for r,color,label in [(20,BLUE,'20–21'),(21,ORANGE,'21–22')]:
        q=[x for x in score if int(x['r_lo'])==r]
        axes[1].errorbar([float(x['scale_shift_linear_mag']) for x in q],
                         [FIELDS.index(x['field'])+(-.13 if r==20 else .13) for x in q],
                         xerr=[float(x['systematic_stress_sd_mag']) for x in q],
                         fmt='o',ms=5,capsize=2,color=color,label=label)
    axes[1].set_yticks(range(len(FIELDS)),FIELDS)
    axes[1].invert_yaxis()
    axes[1].axvline(0,color=GRAY,lw=1)
    axes[1].set(xlabel='Local scale response (mag)',title='Observed signed parallaxes')
    axes[1].legend(fontsize=10,title='Observed r')
    for field,col in [('L240',BLUE),('M240',ORANGE)]:
        q=[x for x in hist if x['field']==field and x['r_lo']=='20']
        if q:
            coord=np.arange(len(q))-.5
            for key,style in [('predicted','--'),('observed','-')]:
                a=np.array([float(x[key]) for x in q])
                axes[2].stairs(a/a.sum(),np.r_[coord,coord[-1]+1],color=col,ls=style,label=field+' '+key)
            axes[2].set_xticks([0,2,4,6,8,9],['<−4','−3 to −2','−1 to 0','1 to 2','3 to 4','4 to 5'],rotation=35)
    axes[2].set(xlabel='Signed parallax / uncertainty',ylabel='Fraction per bin',
                title='Observed and predicted distributions')
    axes[2].legend(fontsize=9)
    finish(fig,'signed_parallax_by_field',gates)


def environment():
    scores=[x for x in read(OUT/'environment_metrics.csv') if x['axis']=='all']
    flows=read(OUT/'environment_flow.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    y=np.arange(len(FIELDS))
    for arm, off,color,label in [('A_full',-.18,GRAY,'SFD'),('B_map_range',0,BLUE,'SFD, map range'),
                                ('C_iteration',.18,ORANGE,'Bayestar iteration')]:
        f=[r for r in scores if r['arm']==arm]
        x=np.array([float(r['bias']) for r in f])
        lo=np.array([float(r['bias_lo']) for r in f])
        hi=np.array([float(r['bias_hi']) for r in f])
        axes[0].errorbar(x,[FIELDS.index(r['field'])+off for r in f],xerr=np.maximum(0,[x-lo,hi-x]),
                         fmt='o',ms=4,capsize=2,color=color,label=label)
    f=[r for r in scores if r['arm']=='C_iteration']
    x=np.array([float(r['crps_minus_B']) for r in f])
    lo=np.array([float(r['crps_delta_lo']) for r in f])
    hi=np.array([float(r['crps_delta_hi']) for r in f])
    axes[1].errorbar(x,[FIELDS.index(r['field']) for r in f],xerr=np.maximum(0,[x-lo,hi-x]),
                     fmt='o',ms=5,capsize=2,color=ORANGE)
    for k,c,l in [('map_valid',GRAY,'Map valid'),('converged',BLUE,'Converged'),('common',ORANGE,'Common comparison')]:
        ordered=sorted(flows,key=lambda r:FIELDS.index(r['field']))
        axes[2].plot([int(r[k])/int(r['registered']) for r in ordered],y,'o',ms=5,color=c,label=l)
    for ax in axes:
        ax.set_yticks(y,FIELDS)
        ax.invert_yaxis()
    for ax in axes[:2]:
        ax.axvline(0,color=GRAY,lw=1,ls='--')
    axes[0].set(title='Distance scale',xlabel=r'Median $\Delta\mu$ (mag)')
    axes[1].set(title='Predictive change',xlabel=r'$\mathrm{CRPS}_{\rm iteration}-\mathrm{CRPS}_{\rm SFD}$ (mag)')
    axes[2].set(title='Reference attrition',xlabel='Fraction of registered sample',xlim=(-.04,1.04))
    axes[0].legend(fontsize=9)
    axes[2].legend(fontsize=9)
    finish(fig,'environment_comparison',scores)


def migration():
    x=read(OUT/'reference_migration.csv')
    x=[r for r in x if (r['observable'],r['source'],r['region'])==('all','SpecDis','north')]
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    labels={'dwarf_feh_ge_minus1p5':'Ordinary dwarfs','dwarf_feh_lt_minus1p5':'Metal-poor dwarfs',
            'intermediate_gravity':'Intermediate gravity','giant':'Giants'}
    colors=(BLUE,ORANGE,GREEN,GRAY)
    show=[]
    for ax,(a,b) in zip(axes,[(2,5),(5,10)]):
        for i,(group,label) in enumerate(labels.items()):
            before,after=[next(r for r in x if r['population']==group and float(r['ref_lo'])==a
                 and float(r['ref_hi'])==b and r['stage']==key) for key in ('supported','selected')]
            if not int(before['n']):
                continue
            key='2_5' if a==2 else '5_10'
            for pos,suffix,arm,c in ((-.22,'median',before,colors[i]),(-.075,'pdf',before,colors[i]),
                                     (.075,'median',after,colors[i]),(.22,'pdf',after,colors[i])):
                value=float(arm[suffix+'_'+key])/int(before['n'])
                ax.bar(i+pos,value,width=.14,color=c,alpha=.8 if arm is before else .35,
                       hatch='//' if suffix=='pdf' else None,lw=.6,edgecolor=c)
                show.append(dict(population=group,ref_lo=a,ref_hi=b,estimator=suffix,stage=arm['stage'],
                                 support=before['n'],fraction_of_supported=value))
        ax.set_xticks(range(4),labels.values(),rotation=20,ha='right')
        ax.set(title=f'SpecDis reference {a}–{b} kpc',ylim=(0,1),ylabel='Same-shell fraction of supported stars')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=GRAY,alpha=.8,label='Before selection'),
        Patch(color=GRAY,alpha=.35,label='After selection'),Patch(facecolor='white',edgecolor=GRAY,hatch='//',label='PDF mass')],fontsize=9)
    g=next(r for r in read(OUT/'gd1_counts.csv') if r['group']=='all' and r['phi1_lo']=='-180')
    for k, off,c,l in [('ref',-.23,GRAY,'Reference track'),('hard',0,BLUE,'Median'),('pdf',.23,ORANGE,'PDF mass')]:
        axes[2].bar(np.arange(4)+off,[float(g[f'{k}_{z}'])/int(g['n']) for z in ('0_2','2_5','5_10','10_plus')],
                    width=.22,color=c,alpha=.75,label=l)
    axes[2].set_xticks(range(4),('0–2','2–5','5–10','>10'))
    axes[2].set(title='Known GD-1 members',xlabel='Heliocentric distance (kpc)',ylabel='Fraction of 259 members',ylim=(0,1.04))
    axes[2].legend(fontsize=9)
    finish(fig,'migration_and_gd1',show)


def main():
    lock = json.loads((ROOT/'configs/v3_10_figure_style.json').read_text())
    assert all(digest(ROOT/p)==sha for p,sha in lock['files'].items())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact-training',action='store_true')
    if parser.parse_args().compact_training:
        compact_training()
        return
    DEST.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        sequence()
        signed()
        environment()
        migration()


if __name__=='__main__':
    main()
