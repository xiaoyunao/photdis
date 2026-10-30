#!/usr/bin/env python3
"""Assemble small auditable results and scientific figures from scale runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photdis.stage1 import require_compute, write_json

DATA=Path('data/cluster_rgb_scales_v1')
FIG=Path('figures/cluster_rgb_scales_v1')


def main():
    require_compute();FIG.mkdir(parents=True,exist_ok=True)
    response=json.loads((DATA/'responses.json').read_text())
    names=list(response['clusters'])
    cfg=json.loads(Path('configs/cluster_rgb_scales_v1.json').read_text())
    labels=cfg['scenarios']
    colors=['#333333','#168b77','#d78916','#7952a3','#bd4963']
    display=['Legacy / paper distance','Paper E, CCM','SF11 proxy E, coupled RR',
             'SF11 proxy + 0.1 dex','Paper E, spectroscopic NGC4147']
    joint={n:json.loads((DATA/f'{n}_joint.json').read_text()) for n in names}
    for n in names:
        if joint[n]['ids_sha256'] != response['clusters'][n]['ids_sha256']:
            raise ValueError('Response/joint identity mismatch')
    manifest=dict(responses=response,joint=joint,
        decision='retain frozen model; external scales ambiguous; joint inference conditional on selected Mg support',
        numerical_rule='review 256/512 quantile differences and nuisance ESS before quoting interval precision',
        product_status='diagnostic prototype, not field-star distances',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in DATA.glob('*.json')})
    write_json(Path('manifests/cluster_rgb_scales_v1.json'),manifest)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for j,(label,color) in enumerate(zip(labels,colors)):
        for k,n in enumerate(names):
            row=response['clusters'][n]['scenarios'].get(label)
            if row is None:continue
            x=k+(j-2)*.12
            axes[0].scatter(x,row['models']['robust_grz']['overall']['median'],color=color,
                            label=display[j] if (k==0 or j==4) else None)
            axes[1].scatter(x,row['geometry']['perpendicular_median'],color=color)
    for ax in axes:
        ax.axhline(0,color='gray',ls=':',lw=1)
        ax.set_xticks(range(4),[n.upper() for n in names]);ax.grid(alpha=.15)
    axes[0].set_ylabel(r'Median $\widehat{\mu}-\mu_{ref}$ (mag)')
    axes[1].set_ylabel('Median whitened perpendicular residual')
    axes[0].set_title('Frozen robust grz: fixed stars and nominal bins')
    axes[1].set_title('Two-color consistency; central frozen curve')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.14,1,1));fig.savefig(FIG/'01_scale_response.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,n in zip(axes.flat,names):
        data=np.load(DATA/f'{n}_response.npz')
        base=data['legacy__color_residual']
        ax.scatter(*base.T,c='.65',s=6,alpha=.4,label='Individual Legacy residuals')
        for label,color,desc in zip(labels,colors,display):
            row=response['clusters'][n]['scenarios'].get(label)
            if row is not None:
                ax.scatter(*row['geometry']['color_median'],c=color,s=55,marker='D',label=desc)
        ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
        ax.set_xlim(-.14,.14);ax.set_ylim(-.12,.12)
        ax.set_title(f'{n.upper()} (N={len(base)})')
        ax.set_xlabel(r'$(g-r)_0-f_{gr}(M_{ref},Z)$ (mag)')
        ax.set_ylabel(r'$(r-z)_0-f_{rz}(M_{ref},Z)$ (mag)')
        # All points stay in arrays/metrics; annotate count outside plot limits.
        outside=np.sum((abs(base[:,0])>.14)|(abs(base[:,1])>.12))
        ax.text(.02,.03,f'{outside} individual points outside displayed axes',transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'02_color_residuals.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,n in zip(axes.flat,names):
        curves=np.load(DATA/f'{n}_joint.npz');mu=curves['mu'];count=joint[n]['n']
        step=mu[1]-mu[0]
        for key,color,linestyle,label in [('prior_only','#888888','-','Shared, no color'),
            ('robust_gr','#168b77','-','Shared gr'),('robust_grz','#485db5','-','Shared grz'),
            ('robust_grz','#bd4963','--','Wrong: independent nuisance')]:
            suffix='_wrong' if linestyle=='--' else ''
            p=curves[f'{key}_{count}{suffix}']
            ax.plot(mu,p/step,color=color,ls=linestyle,label=label)
        rr=joint[n]['rr_mu_comparison_only'];ax.axvline(rr,color='k',ls=':',label='RR comparison only')
        ax.set_xlim(rr-.6,rr+.6);ax.set_title(f'{n.upper()} — {count} selected members')
        ax.set_xlabel(r'Common distance modulus $\mu$');ax.set_ylabel('Conditional density / mag')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Prototype: known lower-RGB range supplies substantial distance information',fontsize=12)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(FIG/'03_shared_distance.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,n in zip(axes.flat,names):
        for key,col in [('prior_only','#888888'),('robust_grz','#485db5')]:
            res=joint[n]['results'][key];counts=sorted(map(int,res))
            ax.plot(counts,[res[str(k)]['width68'] for k in counts],'o-',color=col,label=key+' shared')
        res=joint[n]['results']['robust_grz']
        ax.plot(counts,[res[str(k)]['wrong_independent']['width68'] for k in counts],'s--',
                color='#bd4963',label='grz wrong independent')
        ax.set_title(n.upper());ax.set_xlabel('Nested member count (source-ID order)')
        ax.set_ylabel('68% common-mu width (mag)');ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(FIG/'04_member_count.png',dpi=160);plt.close(fig)
    print('assembled',list(FIG.glob('*.png')),flush=True)


if __name__=='__main__':main()
