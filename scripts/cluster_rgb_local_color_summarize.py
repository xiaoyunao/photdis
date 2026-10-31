#!/usr/bin/env python3
"""Small reproducible tables and scientific figures for the local experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photdis.cluster_rgb_local_color import conditional_mass,summarize_local
from photdis.stage1 import require_compute,write_json

DATA=Path('data/cluster_rgb_local_color_v1')
FIG=Path('figures/cluster_rgb_local_color_v1')


def main():
    require_compute();FIG.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('configs/cluster_rgb_local_color_v1.json').read_text())
    results={};curves={};brightness={};hashes={}
    for name in cfg['clusters']:
        selection_path=DATA/f'{name}_selection.json'
        selection=json.loads(selection_path.read_text());results[name]={'selection':selection,'fits':{}}
        hashes[str(selection_path)]=hashlib.sha256(selection_path.read_bytes()).hexdigest()
        if selection['n']<cfg['minimum_fit_n']:continue
        brightness[name]=dict(np.load(DATA/f'{name}_selection.npz'))
        curves[name]={}
        for se in cfg['physical_ebv_sigmas']:
            for model in cfg['models']:
                for prior in cfg['priors']:
                    key=f'{model}__{prior}__e{se:.3f}'
                    initial=DATA/f'{name}__{key}__n{cfg["draws"]}.json'
                    refined=DATA/f'{name}__{key}__n{cfg["refinement_draws"]}.json'
                    path=refined if refined.exists() else initial
                    row=json.loads(path.read_text())
                    if row['ids_sha256']!=selection['ids_sha256']:raise ValueError('Interior identity mismatch')
                    if path==refined:
                        first=json.loads(initial.read_text())
                        row['initial_numerical_results']={'diagnostics':first['diagnostics'],'stats':first['stats']}
                        hashes[str(initial)]=hashlib.sha256(initial.read_bytes()).hexdigest()
                    results[name]['fits'][key]=row
                    curves[name][key]=dict(np.load(path.with_suffix('.npz')))
                    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        edges=next(iter(curves[name].values()))['edges']
        no_color={}
        for se in cfg['physical_ebv_sigmas']:
            b=brightness[name][f'log_B_{se:.3f}']
            p=conditional_mass(np.exp(b-b.max()),b)
            no_color[f'{se:.3f}']=dict(summarize_local(edges,p),
                maximum_mass_error=float(np.max(abs(p-1/len(p)))))
        results[name]['no_color_identity']=no_color
        differences={}
        for se in cfg['physical_ebv_sigmas']:
            for prior in cfg['priors']:
                gr=results[name]['fits'][f'robust_gr__{prior}__e{se:.3f}']['stats']
                grz=results[name]['fits'][f'robust_grz__{prior}__e{se:.3f}']['stats']
                for weight in gr:
                    a,b=gr[weight],grz[weight]
                    differences[f'{weight}__e{se:.3f}']=dict(
                        gr_minus_grz_width68=a['width68']-b['width68'],
                        grz_minus_gr_kl=b['kl_nats']-a['kl_nats'],
                        grz_minus_gr_median=b['q'][2]-a['q'][2],
                        either_edge_flag=a['edge_flag'] or b['edge_flag'],
                        both_numerical_pass=a['numerical_pass'] and b['numerical_pass'])
        results[name]['same_star_band_comparison']=differences
    write_json(Path('manifests/cluster_rgb_local_color_v1.json'),dict(
        config=cfg,clusters=results,input_hashes=hashes,
        interpretation='local prior updates on exposed interior RGB members; widths/KL are not accuracy',
        aggregation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def panels(title):
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        fig.suptitle(title,fontsize=13)
        for ax,name in zip(axes.flat,cfg['clusters']):
            ax.set_title(f'{name.upper()} — N={results[name]["selection"]["n"]}')
            ax.set_xlabel(r'$\delta\mu=\mu-\mu_{ref}$ (mag)');ax.grid(alpha=.15)
            ax.set_xlim(-.5,.5)
            if name not in curves:ax.text(.1,.5,'Too few interior members',transform=ax.transAxes)
        return fig,axes

    def finish(fig,axes,filename):
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
        fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(FIG/filename,dpi=160);plt.close(fig)

    fig,axes=panels('Brightness boundary check, before conditional cancellation')
    for ax,name in zip(axes.flat,cfg['clusters']):
        if name not in curves:continue
        for se,color in zip(cfg['physical_ebv_sigmas'],['#168b77','#bd4963']):
            b=brightness[name][f'log_B_{se:.3f}']
            ax.plot(brightness[name]['delta_mu'],np.exp(b-b.max()),c=color,label=fr'$\sigma_E={se}$')
        ax.axhline(1,c='.5',ls=':',label='Flat brightness likelihood')
        ymin=min(np.exp(brightness[name][f'log_B_{s:.3f}']-brightness[name][f'log_B_{s:.3f}'].max()).min()
                 for s in cfg['physical_ebv_sigmas'])
        ax.set_ylim(min(.94,ymin-.03),1.01);ax.set_ylabel(r'$B(\mu)/\max B$')
    finish(fig,axes,'01_brightness_boundary.png')

    fig,axes=panels('Conditional color information — reference abundance, sigma(E)=0.005')
    for ax,name in zip(axes.flat,cfg['clusters']):
        if name not in curves:continue
        for model,color in [('robust_gr','#168b77'),('robust_grz','#485db5')]:
            a=curves[name][f'{model}__reference__e0.005'];edges=a['edges'];x=(edges[:-1]+edges[1:])/2
            ax.plot(x,a['reference']/np.diff(edges),color=color,label=model.replace('robust_',''))
        ax.axhline(1,color='.5',ls='--',label='No color = local uniform prior')
        ax.axvline(0,c='.7',ls=':');ax.set_ylabel('Conditional density / mag')
    finish(fig,axes,'02_reference_color_information.png')

    fig,axes=panels('grz conditional posterior — common abundance prior comparison')
    for ax,name in zip(axes.flat,cfg['clusters']):
        if name not in curves:continue
        a=curves[name]['robust_grz__unknown__e0.005'];edges=a['edges'];x=(edges[:-1]+edges[1:])/2
        for key,color in [('equal','#168b77'),('rich','#d78916'),('poor','#7952a3')]:
            ax.plot(x,a[key]/np.diff(edges),color=color,label='Unknown FeH: '+key)
        known=curves[name]['robust_grz__reference__e0.005']['reference']
        ax.plot(x,known/np.diff(edges),color='k',ls='--',label='Reference FeH')
        ax.axhline(1,c='.6',ls=':',label='Local uniform prior');ax.set_ylabel('Conditional density / mag')
    finish(fig,axes,'03_abundance_prior_sensitivity.png')

    fig,axes=panels('grz conditional posterior — shared reddening sensitivity')
    for ax,name in zip(axes.flat,cfg['clusters']):
        if name not in curves:continue
        for prior,weight,color in [('reference','reference','#485db5'),('unknown','equal','#d78916')]:
            for se,style in zip(cfg['physical_ebv_sigmas'],['-','--']):
                a=curves[name][f'robust_grz__{prior}__e{se:.3f}'];edges=a['edges'];x=(edges[:-1]+edges[1:])/2
                ax.plot(x,a[weight]/np.diff(edges),c=color,ls=style,label=f'{prior} FeH, sigma(E)={se}')
        ax.axhline(1,c='.6',ls=':',label='Local uniform prior');ax.set_ylabel('Conditional density / mag')
    finish(fig,axes,'04_reddening_sensitivity.png')
    print('Wrote local-color manifest and four figures',flush=True)


if __name__=='__main__':main()
