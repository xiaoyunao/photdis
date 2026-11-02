#!/usr/bin/env python3
"""Reproduce small public evidence and figures without loading source catalogs."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import ndtr,ndtri
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=Path('figures/kgiant_external_v1');out.mkdir(exist_ok=True)
    source=Path('manifests/kgiant_selected190_evidence_v1.json')
    rows=json.loads(source.read_text())['rows']
    common=[r for r in rows if r['spdist_valid'] and r['specdis_valid']]
    other=[r for r in rows if not r['spdist_valid'] and r['specdis_valid']]
    models=json.load(open('manifests/kgiant_distance_comparison_v1.json'))['models']
    rec=dict(common_n=len(common),remaining_specdis_n=len(other),
        common_teff_min_max=[min(r['desi_teff'] for r in common),max(r['desi_teff'] for r in common)],
        common_teff_ge4950=sum(r['desi_teff']>=4950 for r in common),
        common_specdis_fractional_error_median=float(np.median([r['specdis_fractional_error'] for r in common])),
        remaining_specdis_fractional_error_median=float(np.median([r['specdis_fractional_error'] for r in other])),
        outside_training_box={t:{m:models[t][m]['selected']['outside_training_box_n'] for m in ('A_grz','B_grzW1')} for t in ('spdist','specdis')},
        ideal_noisy_centers=dict(variance_ratio=2.,nominal_central_mass=.68,
            noisy_point_coverage=float(2*ndtr(np.sqrt(2)*ndtri(.84))-1),scope='ideal equal-error Gaussian example, not actual error decomposition'),
        evidence_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert rec['common_n']==39 and rec['remaining_specdis_n']==150 and rec['common_teff_ge4950']==34
    Path('manifests/kgiant_review_addendum_v1.json').write_text(json.dumps(rec,indent=2)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ax[0].hist([r['desi_teff'] for r in rows],bins=np.arange(4200,5025,25),color='.8',label='All selected K (190)')
    ax[0].hist([r['desi_teff'] for r in common],bins=np.arange(4200,5025,25),color='#0072b2',label='Common teachers (39)')
    ax[0].set(xlabel='DESI Teff [K]',ylabel='Count',title='34 / 39 at Teff >= 4950 K');ax[0].legend(fontsize=8)
    for rr,label,color in [(common,'Common39','#0072b2'),(other,'Other150','#d55e00')]:
        x=np.sort([r['specdis_fractional_error'] for r in rr]);ax[1].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label,color=color)
    ax[1].set(xlabel='SPECDIS reported sigma_d / d',ylabel='Cumulative fraction',title='Different teacher-error populations');ax[1].legend()
    x=np.linspace(-4,4,800);normal=np.exp(-x*x/2)/np.sqrt(2*np.pi);wide=np.exp(-x*x/4)/np.sqrt(4*np.pi)
    ax[2].plot(x,normal,label='Noisy teacher points');ax[2].plot(x,wide,label='Error-broadened centers')
    edge=np.sqrt(2)*ndtri(.84);ax[2].axvspan(-edge,edge,alpha=.15,color='orange')
    ax[2].set(xlabel='Offset / teacher sigma',ylabel='Density',title='Ideal raw68% interval: 84.04% coverage');ax[2].legend(fontsize=8)
    fig.savefig(out/'01_teacher_scope_and_width.png',dpi=160);plt.close(fig)
    inv=json.load(open('manifests/kgiant_external_inventory_v1.json'))['rows']
    pred=json.load(open('manifests/kgiant_external_predictions_v1.json'))
    assert len(inv)==6 and len(pred['audit'])==6
    flags=['Unique identity','grz quality','W1 quality','Old dust scope','Known K type','Distance eligible']
    data=[]
    for r,a in zip(inv,pred['audit']):
        assert r['name']==a['name']
        data.append([1 if r['cluster']=='Pal12' else int(r['identity_status']=='unique_position_match'),int(r.get('grz_quality',False)),int(r.get('w1_quality',False)),int(r.get('old_dust_scope',False)),
            -1 if r['cluster']=='Pal12' else int(r.get('nominal_K',False)),int(a['eligible'])])
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    from matplotlib.colors import ListedColormap,BoundaryNorm
    ax.imshow(data,cmap=ListedColormap(['#e5bd63','#e8a6a6','#8fc7b0']),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),aspect='auto')
    ax.set_xticks(np.arange(6),flags);ax.set_yticks(np.arange(6),[r['cluster']+' '+('bright member' if r['cluster']=='Pal12' else r['name']) for r in inv])
    for i,row in enumerate(data):
        for j,v in enumerate(row):ax.text(j,i,{-1:'unknown',0:'no',1:'yes'}[v],ha='center',va='center')
    ax.set_title('Fixed external targets | quality, evidence and scope are separate\nPal12 identity = exact cached Gaia member; literature temperature link is unknown')
    fig.savefig(out/'02_external_quality_and_scope.png',dpi=160);plt.close(fig)
    if pred['rows']:
        fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
        labels=[]
        for i,r in enumerate(pred['rows']):
            q=np.array(r['quantiles_mu'])-r['reference_mu'];ax.plot(q[[0,4]],[i,i],color='.6');ax.plot(q[[1,3]],[i,i],lw=3,color='#0072b2');ax.plot(q[2],i,'o',color='#d55e00')
            labels.append(r['name']+' '+r['teacher']+' '+r['model'])
        ax.set_yticks(np.arange(len(labels)),labels,fontsize=8);ax.axvline(0,color='k',ls='--');ax.set(xlabel='mu_phot - mu_reference [mag]',title='Individual empirical teacher distributions | no population accuracy claim')
        fig.savefig(out/'03_external_intervals.png',dpi=160);plt.close(fig)
    print('Evidence, identity/order checks and figures complete')

if __name__=='__main__':main()
