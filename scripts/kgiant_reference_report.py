#!/usr/bin/env python3
"""Small public batch evidence plots; supports pre-score coverage-only mode."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'figures/kgiant_reference_batch_v1';FIG.mkdir(exist_ok=True)

def read(name):return json.loads((ROOT/'manifests'/name).read_text())

def main():
    inv=read('kgiant_reference_inventory_v1.json');target=read('kgiant_selected190_evidence_v1.json')['rows']
    eligible=[r for r in inv['rows'] if r['eligible']];systems=sorted({r['member_system'] for r in eligible})
    palette={s:plt.get_cmap('tab10')(i%10) for i,s in enumerate(systems)}
    fig,ax=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    valid=[t for t in target if t['specdis_valid']]
    tc=np.array([[t['gr0'],t['rz0'],t['zw10'],t['r'],t['g0']-10-5*np.log10(t['specdis_dist_kpc'])] for t in valid])
    ax[0,0].scatter(tc[:,0],tc[:,1],s=12,c='.75',label='Selected SPECDIS-valid targets (189)')
    ax[0,1].scatter(tc[:,0],tc[:,2],s=12,c='.75')
    ax[1,0].scatter(tc[:,0],tc[:,4],s=12,c='.75');ax[1,1].scatter(tc[:,3],tc[:,4],s=12,c='.75')
    for s in systems:
        for r in [r for r in eligible if r['member_system']==s]:
            p=r['selected_measurement']['photometry'];gr=p['g0']-p['r0'];rz=p['r0']-p['z0'];zw=p['z0']-p['w10'] if p['w10'] is not None else np.nan
            for a,x,y in [(ax[0,0],gr,rz),(ax[0,1],gr,zw),(ax[1,0],gr,r['Mg_ref']),(ax[1,1],p['r'],r['Mg_ref'])]:
                a.scatter(x,y,s=90 if r['historical_diagnostic'] else 45,c=[palette[s]],marker='*' if r['historical_diagnostic'] else 'o',edgecolors='black',linewidths=.4)
        ax[0,0].scatter([],[],c=[palette[s]],label=f'{s}: {sum(r["member_system"]==s for r in eligible)}')
    for a,x,y in [(ax[0,0],'(g-r)0','(r-z)0'),(ax[0,1],'(g-r)0','(z-W1)0'),(ax[1,0],'(g-r)0','Mg (reference / teacher)'),(ax[1,1],'Observed r','Mg (reference / teacher)')]:a.set(xlabel=x,ylabel=y);a.grid(alpha=.2)
    ax[1,0].invert_yaxis();ax[1,1].invert_yaxis();ax[0,0].legend(fontsize=7)
    fig.suptitle('Frozen pre-score coverage | external references vs selected teacher-scale targets\nStars: prior Terzan7 diagnostics; circles: member-first batch; phase remains unknown')
    fig.savefig(FIG/'01_frozen_coverage.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    sc=[170,159,len({r['member_system'] for r in inv['rows'] if not r['historical_diagnostic']}),len(systems)]
    ax[0].barh(['Member archive','Published distance linked','With DESI K members','Eligible incl. prior Terzan7'],sc,color=['.7','.6','#56b4e9','#0072b2']);ax[0].set_xlabel('Systems');ax[0].set_xlim(0,190);ax[0].invert_yaxis()
    for i,n in enumerate(sc):ax[0].text(n+1,i,str(n),va='center')
    for r in inv['rows']:
        if 'selected_measurement' not in r:continue
        m=r['selected_measurement'];ax[1].scatter(abs(r['galactic_b']),m['ebv'],c='#0072b2' if r['eligible'] else '.6',marker='*' if r['historical_diagnostic'] else 'o',s=65 if r['historical_diagnostic'] else 20)
    ax[1].axvline(30,c='black',ls='--');ax[1].axhline(.03,c='black',ls='--');ax[1].set(xlabel='|Galactic latitude| [deg]',ylabel='Catalog EBV',title='Blue: eligible; gray: other measured K members')
    fig.savefig(FIG/'02_system_and_scope.png',dpi=160);plt.close(fig)
    if '--coverage-only' in sys.argv:return
    pred=read('kgiant_reference_predictions_v1.json');rows=pred['rows']
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,t in zip(axes,['spdist','specdis']):
        for r in [r for r in rows if r['teacher']==t and r['model']=='A_grz']:
            q=np.array(r['distance_kpc_quantiles']);ref=10**((r['reference_mu']-10)/5)
            lo=ref-10**((r['reference_mu']-r['reference_mu_minus']-10)/5);hi=10**((r['reference_mu']+r['reference_mu_plus']-10)/5)-ref
            ax.errorbar(ref,q[2],xerr=np.array([[lo],[hi]]),yerr=np.array([[q[2]-q[1]],[q[3]-q[2]]]),fmt='*' if r['historical_diagnostic'] else 'o',c=palette[r['system']],alpha=.7,ms=6)
        ax.plot([1,150],[1,150],'k--',lw=1);ax.set(xscale='log',yscale='log',xlabel='BV21 reference distance [kpc]',ylabel='grz empirical phot-d [kpc]',title=t+' | central68% widths')
        ax.grid(alpha=.2)
    for s in systems:axes[0].plot([],[],'o',c=palette[s],label=s)
    axes[0].legend(fontsize=7);fig.suptitle('Fixed models: individual members share system reference errors')
    fig.savefig(FIG/'04_distance_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,t in zip(axes,['spdist','specdis']):
        for i,s in enumerate(systems):
            for model,offset,color in [('A_grz',-.12,'#0072b2'),('B_grzW1',.12,'#d55e00')]:
                common={r['source_id'] for r in rows if r['system']==s and r['teacher']==t and r['model']=='B_grzW1'}
                rr=[r for r in rows if r['system']==s and r['teacher']==t and r['model']==model and r['source_id'] in common]
                if not rr:
                    if model=='A_grz':ax.text(i,.02,'No W1 pair',ha='center',fontsize=8,transform=ax.get_xaxis_transform())
                    continue
                median=np.median([r['delta_mu_median'] for r in rr]);ax.errorbar(i+offset,median,yerr=np.array([[rr[0]['reference_mu_plus']],[rr[0]['reference_mu_minus']]]),fmt='o',c=color)
                ax.text(i+offset,median+.025,str(len(rr)),fontsize=7,ha='center')
        ax.axhline(0,c='black',ls='--');ax.set(xticks=range(len(systems)),xticklabels=systems,ylabel='Median member delta mu [mag]',title=t+' | identical A/B members');ax.tick_params(axis='x',rotation=35);ax.set_xlim(-.5,len(systems)-.5)
    axes[0].plot([],[],'o',c='#0072b2',label='A grz');axes[0].plot([],[],'o',c='#d55e00',label='B grzW1');axes[0].legend()
    fig.suptitle('Descriptive system centers; bars show shared reference error ONLY; numbers are stars')
    fig.savefig(FIG/'05_system_centers.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
