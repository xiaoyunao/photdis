#!/usr/bin/env python3
"""Compare the earlier band, known-class, and common-distance controls."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


NAME = 'F10_historical_method_controls'


def source(folder,name,inputs):
    path = ROOT/'manifests'/folder/name
    marker = path.with_name('complete.json')
    receipt = json.loads(marker.read_text())
    if sha(path)!=receipt['outputs'][name]:
        raise ValueError(f'Source differs from its receipt: {path}')
    inputs.extend((marker,path))
    with path.open() as stream:
        return list(csv.DictReader(stream)),receipt


def main():
    inputs = []
    rows,_ = source('v3_0/controls','probe_metrics.csv',inputs)
    probe = {r['view']:r for r in rows if r['version']=='calibrated' and r['group']=='all_optional_measured'}
    band_names = ('no_color','grz','griz','multiband')
    n = int(probe['no_color']['n'])
    if {int(probe[k]['n']) for k in band_names}!={n}:
        raise ValueError('Band comparisons require the same evaluation identities')
    gain = np.array([float(probe[k]['log_score']) for k in band_names])
    gain -= gain[0]
    rows,_ = source('v3_3/giant_conditional','distance.csv',inputs)
    known = {r['view']:r for r in rows if r['group']=='all_giants'}
    models = ('nominal','candidate')
    n_giant = int(known['nominal']['n'])
    if {int(known[k]['n']) for k in models}!={n_giant}:
        raise ValueError('The giant conditional comparison changed its sample')
    gm = np.asarray([[100*float(known[k][x]) for x in ('nmad_relative','severe_fraction')] for k in models])
    rows,benchmark = source('v3_3/structure','distance_score.csv',inputs)
    comparison = {r['branch']:r for r in rows if r['variant']=='fine'}
    branches = ('full','gaussian')
    distance = np.asarray([[float(comparison[k][x]) for x in ('q16_kpc','median_kpc','q84_kpc')] for k in branches])
    ref = benchmark['benchmark_distance_kpc']

    FIGURE.mkdir(parents=True,exist_ok=True)
    TABLE.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,3,figsize=(15.3,5.0),layout='constrained')
        a,b,c = axes
        x = np.arange(len(band_names))
        a.bar(x,gain,width=.66,color=[GRAY,BLUE,GREEN,ORANGE])
        a.set(title='Class information',ylabel='Log-score gain [nat / star]',ylim=(0,.142),
              xticks=x,xticklabels=['No color','$grz$','$griz$','$griz$ + WISE'])
        a.tick_params(axis='x',rotation=21)
        for k,y in enumerate(gain):
            a.text(k,y+.003,f'{y:.3f}',ha='center',fontsize=10)
        a.text(.03,.95,f'Same {n:,} labeled stars',transform=a.transAxes,va='top',fontsize=10)
        for j,(label,color) in enumerate((('Nominal SED',GRAY),('Class-partition SED',BLUE))):
            b.bar(np.arange(2)+(j-.5)*.34,gm[j],width=.32,label=label,color=color)
        b.set(title='Distance given giant class',ylabel='Percent',ylim=(0,35),
              xticks=[0,1],xticklabels=['Relative NMAD','Severe errors'])
        b.legend(fontsize=9,loc='upper right',frameon=False)
        for j,(label,color) in enumerate((('Full likelihood',BLUE),('Gaussian summary',ORANGE))):
            low,mid,high = distance[j]
            c.errorbar(j,mid,yerr=[[mid-low],[high-mid]],color=color,fmt='o',capsize=5,lw=2)
        c.axhline(ref,ls='--',color=GRAY,label='Held-out scale')
        c.set(title='NGC 288 common distance',ylabel='Distance [kpc]',xlim=(-.55,1.55),ylim=(8.7,10.85),
              xticks=[0,1],xticklabels=['Full likelihood','Gaussian summary'])
        c.tick_params(axis='x',rotation=17)
        c.legend(fontsize=9,loc='lower right',frameon=False)
        for panel in axes:
            panel.xaxis.grid(False)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)
    table = TABLE/f'{NAME}.csv'
    records = [dict(panel='bands',comparison=k,n=n,log_score=probe[k]['log_score'],log_score_gain=gain[j])
               for j,k in enumerate(band_names)]
    records += [dict(panel='known_giants',comparison=k,**known[k]) for k in models]
    records += [dict(panel='NGC 288',comparison=k,n=benchmark['n'],reference_kpc=ref,**comparison[k]) for k in branches]
    with table.open('w',newline='') as stream:
        fields = list(dict.fromkeys(k for r in records for k in r))
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    caption = ('Historical method comparisons on three fixed cohorts. Left: calibrated class log score '
               'on the same labeled v3.0 development identities with optional bands measured. The '
               'gain includes the effect of the fixed classifier. Middle: v3.3 development distances '
               'conditional on a known giant class. Severe errors are absolute modulus residuals '
               'over 1 mag. Right: NGC 288 median and central 68% interval with a held-out scale '
               'at 9.03 kpc. The historical SED controls and cluster inference have their own '
               'model and sample assumptions. Current DR11 QRF catalog performance is measured elsewhere.')
    outputs = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    main()
