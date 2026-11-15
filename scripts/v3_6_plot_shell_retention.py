#!/usr/bin/env python3
"""Plot the selection cost in the frozen v3.5 posterior-weighted shell counts."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, STYLE


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'manifests/v3_5/delivery'
TABLE = ROOT/'manifests/v3_6/figures'
FIGURE = ROOT/'figures/v3_6'
NAME = 'F07_shell_retention'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = INPUT/'complete.json'
    record = json.loads(receipt.read_text())
    path = INPUT/'shell_mass.csv'
    if sha(path)!=record['outputs'][path.name]:
        raise ValueError('Shell counts changed since v3.5 delivery')
    with path.open() as f:
        shell = {(r['selection'],r['shell']):float(r['model_count'])
                 for r in csv.DictReader(f) if r['cohort']=='all'}
    cuts = ('Primary selection','Strict selection')
    labels = ('0–2 kpc','2–5 kpc','5–10 kpc','>10 kpc')
    counts = [[shell[cut,k] for k in labels[:3]] for cut in cuts]
    counts = np.asarray([r+[record['totals'][cut]-sum(r)]
                         for cut,r in zip(('primary','strict'),counts)])
    if not (counts>0).all() or (counts[1]>counts[0]).any():
        raise ValueError('Strict counts are not nested in the primary sample')
    retained = counts[1]/counts[0]
    TABLE.mkdir(parents=True,exist_ok=True)
    FIGURE.mkdir(parents=True,exist_ok=True)
    with (TABLE/f'{NAME}.csv').open('w',newline='') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow(('distance_shell','primary_posterior_count','strict_posterior_count','strict_over_primary'))
        writer.writerows(zip(labels,*counts,retained))
    with plt.rc_context(STYLE):
        fig,(ax,fx) = plt.subplots(1,2,figsize=(12.5,4.3),layout='constrained')
        x = np.arange(len(labels))
        for i,(cut,color) in enumerate(zip(cuts,(BLUE,GREEN))):
            ax.bar(x+(i-.5)*.36,counts[i],width=.36,color=color,label=cut)
        ax.set(yscale='log',ylabel='Posterior-weighted source count',title='Radial counts')
        ax.legend(frameon=False)
        fx.bar(x,retained,color=GREEN,width=.52)
        fx.axhline(1,color=GRAY,linewidth=1,linestyle='--')
        for j,y in enumerate(retained):
            fx.text(j,y+.024,f'{y:.1%}',ha='center')
        fx.set(ylim=(0,1.06),ylabel='Strict / primary',title='Retention under the strict cut')
        for panel in (ax,fx):
            panel.set_xticks(x,labels)
            panel.xaxis.grid(False)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)
    caption = ('The frozen v3.5 primary and strict selections applied to the same DECam inventory. '
               'Sources contribute their complete QRF posterior mass. The final bin retains mass beyond '
               '10 kpc. Counts are conditional on the photometric catalog and its selection.')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,inputs={
        str(p.relative_to(ROOT)):sha(p) for p in (receipt,path)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in
                 (TABLE/f'{NAME}.csv',FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')},
        dpi=300),indent=2)+'\n')


if __name__ == '__main__':
    main()
