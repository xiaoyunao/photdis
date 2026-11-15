#!/usr/bin/env python3
"""Show which observed color and magnitude cohorts contribute to each shell."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from photdis_figure_style import BLUE, GRAY, ORANGE, STYLE
from v3_6_plot_shell_retention import FIGURE, INPUT, ROOT, TABLE, sha


NAME = 'F07_radial_photometric_composition'
SHELLS = ('0–2','2–5','5–10','10–20','20–50','>50')
LABELS = ('Bright red analogs','Faint blue','Other photometry')
CHOICES = ROOT/'configs/v3_4_near_release_domain.json'


def prepare(path,receipt):
    with np.load(path,allow_pickle=False) as archived:
        matrix = archived['mass'].sum(axis=-1)
    totals = [receipt['totals'][k] for k in ('primary','strict')]
    np.testing.assert_allclose(matrix[:,0,:].sum(axis=1),totals,rtol=0,atol=1e-6)
    groups = np.concatenate([matrix[:,1:,:],matrix[:,:1,:]-matrix[:,1:,:].sum(axis=1,keepdims=True)],axis=1)
    if (groups<0).any() or (groups[1]>groups[0]).any():
        raise ValueError('Full PDF sums or observed groups do not reconcile')
    return matrix[:,0,:],groups


def plot(all_counts,groups,share):
    x = np.arange(len(SHELLS))
    colors = (ORANGE,BLUE,GRAY)
    with plt.rc_context(STYLE):
        fig,(a,b) = plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        for cut,offset in enumerate((-.20,.20)):
            bottom = np.zeros(len(x))
            for j,color in enumerate(colors):
                a.bar(x+offset,share[cut,j],width=.35,bottom=bottom,color=color,
                      alpha=.94 if cut==0 else .48,hatch=None if cut==0 else '//')
                bottom += share[cut,j]
        a.set(title='Photometry of the assigned shells',ylabel='Share of posterior-weighted count',
              ylim=(0,1.025))
        a.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        a.text(.01,-.16,'Left bars: primary    Hatched bars: strict',
               transform=a.transAxes,fontsize=10)
        for j,(name,color) in enumerate(zip(LABELS,colors)):
            b.plot(x,groups[1,j]/groups[0,j],marker='o',color=color,label=name,lw=2)
        b.plot(x,all_counts[1]/all_counts[0],marker='s',color='black',ls='--',label='Full sample',lw=1.5)
        b.set(title='Retention under the strict cut',ylabel='Strict / primary',ylim=(0,1.025))
        b.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        b.legend(fontsize=10,frameon=False,loc='best')
        for panel in (a,b):
            panel.set_xticks(x,SHELLS)
            panel.set_xlabel('Assigned distance [kpc]')
            panel.xaxis.grid(False)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)


def main(path):
    marker = INPUT/'complete.json'
    receipt = json.loads(marker.read_text())
    digest = sha(path)
    if digest!=receipt['outputs']['spatial.npz']:
        raise ValueError('The spatial data differs from the frozen v3.5 delivery')
    all_counts,groups = prepare(path,receipt)
    share = groups/all_counts[:,None,:]
    FIGURE.mkdir(parents=True,exist_ok=True)
    TABLE.mkdir(parents=True,exist_ok=True)
    table = TABLE/f'{NAME}.csv'
    with table.open('w',newline='') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow(('cohort','distance_kpc','primary_mass','strict_mass','primary_share',
                         'strict_share','strict_over_primary'))
        for j,label in enumerate(LABELS):
            for k,shell in enumerate(SHELLS):
                a,b = groups[:,j,k]
                writer.writerow((label,shell,a,b,*share[:,j,k],b/a))
        for k,shell in enumerate(SHELLS):
            a,b = all_counts[:,k]
            writer.writerow(('Full sample',shell,a,b,1.,1.,b/a))
    plot(all_counts,groups,share)
    caption = ('Full native QRF posterior masses by assigned distance, observed photometric cohort '
               'and selection. Bright red analogs use the frozen v3.4 observable domain '
               'conditions. Faint blue means 18 ≤ r < 20 and 0.5 ≤ dereddened g − r < 0.8. '
               'The other-photometry group is the disjoint remainder. All bins including '
               'distances beyond 50 kpc are summed from the delivered PDFs. These shares '
               'and retention fractions describe the magnitude-limited predicted catalog.')
    outputs = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,spatial_input_sha256=digest,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (marker,CHOICES)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('spatial',type=Path)
    main(p.parse_args().spatial)
