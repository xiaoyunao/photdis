#!/usr/bin/env python3
"""Show selection survival by archived stellar class and reference distance."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v2_1_near import csv_rows
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


NAME = 'F09_class_retention'
SOURCES = ('specdis','spdist')
CLASSES = (('0','Dwarf',BLUE),('1','Giant',ORANGE),('2','Other stars',GREEN),('-1','Unknown class',GRAY))
SHELLS = dict(zip((0.,2.,5.,10.,20.,50.),('0–2','2–5','5–10','10–20','20–50','>50')))


def source(kind):
    folder = ROOT/'manifests/v3_6/vac_flow'/kind
    marker,table = (folder/k for k in ('complete.json','performance.csv'))
    record = json.loads(marker.read_text())
    if sha(table)!=record['outputs'][table.name]:
        raise ValueError(f'Unverified {kind} performance')
    with table.open() as f:
        values = {(r['selection'],float(r['distance_lo_kpc']),r['class_code']):int(r['outside_qrf_train_n'])
                  for r in csv.DictReader(f) if r['measurement_domain']=='all'
                  and r['reference_quality']=='legacy_quality'}
    output = []
    for distance,label in SHELLS.items():
        parts = []
        for code,cls,color in CLASSES:
            n,p,s = (values[cut,distance,code] for cut in ('supported','primary','strict'))
            if not 0<=s<=p<=n:
                raise ValueError('Stellar class counts violate nested selection')
            parts.append([n,p,s])
            output.append(dict(source=kind,ref_distance_kpc=label,reference_class=cls,
                supported=n,primary=p,strict=s,primary_retention=p/n if n else None,
                strict_retention=s/n if n else None))
        np.testing.assert_array_equal(np.asarray(parts).sum(axis=0),
                                      [values[k,distance,'all'] for k in ('supported','primary','strict')])
    return output,(marker,table)


def render(rows):
    values = {(r['source'],r['ref_distance_kpc'],r['reference_class']):r for r in rows}
    x = np.arange(len(SHELLS))
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(2,2,figsize=(13.1,8.2),layout='constrained',sharex=True,sharey=True)
        for i,kind in enumerate(SOURCES):
            for j,cut in enumerate(('primary','strict')):
                ax = axs[i,j]
                for code,cls,color in CLASSES:
                    strata = [values[kind,d,cls] for d in SHELLS.values()]
                    retained = np.asarray([r[f'{cut}_retention'] for r in strata],dtype=float)
                    plenty = np.asarray([r['supported']>=20 for r in strata])
                    ax.plot(x,np.where(plenty,retained,np.nan),'-o',ms=5,c=color,label=cls)
                    small = ~plenty & np.isfinite(retained)
                    ax.plot(x[small],retained[small],'o',mfc='white',mec=color,ms=5)
                label = 'SpecDis' if kind=='specdis' else 'SPDist'
                ax.set(title=f'{label} | {cut.title()}',ylim=(-.0001,1.18),
                       xticks=x,xticklabels=list(SHELLS.values()))
                ax.set_yscale('symlog',linthresh=.001)
                ax.set_yticks([0,.001,.01,.1,1],['0%','0.1%','1%','10%','100%'])
                ax.xaxis.grid(False)
                if i==1:
                    ax.set_xlabel('Reference distance [kpc]')
        fig.supylabel('Fraction of matched reference stars retained')
        handles,labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',frameon=False,ncols=4)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)


def main():
    rows,inputs = [],[]
    for kind in SOURCES:
        r,paths = source(kind)
        rows.extend(r)
        inputs.extend(paths)
    table = TABLE/f'{NAME}.csv'
    csv_rows(table,rows)
    render(rows)
    caption = ('Per-source selection survival among verified native PSFs with an archived '
               'quality distance, outside present QRF training. The denominator is the '
               'supported reference count of that class and distance shell. Hollow markers '
               'have fewer than 20 supported references. A class with zero supported stars '
               'has no rate. The vertical axis uses a linear scale below 0.1% and a '
               'logarithmic scale above. Source targeting and shared observations limit '
               'the interpretation to the displayed reference subsets.')
    files = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n')


if __name__=='__main__':
    main()
