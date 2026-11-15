#!/usr/bin/env python3
"""Show the delivered DR11 selection and the available distance-reference inventory."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v3_6_plot_shell_retention import FIGURE, INPUT, ROOT, TABLE, sha


NAME = 'F01_catalog_admission'
VAC = ROOT/'manifests/v3_6/vac_distances'
STEPS = (
    ('Native sweep rows',()),
    ('Native PSF',('outside_tile','other_release','missing_type','non_psf')),
    ('High latitude and low extinction',('sky_dust',)),
    ('Observed $r<20$',('outside_r',)),
    ('Clean optical photometry',('optical_quality',)),
    ('Within QRF support',('outside_qrf_support',)),
)


def admission(delivery):
    exclusive = delivery['exclusions']
    n = sum(exclusive.values())
    values = []
    for label,remove in STEPS:
        removed = sum(exclusive[k] for k in remove)
        n -= removed
        values.append(dict(panel='Native catalog',stage=label,rows=n,excluded=removed))
    if n!=delivery['totals']['rows']:
        raise ValueError('Frozen native selection and supported catalog do not reconcile')
    for cut in ('primary','strict'):
        values.append(dict(panel='Native catalog',stage=cut.title()+' selection',
                           rows=delivery['totals'][cut],excluded=n-delivery['totals'][cut]))
    return values


def matches(name):
    marker,table = (VAC/name/k for k in ('complete.json','inventory.csv'))
    receipt = json.loads(marker.read_text())
    if sha(table)!=receipt['outputs'][table.name]:
        raise ValueError(f'{name} inventory does not match its receipt')
    with table.open() as stream:
        source = list(csv.DictReader(stream))
    if sum(int(r['raw_rows']) for r in source)!=receipt['inventory_rows']:
        raise ValueError(f'{name} raw inventory does not reconcile')
    rows = [dict(panel='Distance inventory',source=name,
                 distance_lo_kpc=float(r['reference_distance_lo_kpc']),
                 distance_hi_kpc=float(r['reference_distance_hi_kpc']),
                 raw_reference_rows=int(r['raw_rows']),verified_supported_rows=int(r['verified_historical_position']))
            for r in source]
    return [r for r in rows if r['distance_lo_kpc']>=0],(marker,table)


def render(selection,lookup):
    with plt.rc_context(STYLE):
        fig,(ax,compare) = plt.subplots(1,2,figsize=(14.5,5.7),layout='constrained')
        y = np.arange(len(selection))
        col = [GRAY]*5+[BLUE,ORANGE,GREEN]
        counts = [r['rows'] for r in selection]
        ax.barh(y,counts,color=col,height=.68)
        ax.set(yticks=y,yticklabels=[r['stage'] for r in selection],xscale='log',
               xlabel='Catalog rows',xlim=(1e6,1.3e10),title='Photometric sample')
        ax.invert_yaxis()
        ax.yaxis.grid(False)
        for j,n in enumerate(counts):
            label = f'{n/1e9:.2f}B' if n>=1e9 else f'{n/1e6:.2f}M'
            ax.text(n*1.09,j,label,va='center',fontsize=11)
        x = np.arange(6)
        for j,(label,color) in enumerate((('SpecDis',BLUE),('SPDist',GREEN))):
            rows = lookup[label.lower()]
            raw = [r['raw_reference_rows'] for r in rows]
            matched = [r['verified_supported_rows'] for r in rows]
            dx = (j-.5)*.33
            compare.bar(x+dx,raw,width=.31,color=color,alpha=.23,edgecolor=color)
            compare.bar(x+dx,matched,width=.31,color=color,label=label)
        compare.set(yscale='log',ylim=(1e3,4e6),xticks=x,
                    xticklabels=['0–2','2–5','5–10','10–20','20–50','>50'],
                    xlabel='Reference distance [kpc]',ylabel='Distance VAC rows',
                    title='Catalog support of distance references')
        compare.xaxis.grid(False)
        compare.plot([],[],color=GRAY,alpha=.28,lw=7,label='Raw inventory')
        compare.plot([],[],color=GRAY,lw=7,label='Verified historical match')
        compare.legend(fontsize=9,frameon=False)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)


def main():
    marker = INPUT/'complete.json'
    selection = admission(json.loads(marker.read_text()))
    by_source = {}
    inputs = [marker]
    for name in ('specdis','spdist'):
        by_source[name],paths = matches(name)
        inputs.extend(paths)
    TABLE.mkdir(parents=True,exist_ok=True)
    FIGURE.mkdir(parents=True,exist_ok=True)
    table = TABLE/f'{NAME}.csv'
    rows = [*selection,*by_source['specdis'],*by_source['spdist']]
    with table.open('w',newline='') as stream:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    render(selection,by_source)
    caption = ('Left: native rows from the frozen eligible DR11.0 DECam sweeps and the catalog '
               'selection stages. Both final cuts use the same QRF-supported parent. '
               'Right: raw SpecDis and SPDist rows with a positive reference distance, and '
               'archived position-verified associations to QRF-supported PSF rows before the '
               'dwarf and near-distance cuts. The two reference catalogs can share stars. '
               'The complete raw inventories also include rows without a usable reference distance.')
    outputs = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    main()
