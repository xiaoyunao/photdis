#!/usr/bin/env python3
"""Render optical and optional-band color planes for the verified blue cases."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, ORANGE, STYLE
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


NAME = 'F05_blue_color_planes'
MARKER = ROOT/'manifests/v3_6/blue_replay/complete.json'
BANDS = ('G','R','I','Z','W1','W2')
LABELS = {k:k.lower() for k in BANDS[:4]}|{k:k for k in BANDS[4:]}
PLANES = ((('G','R'),('R','Z')), (('G','R'),('Z','W1')),
          (('R','Z'),('Z','W1')), (('Z','W1'),('W1','W2')),
          (('G','R'),('R','I')), (('R','I'),('I','Z')))
CASE = (('faint_blue_near','Faint blue foreground',ORANGE),
        ('blue_2_to_5','Blue 2–5 kpc references',BLUE))


def label(color):
    a,b = (LABELS[x] for x in color)
    return f'{a} − {b} [mag]'


def main(path):
    receipt = json.loads(MARKER.read_text())
    if sha(path)!=receipt['outputs']['blue_objects.npz']:
        raise ValueError('F05 inputs differ from the verified blue replay')
    with np.load(path,allow_pickle=False) as z:
        t = {k:z[k] for k in ('ls_id','sample','TYPE','band_state','band_mask','distance_ratio',
                              *[f'{x}_dereddened' for x in BANDS])}
    if not (t['TYPE']=='PSF').all():
        raise ValueError('F05 requires verified native PSFs')
    state = t['band_state']
    available = {k:t['band_mask'][:,i] & np.isfinite(t[f'{k}_dereddened']) for i,k in enumerate(BANDS)}
    for k in BANDS[4:]:
        available[k] &= state[:,BANDS.index(k)]==4
    FIGURE.mkdir(parents=True,exist_ok=True)
    TABLE.mkdir(parents=True,exist_ok=True)
    rows = []
    counts = {}
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(2,3,figsize=(15.1,8.6),layout='constrained')
        for i,(ax,(x,y)) in enumerate(zip(axs.ravel(),PLANES),1):
            use = np.logical_and.reduce([available[k] for k in (*x,*y)])
            cx,cy = [t[f'{a}_dereddened']-t[f'{b}_dereddened'] for a,b in (x,y)]
            counts[str(i)] = {'eligible':int(use.sum()),'missing_or_weak':int((~use).sum())}
            for sample,title,color in CASE:
                eligible = use&(t['sample']==sample)
                ax.scatter(cx[eligible],cy[eligible],s=17,c=color,linewidths=0,alpha=.74,
                           label=f'{title} ({eligible.sum():,})')
                for j in np.flatnonzero(eligible):
                    rows.append(dict(plane=i,x_color='-'.join(x),y_color='-'.join(y),
                        legacy_id=t['ls_id'][j],sample=sample,color_x=cx[j],color_y=cy[j],
                        qrf_gaia_distance_ratio=t['distance_ratio'][j]))
            ax.set(xlabel=label(x),ylabel=label(y))
            ax.legend(fontsize=8.5,loc='best',frameon=False)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)
    table = TABLE/f'{NAME}.csv'
    with table.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    caption = ('The same two blue cohorts as F03 in six color planes. Photometry is '
               'dereddened. Each plane requires only its own bands, with usable WISE states '
               'for W1 and W2 and the native valid-band flag for i. The raw count in each '
               'legend can change across planes. Distance ratios in the table are diagnostic '
               'for the original frozen Gaia sample.')
    outputs = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,planes=counts,
        blue_replay_receipt_sha256=sha(MARKER),native_table_sha256=sha(path),
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    main(p.parse_args().data)
