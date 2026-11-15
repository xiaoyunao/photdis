#!/usr/bin/env python3
"""Render magnitude–color slices of the frozen external Gaia check."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from photdis_figure_style import BLUE, GRAY, ORANGE, STYLE


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'manifests/v3_5/validation'
FIGURE = ROOT/'figures/v3_6'
TABLE = ROOT/'manifests/v3_6/figures'
NAME = 'F04_gaia_magnitude_color'
RANGES = ((14,16),(16,18),(18,19),(19,20))
COLORS = ((.2,.5),(.5,.8),(.8,1.1),(1.1,1.4))
DISTANCES = ((0,2),(2,5),(5,10))
CUTS = ('primary','strict')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells(source):
    groups = {(row['selection'],float(row['lower_kpc']),row['group']):row
              for row in source}
    rows = []
    for cut in CUTS:
        for d0,d1 in DISTANCES:
            for r0,r1 in RANGES:
                for c0,c1 in COLORS:
                    group = f'r{r0}_{r1}_color{c0:g}_{c1:g}'
                    key = cut,d0,group
                    row = groups.get(key,{})
                    n = int(row.get('geometry_n',0))
                    out = dict(selection=cut,distance_lo_kpc=d0,distance_hi_kpc=d1,
                               observed_r_lo=r0,observed_r_hi=r1,color_lo=c0,color_hi=c1,
                               reported_cell=bool(row),reference_n=n,draw_n=int(row.get('draw',0)))
                    out.update({k:v for k,v in row.items() if k.startswith('geometry_') and k!='geometry_n'})
                    if n:
                        out['bias_percent'] = 100*float(row['geometry_median_relative'])
                    rows.append(out)
    return rows


def render(rows):
    cmap = LinearSegmentedColormap.from_list('relative_bias',[BLUE,'#ffffff',ORANGE])
    cmap.set_bad('#ececec')
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,3,figsize=(15.3,8.8),layout='constrained')
        for i,cut in enumerate(CUTS):
            for j,(lo,hi) in enumerate(DISTANCES):
                ax = axes[i,j]
                subset = [r for r in rows if r['selection']==cut and r['distance_lo_kpc']==lo]
                bias = np.array([r.get('bias_percent',np.nan) for r in subset]).reshape(4,4)
                im = ax.imshow(bias,origin='lower',aspect='auto',cmap=cmap,vmin=-80,vmax=80)
                for k,row in enumerate(subset):
                    y,x = divmod(k,4)
                    val = bias[y,x]
                    rounded = int(np.rint(val)) if np.isfinite(val) else None
                    value_label = f'{rounded:+d}' if rounded else '0'
                    label = f'{value_label}%\nN={row["reference_n"]:,}' if rounded is not None else 'N=0'
                    text_color = 'white' if val<=-55 else 'black' if np.isfinite(val) else GRAY
                    ax.text(x,y,label,ha='center',va='center',fontsize=11,color=text_color)
                ax.set_title(f'{cut.capitalize()} · {lo}–{hi} kpc')
                ax.set_xticks(range(4),[f'{a:g}–{b:g}' for a,b in COLORS])
                ax.set_yticks(range(4),[f'{a}–{b}' for a,b in RANGES] if j==0 else ['']*4)
                ax.grid(False)
                if i==1:
                    ax.set_xlabel('Dereddened $g-r$ [mag]')
                if j==0:
                    ax.set_ylabel('Observed $r$ [mag]')
        fig.colorbar(im,ax=axes,extend='both',pad=.022,label='Median relative distance bias [%]')
        for extension in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{extension}',dpi=300)
        plt.close(fig)


def main():
    marker,path = SOURCE/'complete.json',SOURCE/'gaia_joint.csv'
    receipt = json.loads(marker.read_text())
    if sha(path)!=receipt['outputs'][path.name]:
        raise ValueError('The Gaia table differs from its delivery receipt')
    with path.open() as stream:
        rows = cells(list(csv.DictReader(stream)))
    FIGURE.mkdir(parents=True,exist_ok=True)
    TABLE.mkdir(parents=True,exist_ok=True)
    table = TABLE/f'{NAME}.csv'
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with table.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    render(rows)
    caption = ('Frozen v3.5 probability draw. Counts and median fractional residuals use the '
               'reference-distance shells and qualifying Gaia astrometry (corrected S/N at least 10). '
               'Statistics are unweighted within each reference subset. Gray cells have no qualifying '
               'reference. The color scale saturates at ±80%; printed values retain the full range. '
               'The table also gives NMAD, severe errors, interval width, and reference-center containment.')
    files = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,cells=len(rows),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (marker,path)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n')


if __name__ == '__main__':
    main()
