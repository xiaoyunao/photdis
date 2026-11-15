#!/usr/bin/env python3
"""Compare frozen QRF shell counts across footprint, depth and Galactic latitude."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v2_1_near import csv_rows
from v3_6_plot_shell_retention import FIGURE, INPUT, ROOT, TABLE, sha


NAME = 'F07_radial_sensitivity'
NEAR = ('0–2 kpc','2–5 kpc','5–10 kpc')
LATITUDE = (('30','45'),('45','60'),('60','90'))
MASKS = {'measured_area':'All covered area','above_20pct_pixel':'At least 20% covered',
         'above_50pct_pixel':'At least 50% covered','shallow_half':'Shallower half','deep_half':'Deeper half'}
SHELLS = dict(zip(('0_2','2_5','5_10','above_10'),(*NEAR,'Beyond 10 kpc')))


def verified_table(path,marker,output_key):
    body = json.loads(marker.read_text())
    if sha(path)!=body['outputs'][output_key]:
        raise ValueError(f'Input does not match frozen receipt: {path}')
    with path.open() as stream:
        return list(csv.DictReader(stream))


def latitude(raw):
    cells = defaultdict(lambda: np.zeros((2,2)))
    for r in raw:
        if r['cohort']!='all':
            continue
        key = r['hemisphere'],r['latitude_min'],r['latitude_max'],r['shell']
        i = ('Primary selection','Strict selection').index(r['selection'])
        cells[key][i] += float(r['area_deg2']),float(r['posterior_mass'])
    rows = []
    for (hem,lo,hi,shell),values in sorted(cells.items()):
        area,mass = values.T
        np.testing.assert_allclose(area[0],area[1],rtol=0,atol=1e-7)
        if mass[1]>mass[0] or area[0]<=0:
            raise ValueError('Frozen latitude counts and cut ordering are inconsistent')
        rows.append(dict(axis='galactic_latitude',group=f'{hem.title()} {lo}–{hi}°',shell=shell,
            area_deg2=area[0],primary_mass=mass[0],strict_mass=mass[1],
            primary_per_deg2=mass[0]/area[0],strict_per_deg2=mass[1]/area[0],retention=mass[1]/mass[0]))
    return rows


def depth(raw):
    rows = []
    fields = ('area_deg2','primary_mass','strict_mass','primary_per_deg2','strict_per_deg2','retention')
    for r in raw:
        rows.append(dict(axis='coverage_and_depth',group=MASKS[r['footprint']],shell=SHELLS[r['shell']],
                         **{key:float(r[key]) for key in fields}))
    return rows


def plot(rows):
    groups = ['North','South','Coverage and depth']
    lookup = {(r['group'],r['shell']):r for r in rows}
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(2,3,figsize=(16.5,9.),layout='constrained',
                               gridspec_kw={'width_ratios':[1,1,1.25]})
        for i,group in enumerate(groups):
            labels = ([f'{a}–{b}°' for a,b in LATITUDE] if i<2 else
                      ['All','20%+','50%+','Shallower','Deeper'])
            keys = [f'{group} {k}' for k in labels] if i<2 else list(MASKS.values())
            a,b = axs[:,i]
            x = np.arange(len(keys))
            for shell,color in zip(NEAR,(BLUE,GREEN,ORANGE)):
                t = [lookup[k,shell] for k in keys]
                a.plot(x,[r['primary_per_deg2'] for r in t],'-o',color=color,ms=5,label=shell)
                a.plot(x,[r['strict_per_deg2'] for r in t],'--s',color=color,ms=4,alpha=.84)
                b.plot(x,[r['retention'] for r in t],'-o',color=color,ms=5)
            a.set(title=group,yscale='log',ylim=(1,2300))
            b.set(ylim=(0,1))
            b.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            for ax in (a,b):
                ax.set_xticks(x,labels,rotation=(23 if i==2 else 0))
                ax.xaxis.grid(False)
            b.set_xlabel('Absolute Galactic latitude' if i<2 else 'Imaging area and depth')
        axs[0,0].set_ylabel('Posterior-weighted density\n[stars per square degree]')
        axs[1,0].set_ylabel('Strict / primary')
        axs[0,0].plot([],[],color=GRAY,ls='-',label='Primary')
        axs[0,0].plot([],[],color=GRAY,ls='--',label='Strict')
        handles,labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',fontsize=11,frameon=False,ncols=5)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)


def main():
    angular = INPUT/'angular_density.csv'
    delivery = INPUT/'complete.json'
    sensitivity = TABLE/'F06_shell_density_coverage_sensitivity.csv'
    coverage = TABLE/'F06_shell_density.json'
    rows = latitude(verified_table(angular,delivery,angular.name))
    rows += depth(verified_table(sensitivity,coverage,str(sensitivity.relative_to(ROOT))))
    table = TABLE/f'{NAME}.csv'
    csv_rows(table,rows)
    plot(rows)
    caption = ('Area-weighted density from full frozen QRF posterior masses. Latitude panels use '
               'the v3.5 longitude sectors aggregated within each hemisphere. The right panels '
               'use the frozen random coverage and split usable pixels by the area-weighted '
               'median r depth. Imaging sensitivity and Galactic position covary. The plot '
               'summarizes a magnitude-limited catalog, with a shared measured imaging-area '
               'denominator for both cuts. The table preserves the combined beyond-10 kpc '
               'tail of the imaging sensitivity calculation.')
    outputs = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (angular,delivery,sensitivity,coverage)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    main()
