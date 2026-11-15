#!/usr/bin/env python3
"""Map both selections and strict retention with a common measured sky area."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from photdis_figure_style import STYLE
from v3_6_plot_shell_retention import FIGURE, INPUT, ROOT, TABLE, sha
from v3_6_plot_sky import log_scale, prepare


NAME = 'F06_shell_density'
SHELLS = ('0_2','2_5','5_10','above_10')
LABELS = ('0–2 kpc','2–5 kpc','5–10 kpc','>10 kpc')
COORDINATES = {'equatorial':('ra_deg','dec_deg','Right ascension','Declination'),
               'galactic':('galactic_l_deg','galactic_b_deg','Galactic longitude','Galactic latitude')}


def densities(p):
    area = p['random_area_deg2']
    primary = np.column_stack([p[f'primary_{k}_mass'] for k in SHELLS])
    strict = np.column_stack([p[f'strict_{k}_mass'] for k in SHELLS])
    if (strict-primary>1e-9).any():
        raise ValueError('Strict count is outside the primary count')
    ratio = np.divide(strict,primary,out=np.full_like(strict,np.nan),where=primary>0)
    return primary/area[:,None],strict/area[:,None],ratio


def bounds(p):
    area,depth,a = (p[k] for k in ('random_area_deg2','r_depth','area_fraction'))
    full = a>0
    finite = (a>=.2)&np.isfinite(depth)
    order = np.argsort(depth[finite])
    included = depth[finite][order]
    cumulative = np.cumsum(area[finite][order])
    pivot = included[np.searchsorted(cumulative,cumulative[-1]/2)]
    return dict(measured_area=full,above_20pct_pixel=a>=.2,above_50pct_pixel=a>=.5,
                shallow_half=finite&(depth<pivot),deep_half=finite&(depth>=pivot)),float(pivot)


def render(p,system,styles):
    ra,dec,xlabel,_ = COORDINATES[system]
    x,y = np.deg2rad((180-p[ra])%360-180),np.deg2rad(p[dec])
    primary,strict,ratio = styles
    path = FIGURE/f'{NAME}_{system}'
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(3,4,figsize=(17.4,9.1),subplot_kw={'projection':'aitoff'},layout='constrained')
        for j,label in enumerate(LABELS):
            norm = log_scale(np.r_[primary[:,j],strict[:,j]])
            for i,(values,name,scale) in enumerate(((primary,'Primary',norm),(strict,'Strict',norm),
                                                   (ratio,'Strict / primary',None))):
                ax = axs[i,j]
                z = values[:,j]
                live = np.isfinite(z)&((z>0) if scale else True)
                ax.scatter(x,y,c='#ddd',s=3,linewidths=0,rasterized=True)
                if scale:
                    im = ax.scatter(x[live],y[live],c=z[live],s=4,linewidths=0,rasterized=True,
                                    cmap='viridis',norm=scale)
                else:
                    im_ratio = ax.scatter(x[live],y[live],c=z[live],s=4,linewidths=0,rasterized=True,
                                          cmap='cividis',vmin=0,vmax=1)
                ax.set_longitude_grid(90)
                ax.set_latitude_grid(45)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{(-round(np.rad2deg(v)))%360}°'))
                ax.tick_params(labelsize=8)
                ax.grid(alpha=.2)
                if i==0:
                    ax.set_title(label)
                if j==0:
                    ax.set_ylabel(name,fontsize=14)
                if i==2:
                    ax.set_xlabel(xlabel,fontsize=13)
            # Comparative densities share a scale for each shell.
            ticks = np.geomspace(norm.vmin,norm.vmax,3)
            bar = fig.colorbar(im,ax=axs[:2,j],orientation='horizontal',shrink=.72,pad=.035,
                               extend='both',ticks=ticks,format=FuncFormatter(lambda v,_:f'{v:.0f}' if v>=10 else f'{v:.2g}'))
            bar.minorticks_off()
        fig.colorbar(im_ratio,ax=axs[2,:],orientation='vertical',shrink=.74,pad=.016,label='Strict / primary')
        fig.suptitle(f'{system.title()} coordinates · posterior-weighted sources per square degree')
        for ext in ('png','pdf'):
            fig.savefig(path.with_suffix('.'+ext),dpi=300)
        plt.close(fig)
    return (path.with_suffix('.png'),path.with_suffix('.pdf'))


def main(path):
    delivery = INPUT/'complete.json'
    receipt = json.loads(delivery.read_text())
    digest = sha(path)
    if digest!=receipt['outputs']['spatial.npz']:
        raise ValueError('The spatial payload differs from the archived delivery')
    p,_,nside,_ = prepare(path,receipt['area_deg2'])
    fields = densities(p)
    cuts,depth_cut = bounds(p)
    area = p['random_area_deg2']
    rows = []
    for scope,mask in cuts.items():
        for j,label in enumerate(SHELLS):
            d = area[mask].sum()
            a,b = (float(area[mask]@z[mask,j]) for z in fields[:2])
            rows.append(dict(footprint=scope,shell=label,pixels=int(mask.sum()),area_deg2=d,
                primary_mass=a,strict_mass=b,primary_per_deg2=a/d,strict_per_deg2=b/d,retention=b/a))
    TABLE.mkdir(parents=True,exist_ok=True)
    FIGURE.mkdir(parents=True,exist_ok=True)
    table = TABLE/f'{NAME}_coverage_sensitivity.csv'
    with table.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    pictures = [file for system in COORDINATES for file in render(p,system,fields)]
    caption = ('Equatorial and Galactic Aitoff rotations of the same covered sky cells. '
               'For each distance shell the primary and strict densities share a color scale '
               'from the 2nd to 98th percentiles of positive pixel densities. The bottom row '
               'gives strict retention. Blank sky is outside the random-area coverage. The '
               'tail includes all probabilities beyond 10 kpc. The sensitivity table uses '
               'the same cells and gives cuts at 20% and 50% covered area and a depth split '
               'among cells with at least 20% coverage.')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,nside=nside,
        depth_split_r_limiting_mag=depth_cut,spatial_input_sha256=digest,
        delivery_receipt_sha256=sha(delivery),
        plotted_cells_csv='manifests/v3_6/figures/F06_coverage_and_near_counts.csv',
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [table,*pictures]}),indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spatial',type=Path)
    main(parser.parse_args().spatial)
