#!/usr/bin/env python3
"""Plot the frozen survey coverage, imaging depth and nearby surface density."""
import argparse
import csv
import json
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np

from photdis.stage1 import galactic
from photdis_figure_style import STYLE
from v3_6_plot_shell_retention import ROOT, INPUT, FIGURE, TABLE, sha


NAME = 'F06_coverage_and_near_counts'


def prepare(path, expected_area):
    with np.load(path,allow_pickle=False) as z:
        area,depth,mass = [z[k] for k in ('area_deg2','r_limiting_mag','mass')]
    nside = hp.npix2nside(len(area))
    np.testing.assert_allclose(area.sum(),expected_area,rtol=0,atol=1e-7)
    covered = np.flatnonzero(area>0)
    pixel_area = hp.nside2pixarea(nside,degrees=True)
    ra,dec = hp.pix2ang(nside,covered,nest=True,lonlat=True)
    lon,lat = galactic(ra,dec)
    bins = {'area_fraction':area[covered]/pixel_area,'r_depth':depth[covered],
            'near_density':mass[0,0,0,covered]/area[covered]}
    tab = {'healpix_nested':covered,'ra_deg':ra,'dec_deg':dec,'galactic_l_deg':lon,
           'galactic_b_deg':lat,'random_area_deg2':area[covered],**bins}
    for j,cut in enumerate(('primary','strict')):
        for k,label in enumerate(('0_2','2_5','5_10','above_10')):
            tab[f'{cut}_{label}_mass'] = mass[j,0,k,covered]
    return tab,bins,nside,float(mass[0,0,0,area<=0].sum())


def log_scale(values):
    positive = values[np.isfinite(values)&(values>0)]
    lo,hi = np.quantile(positive,[.02,.98])
    return LogNorm(lo,hi)


def render(tab,bins):
    coords = (('ra_deg','dec_deg','Right ascension','Declination'),
              ('galactic_l_deg','galactic_b_deg','Galactic longitude','Galactic latitude'))
    series = (
        ('area_fraction','Imaging coverage','Covered pixel fraction','Blues',log_scale(bins['area_fraction'])),
        ('r_depth','$r$-band depth','Limiting magnitude [mag]','cividis',
         Normalize(*np.nanquantile(bins['r_depth'],[.02,.98]))),
        ('near_density','Primary · 0–2 kpc','Posterior-weighted sources [deg$^{-2}$]',
         'viridis',log_scale(bins['near_density'])))
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,3,figsize=(16,7.4),subplot_kw={'projection':'aitoff'},
                                layout='constrained')
        for j,(key,title,cbar,cmap,norm) in enumerate(series):
            values = bins[key]
            good = np.isfinite(values)&((values>0) if isinstance(norm,LogNorm) else True)
            for i,(a,b,xlabel,ylabel) in enumerate(coords):
                ax = axes[i,j]
                x,y = np.deg2rad((180-tab[a])%360-180),np.deg2rad(tab[b])
                ax.scatter(x,y,c='#d9d9d9',s=3,linewidths=0,rasterized=True)
                m = ax.scatter(x[good],y[good],c=values[good],s=4,linewidths=0,
                               cmap=cmap,norm=norm,rasterized=True)
                ax.set_longitude_grid(60)
                ax.set_latitude_grid(30)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{(-round(np.rad2deg(v)))%360}°'))
                ax.tick_params(labelsize=9)
                ax.grid(alpha=.27)
                ax.set_xlabel(xlabel)
                if j==0:
                    ax.set_ylabel(ylabel)
                if i==0:
                    ax.set_title(title)
            fig.colorbar(m,ax=axes[:,j],orientation='horizontal',shrink=.84,
                         pad=.06,extend='both',label=cbar)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)


def main(path):
    marker = INPUT/'complete.json'
    record = json.loads(marker.read_text())
    digest = sha(path)
    if digest!=record['outputs']['spatial.npz']:
        raise ValueError('Sky map data differ from the frozen v3.5 delivery')
    tab,bins,nside,uncovered = prepare(path,record['area_deg2'])
    TABLE.mkdir(parents=True,exist_ok=True)
    FIGURE.mkdir(parents=True,exist_ok=True)
    csv_path = TABLE/f'{NAME}.csv'
    with csv_path.open('w',newline='') as stream:
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(tab)
        writer.writerows(zip(*tab.values()))
    render(tab,bins)
    caption = ('Equatorial and Galactic Aitoff projections of the same covered HEALPix cells. '
               'Coverage and area-weighted limiting magnitude come from frozen DR11 imaging randoms. '
               'The near surface density divides summed primary QRF probability in 0–2 kpc by '
               'random-covered area. Cell centers represent entire footprints. Gray points represent '
               'covered pixels with missing depth or zero displayed density. Color ranges use the '
               '2nd and 98th percentiles of positive or usable values. The table retains all three '
               'near shells and the beyond-10 kpc mass for both cuts.')
    outputs = (csv_path,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    result = dict(caption=caption,dpi=300,nside=nside,pixel_order='nested',
                  covered_pixels=len(tab['healpix_nested']),area_deg2=record['area_deg2'],
                  uncovered_primary_0_2_mass=uncovered,spatial_input_sha256=digest,
                  delivery_receipt_sha256=sha(marker),
                  outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs})
    (TABLE/f'{NAME}.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spatial',type=Path)
    main(parser.parse_args().spatial)
