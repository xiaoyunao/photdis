#!/usr/bin/env python3
"""Inspect public DR11 South r coadds around four frozen anomaly coordinates."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.patches import Circle
import numpy as np

from photdis_figure_style import ORANGE, STYLE
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


NAME = 'F03_selected_coadd_cutouts'
DATA = ROOT/'data/v3_6/inputs/blue_cutouts'
REPLAY = ROOT/'manifests/v3_6/blue_replay'
SERVER = 'https://www.legacysurvey.org/viewer/fits-cutout'
SCALE = .262


def anchors(rows):
    near = [r for r in rows if r['sample']=='faint_blue_near']
    far = [r for r in rows if r['sample']=='blue_2_to_5']
    def ratio(row):
        return float(row['distance_ratio'])
    def center(group):
        median = np.median([ratio(row) for row in group])
        return min(group,key=lambda row:(abs(ratio(row)-median),int(row['ls_id'])))
    old, = [r for r in near if r['old_anomaly_gaia_match']=='True']
    return (('Foreground and historical overlap',old),('Typical foreground error',center(near)),
            ('Largest 2–5 kpc underestimate',min(far,key=ratio)),('Typical 2–5 kpc error',center(far)))


def read(item):
    selection,row = item
    ra,dec = float(row['RA']),float(row['DEC'])
    name = row['ls_id']+'.fits'
    path = DATA/name
    query = urlencode(dict(ra=f'{ra:.12f}',dec=f'{dec:.12f}',layer='ls-dr11-south',
                           pixscale=SCALE,width=80,height=80,bands='r'))
    url = SERVER+'?'+query
    if not path.exists():
        with urlopen(Request(url,headers={'User-Agent':'Photdis astronomical cutout analysis'}),
                     timeout=25) as response:
            content = response.read()
        if not content.startswith(b'SIMPLE'):
            raise ValueError(f'Legacy Survey returned a non-FITS payload for {row["ls_id"]}')
        path.write_bytes(content)
    with fits.open(path) as f:
        z = f[0].data.copy()
        wcs = WCS(f[0].header)
    x,y = wcs.world_to_pixel_values(ra,dec)
    if z.shape!=(80,80) or not (38<x<42 and 38<y<42):
        raise ValueError(f'Unexpected cutout geometry for {row["ls_id"]}')
    u,v = np.meshgrid((np.arange(z.shape[1])-x)*SCALE,(np.arange(z.shape[0])-y)*SCALE)
    radius = np.hypot(u,v)
    high = (radius<=1.5)&np.isfinite(z)
    edge = (radius>5.)&(radius<9.)&np.isfinite(z)
    background = np.median(z[edge])
    intensity = np.maximum(z[high]-background,0)
    cx = intensity@u[high]/intensity.sum()
    cy = intensity@v[high]/intensity.sum()
    out = dict(selection=selection,legacy_id=row['ls_id'],gaia_source_id=row['source_id'],
               ra_deg=ra,dec_deg=dec,file=name,sha256=sha(path),cutout_url=url,
               qrf_gaia_distance_ratio=float(row['distance_ratio']),strict=row['strict'],
               r_fit_reduced_chi_square=float(row['RCHISQ_R']),r_neighbor_flux_fraction=float(row['FRACFLUX_R']),
               centroid_radius_arcsec=1.5,local_centroid_offset_arcsec=float(np.hypot(cx,cy)))
    bounds = ((-x-.5)*SCALE,(z.shape[1]-x-.5)*SCALE,(-y-.5)*SCALE,(z.shape[0]-y-.5)*SCALE)
    return out,z-background,bounds


def main():
    source = REPLAY/'blue_objects.csv'
    marker = REPLAY/'complete.json'
    if sha(source)!=json.loads(marker.read_text())['outputs'][source.name]:
        raise ValueError('Cutout targets differ from the verified blue replay')
    with source.open() as f:
        selected = anchors(list(csv.DictReader(f)))
    DATA.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(read,selected))
    FIGURE.mkdir(parents=True,exist_ok=True)
    TABLE.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,4,figsize=(14.9,4.1),layout='constrained')
        titles = ('Historical foreground','Typical foreground','Largest underestimate','Typical 2–5 kpc case')
        for ax,title,(r,z,extent) in zip(axes,titles,outputs):
            lo,hi = np.nanpercentile(z,[2,99.6])
            ax.imshow(z,origin='lower',extent=extent,cmap='gray_r',
                      norm=AsinhNorm(linear_width=max((hi-lo)/30,1e-9),vmin=lo,vmax=hi),
                      interpolation='nearest')
            ax.add_patch(Circle((0,0),1.5,edgecolor=ORANGE,facecolor='none',lw=1.4))
            ax.set(title=title,xlabel='Image offset [arcsec]',xlim=(-7,7),ylim=(-7,7))
            ax.text(.05,.06,f"QRF / Gaia = {r['qrf_gaia_distance_ratio']:.2f}",transform=ax.transAxes,
                    fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
            ax.grid(False)
        axes[0].set_ylabel('Image offset [arcsec]')
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)
    table = TABLE/f'{NAME}.csv'
    tab = [item[0] for item in outputs]
    with table.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(tab[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(tab)
    caption = ('Selected DR11 South r-band viewer coadds. Circles mark 1.5 arcsec around the '
               'frozen catalog coordinate. The background-subtracted centroid uses positive '
               'flux inside that radius. Image stretches are local. These examples detect '
               'visible gross offsets or neighbors at the coadd resolution. They do not give '
               'individual exposure epochs or establish a physical class.')
    paths = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        documentation='https://www.legacysurvey.org/dr11/description/',
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (marker,source)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in paths}),indent=2)+'\n')


if __name__=='__main__':
    main()
