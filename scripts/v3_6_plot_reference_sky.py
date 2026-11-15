#!/usr/bin/env python3
"""Show the projected parent and the per-source reference support on measured area."""
import argparse
import json
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
import numpy as np

from photdis.stage1 import galactic, require_compute
from photdis_figure_style import STYLE
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_spatial import DELIVERY
from v3_6_catalog_observables import OUT as OBSERVABLES
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_plot_shell_retention import FIGURE, INPUT, TABLE


OUT = ROOT/'data/v3_6/figures/reference_sky'
CUTS = ('supported','primary','strict')


def positions(mask,nside):
    a,b = hp.pix2ang(nside,np.flatnonzero(mask),nest=True,lonlat=True)
    c,d = galactic(a,b)
    return {'Equatorial':(a,b),'Galactic':(c,d)}


def metrics(p,nside):
    """Use all qualified rows outside current QRF training in each cut."""
    d = p['reference_kpc']
    q = p['qrf_quantiles_kpc'][:,2]
    ref = p['high_quality']&~p['qrf_training']&np.isfinite(d)&(d>0)&np.isfinite(q)&(q>0)
    pixel = hp.ang2pix(nside,p['RA'],p['DEC'],nest=True,lonlat=True)
    residual = np.divide(q,d,out=np.full(len(d),np.nan),where=ref)-1
    cube = np.full((len(CUTS),4,hp.nside2npix(nside)),np.nan)
    for k,cut in enumerate(CUTS):
        use = ref if cut=='supported' else ref&p[cut]
        cells,diff = pixel[use],residual[use]
        order = np.argsort(cells)
        breaks = np.r_[0,1+np.flatnonzero(np.diff(cells[order])),len(order)]
        cube[k,0] = 0
        for a,b in zip(breaks[:-1],breaks[1:]):
            if a==b:
                continue
            key = cells[order[a]]
            x = diff[order[a:b]]
            mid = np.median(x)
            severe = np.mean(np.abs(5*np.log10(1+x))>1)
            cube[k,:,key] = len(x),mid,1.4826*np.median(np.abs(x-mid)),severe
    return cube


def map_panel(ax,coordinates,values,norm,cmap):
    lon,lat = coordinates
    x,y = np.deg2rad((180-lon)%360-180),np.deg2rad(lat)
    use = np.isfinite(values)
    if isinstance(norm,LogNorm):
        use &= values>0
    ax.scatter(x,y,c='#e0e0e0',s=3,linewidths=0,rasterized=True)
    ax.scatter(x[use],y[use],c=values[use],s=5,linewidths=0,rasterized=True,norm=norm,cmap=cmap)
    ax.set_longitude_grid(90)
    ax.set_latitude_grid(45)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{(-round(np.rad2deg(v)))%360}°'))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.2)


def panels(name,data,coordinates,names,norms,cmaps,row_labels,figure_dir=OUT):
    outputs = []
    nrows,ncols = data.shape[:2]
    for system,pos in coordinates.items():
        with plt.rc_context(STYLE):
            fig,axs = plt.subplots(nrows,ncols,figsize=(4.15*ncols,2.55*nrows+1.15),
                squeeze=False,subplot_kw={'projection':'aitoff'},layout='constrained')
            fig.suptitle(f'{name.replace("_"," ").capitalize()} · {system} coordinates')
            for j,(label,(norm,cmap)) in enumerate(zip(names,zip(norms,cmaps))):
                for i,ax in enumerate(axs[:,j]):
                    map_panel(ax,pos,data[i,j],norm,cmap)
                    if j==0:
                        ax.set_ylabel(row_labels[i],fontsize=12)
                    if i==0:
                        ax.set_title(label,fontsize=14)
                color = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axs[:,j],
                                     orientation='horizontal',shrink=.72,pad=.055,extend='both')
                if isinstance(norm,LogNorm):
                    ticks = np.geomspace(norm.vmin,norm.vmax,3)
                    color.set_ticks(ticks,labels=[f'{v:,.0f}' if v>=10 else f'{v:.2g}' for v in ticks])
                color.ax.tick_params(labelsize=10)
                color.minorticks_off()
            for ext in ('png','pdf'):
                path = figure_dir/f'F06_{name}_{system.lower()}.{ext}'
                fig.savefig(path,dpi=300)
                outputs.append(path)
            plt.close(fig)
    return outputs


def log_span(values):
    pos = values[np.isfinite(values)&(values>0)]
    if not len(pos):
        return LogNorm(1.,2.)
    a,b = np.quantile(pos,[.02,.98])
    return LogNorm(a,max(a*1.001,b))


def catalog_density(delivery,spatial,figure_dir,table_dir):
    production = json.loads(delivery.read_text())
    verified(spatial,production['outputs']['spatial.npz'])
    with np.load(spatial,allow_pickle=False) as z:
        area = z['area_deg2']
    mask = area>0
    nside = hp.npix2nside(len(area))
    coords = positions(mask,nside)
    obs = OBSERVABLES/'complete.json'
    guide = json.loads(obs.read_text())
    with np.load(verified(OBSERVABLES/'observables.npz',guide['outputs']['observables.npz']),allow_pickle=False) as z:
        all_counts = z['sky']
    np.testing.assert_array_equal(all_counts.sum(axis=1),[production['totals'][k] for k in ('rows','primary','strict')])
    parent = (all_counts[:,mask]/area[mask]).T
    figure_dir.mkdir(parents=True,exist_ok=True)
    table_dir.mkdir(parents=True,exist_ok=True)
    span = log_span(parent)
    output = panels('catalog_density',parent.T[None,:,:],coords,['Supported','Primary','Strict'],
                    [span]*3,['viridis']*3,['Stars per square degree'],figure_dir)
    tab = table_dir/'F06_catalog_density.csv'
    density = np.divide(all_counts,area,out=np.full(all_counts.shape,np.nan),where=mask)
    rows = [dict(healpix_nested=int(pixel),selection=cut,area_deg2=area[pixel],
                 stars=int(all_counts[i,pixel]),stars_per_deg2=density[i,pixel])
            for pixel in np.flatnonzero(mask|np.any(all_counts>0,axis=0))
            for i,cut in enumerate(CUTS)]
    csv_rows(tab,rows)
    output.append(tab)
    marker = table_dir/'F06_catalog_density.json'
    save_json(marker,dict(dpi=300,nside=nside,area_deg2=float(area.sum()),
        catalog_counts=dict(zip(CUTS,all_counts.sum(axis=1).astype(int).tolist())),
        zero_area_counts=dict(zip(CUTS,all_counts[:,~mask].sum(axis=1).astype(int).tolist())),
        density_scale_stars_per_deg2=[float(span.vmin),float(span.vmax)],
        inputs={str(path.relative_to(ROOT)):sha(path) for path in (delivery,obs,Path(__file__))},
        caption='Counts per square degree of imaging-random area in each nested HEALPix cell. '
                'All supported native PSFs and both cuts use the same denominator and color scale. '
                'Colorbar extensions mark density clipping. Cells without random area appear only '
                'in the count table. The positions are pixel centers.',
        outputs={str(path.relative_to(ROOT)):sha(path) for path in output}))
    return area,mask,nside,coords,all_counts,output+[marker]


def main():
    require_compute()
    delivery = DELIVERY/'complete.json'
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    area,mask,nside,coords,all_counts,output = catalog_density(delivery,DELIVERY/'spatial.npz',OUT,OUT)
    evidence = EVIDENCE/'complete.json'
    inputs = json.loads(evidence.read_text())
    summary = []
    for name in inputs['rows']:
        path = EVIDENCE/f'{name}.npz'
        keys = ('ls_id','TYPE','RA','DEC','reference_kpc','qrf_quantiles_kpc','high_quality','qrf_training','primary','strict')
        with np.load(verified(path,inputs['outputs'][path.name]),allow_pickle=False) as z:
            p = {k:z[k] for k in keys}
        if not (p['TYPE']=='PSF').all():
            raise ValueError('The reference map requires verified native PSFs')
        result = metrics(p,nside)
        rows = []
        for i,cut in enumerate(CUTS):
            for pix,(n,bias,nmad,severe) in enumerate(result[i].T):
                if area[pix]>0 or n:
                    rows.append(dict(source=name,selection=cut,healpix_nested=pix,area_deg2=area[pix],
                        catalog_rows=int(all_counts[i,pix]),reference_rows=int(n),median_relative=bias,
                        nmad_relative=nmad,severe_fraction=severe))
            summary.append(dict(source=name,selection=cut,reference_count=int(result[i,0].sum()),
                                reference_in_area=int(result[i,0,mask].sum()),
                                area_with_reference_deg2=float(area[result[i,0]>0].sum())))
        tab = OUT/f'F06_{name}_support.csv'
        csv_rows(tab,rows)
        output.append(tab)
        show = result[:, :, mask].copy()
        show[:,0] /= area[mask]
        show[:,1:] = np.where(result[:,0,mask][:,None,:]>=10,show[:,1:],np.nan)
        spans = [log_span(show[:,0]),Normalize(-.45,.45),Normalize(0,.45),Normalize(0,1)]
        output += panels(name,show,coords,['References per deg²','Relative bias','Relative NMAD','Severe fraction'],
                         spans,['viridis','coolwarm','magma','magma'],
                         ['Supported','Primary','Strict'])
    tab = OUT/'F06_reference_coverage.csv'
    csv_rows(tab,summary)
    output.append(tab)
    save_json(OUT/'complete.json',dict(dpi=300,nside=nside,inputs={str(p.relative_to(ROOT)):sha(p)
        for p in (delivery,OBSERVABLES/'complete.json',evidence,Path(__file__))},
        caption='Positive imaging-random area sets the displayed footprint. Reference maps show native high-quality matches outside current QRF training per source. Sky cells with fewer than ten references appear in density and in the full tables. Error colors use cells with at least ten rows. Uncovered reference rows remain in the tables. Area with reference sums the random area of cells containing at least one qualified row.',
        outputs={p.name:sha(p) for p in output}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-catalog',type=Path,metavar='SPATIAL_NPZ',
                        help='Publish only the catalog density from the compact spatial tables')
    args = parser.parse_args()
    if args.publish_catalog:
        catalog_density(INPUT/'complete.json',args.publish_catalog,FIGURE,TABLE)
    else:
        main()
