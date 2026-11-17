#!/usr/bin/env python3
"""Produce Galactic Aitoff maps of source counts per effective imaging area."""
import json
import os

import healpy as hp
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import SHELLS, STYLE, plt, save
from v3_10_figures import check_templates
from final_catalogue_atlas import random_areas
from v3_final_scan import NSIDE, NPIX, OUT, SHARDS, accumulate, pixels

SHELL_LABELS = ('All distances', '0–1 kpc', '1–5 kpc', '5–10 kpc', '0–10 kpc')
PIXEL_AREA = hp.nside2pixarea(NSIDE, degrees=True)
MIN_AREA = .10*PIXEL_AREA
COLORMAP = LinearSegmentedColormap.from_list('stellar_surface_density',
                    ('#203b74', '#3c6ca8', '#81b8d5', '#f8efc0', '#f6bb63', '#d84932', '#7f151b'))


def read_sources(inputs):
    counts = np.zeros((4, 3, 1+len(SHELLS), NPIX), 'i8')
    total = 0
    for i in range(SHARDS):
        mark = OUT/'scan'/f'{i:02}.json'
        r = json.loads(mark.read_text())
        inputs[str(mark.relative_to(ROOT))] = sha(mark)
        name = mark.with_suffix('.npz').name
        counts += read_columns(verified(mark.with_name(name), r['outputs'][name]), ('counts',))['counts']
        total += r['parts']
    assert total == 28917
    np.testing.assert_array_equal(counts[:, :, 0].sum(axis=(0, 2)), (72289386, 43347612, 2281023))
    np.testing.assert_array_equal(counts[:, 1:, 1:4].sum(axis=2), counts[:, 1:, 4])
    assert counts[:, 1, 4].sum() == 31466411
    return counts


def read_area(inputs):
    results = {}
    old = random_areas(inputs)
    for side in ('north', 'south'):
        mark = OUT/'area'/side/'complete.json'
        r = json.loads(mark.read_text())
        inputs[str(mark.relative_to(ROOT))] = sha(mark)
        assert r['nside'] == NSIDE
        a = read_columns(verified(mark.with_name('area.npz'), r['outputs']['area.npz']),
                         ('effective_deg2',))['effective_deg2']
        if side == 'north':
            results['north'] = a[1, 1]
            pairs = ((a[1],old[3]),)
        else:
            results.update(south_high=a[1, 1], south_low=a[0, 1])
            pairs = ((a[0],old[2]),(a[1],old[0]))
        # Nested fine cells must exactly reconstruct the earlier verified coarse random areas.
        for fine, coarse in pairs:
            np.testing.assert_allclose(fine.reshape(7, coarse.shape[-1], -1).sum(axis=-1),
                                       coarse, rtol=0, atol=1e-11)
    return results


def reference_maps(inputs):
    mark = OUT/'science/complete.json'
    r = json.loads(mark.read_text())
    inputs[str(mark.relative_to(ROOT))] = sha(mark)
    name = 'plot_reference.npz'
    t = read_columns(verified(mark.with_name(name), r['outputs'][name]),
                     ('TYPE','ls_id','RA','DEC','specdis_kpc','spdist_kpc','group'))
    assert (t['TYPE']=='PSF').all() and len(np.unique(t['ls_id']))==len(t['ls_id'])
    pix = pixels(t['RA'], t['DEC'])
    q = np.where(np.isfinite(t['specdis_kpc']), t['specdis_kpc'], t['spdist_kpc'])
    result = np.zeros((3, 1+len(SHELLS), NPIX), 'i8')
    for i, mask in enumerate((np.isfinite(q), np.char.startswith(t['group'], 'dwarf_'), t['group']=='giant')):
        k = mask & np.isfinite(q)
        accumulate(result[i], pix[k], q[k])
    np.testing.assert_array_equal(result[:, 1:4].sum(axis=1), result[:, 4])
    return result


def plotting_grid():
    l = np.linspace(-np.pi, np.pi, 1441)
    b = np.linspace(-np.pi/2, np.pi/2, 721)
    lc, bc = np.meshgrid((l[:-1]+l[1:])/2, (b[:-1]+b[1:])/2)
    ra, dec = hp.Rotator(coord=('G','C'))(-np.rad2deg(lc.ravel()), np.rad2deg(bc.ravel()), lonlat=True)
    return l, b, pixels(ra, dec).reshape(lc.shape)


def atlas(dest, name, title, counts, areas, labels, nrows, ncols, grid, lin):
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8*ncols, 4.6*nrows),
        subplot_kw={'projection':'aitoff'}, layout='constrained')
    rates = np.divide(counts, areas, out=np.full_like(counts, np.nan, dtype='f8'), where=areas>=MIN_AREA)
    pop = rates[(rates>0)&np.isfinite(rates)]
    cap = max(2*lin, np.ceil(np.quantile(pop,.995))) if len(pop) else 2*lin
    # Allocate most color resolution to the occupied densities, retaining a
    # finite color for covered cells with zero sources.
    lin = max(lin, float(np.quantile(pop,.05))) if len(pop) else lin
    norm = SymLogNorm(linthresh=lin, linscale=.32, vmin=0, vmax=cap, base=10)
    decades = 10.**np.arange(np.floor(np.log10(lin)), np.ceil(np.log10(cap))+1)
    ticks = np.sort((decades[:,None]*[1,2,5]).ravel())
    ticks = [0,*ticks[(ticks>=lin)&(ticks<cap)]]
    rows = []
    l, b, pix = grid
    for j, (ax,label) in enumerate(zip(np.atleast_1d(axes).ravel(),labels)):
        # Values are defined on equal-area HEALPix cells. The fine display grid
        # evaluates the cell at each position without smoothing across the mask.
        ax.pcolormesh(l,b,np.ma.masked_invalid(rates[j,pix]),shading='flat',
                      cmap=COLORMAP,norm=norm,rasterized=True,zorder=1)
        ax.set_axisbelow(False)
        ax.grid(color='0.35', linestyle=':', linewidth=.6)
        ax.set_xticks(np.deg2rad([-120,-60,0,60,120]))
        ax.set_xticklabels(['120°','60°','0°','300°','240°'])
        ax.set_yticks(np.deg2rad([-60,-30,0,30,60]))
        ax.set_title(f'{label} · N = {counts[j].sum():,}')
        shown = areas[j]>=MIN_AREA
        rows.append(dict(view=label,n=int(counts[j].sum()),displayed=int(counts[j,shown].sum()),
                         omitted_small_area=int(counts[j,~shown].sum()),
                         zero_random_area_sources=int(counts[j,areas[j]==0].sum()),
                         displayed_area_deg2=float(areas[j,shown].sum()), linear_to=lin, upper_color=cap))
    for ax in np.atleast_1d(axes).ravel()[len(labels):]:
        ax.set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=COLORMAP),ax=np.atleast_1d(axes).ravel().tolist(),
                 shrink=.76,pad=.035,extend='max',ticks=ticks,format=plt.FormatStrFormatter('%g'),
                 label='Sources per effective deg²')
    fig.suptitle(title)
    fig.supxlabel('Galactic longitude and latitude · white: insufficient eligible imaging area',fontsize=11)
    save(fig,dest,name,rows)
    return rates


def main():
    require_compute()
    dest = OUT/'atlas'
    dest.mkdir(parents=True,exist_ok=True)
    inputs = {}
    counts = read_sources(inputs)
    areas = read_area(inputs)
    reference = reference_maps(inputs)
    by_domain = np.stack(list(areas.values()))
    total = by_domain.sum(axis=0)
    ref_area = areas['north']+areas['south_low']
    all_counts = counts.sum(axis=0)
    grid = plotting_grid()
    with plt.rc_context(STYLE):
        atlas(dest,'F21_full_density','Photdis DR11 · catalogue surface density',
              all_counts[:,0],np.broadcast_to(total,(3,NPIX)),
              ('Owned native PSFs','Full distance PDFs','Core science sample'),3,1,grid,1)
        take = [1,2,3,4]
        atlas(dest,'F22_pdf_density','Full distance PDFs · predicted median distance',
              all_counts[1,take],np.broadcast_to(total,(4,NPIX)),SHELL_LABELS[1:],2,2,grid,1)
        atlas(dest,'F23_core_density','Core science sample · predicted median distance',
              all_counts[2,take],np.broadcast_to(total,(4,NPIX)),SHELL_LABELS[1:],2,2,grid,1)
        atlas(dest,'F24_reference_density','Held-out spectra within the imaging footprint · reference distance',
              reference[:,take].reshape(12,NPIX),np.broadcast_to(ref_area,(12,NPIX)),
              [f'{name} · {shell}' for name in ('All spectra','Dwarf labels','Giant labels')
               for shell in SHELL_LABELS[1:]],3,4,grid,.3)
        native = np.stack((counts[3,1,0], counts[:3,1,0].sum(axis=0)))
        atlas(dest,'F25_native_density','Full distance PDFs by native photometric system',
              native,np.stack((areas['north'],areas['south_high']+areas['south_low'])),
              ('BASS/MzLS','DECam'),2,1,grid,1)
    atomic_npz(dest/'density_cells.npz',catalogue_counts=counts,reference_counts=reference,
               native_area_deg2=by_domain)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,style=check_templates(),
         nside=NSIDE,min_area_deg2=MIN_AREA,area_order=list(areas),area_deg2=total.sum(),
         output_shells=SHELL_LABELS,
         outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}))


if __name__ == '__main__':
    main()
