#!/usr/bin/env python3
"""Re-render distance plates from exported bins; no per-star reads or selection."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator
import v2_0_revision as r


def density(ax, rows, label, panel):
    selected = [row for row in rows if row['panel'] == label]
    if selected:
        # Distance edges are geometric; recover exact per-cell geometry from CSV.
        from matplotlib.collections import PolyCollection
        vertices = [[(float(v['xlo']),float(v['ylo'])),(float(v['xhi']),float(v['ylo'])),
                     (float(v['xhi']),float(v['yhi'])),(float(v['xlo']),float(v['yhi']))] for v in selected]
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_bad((0,0,0,0))
        im = PolyCollection(vertices, array=np.array([float(v['n']) for v in selected]),
                            cmap=cmap, edgecolors='none', rasterized=False)
        ax.add_collection(im)
    else:
        im = ax.pcolormesh(panel['x_range'], panel['y_range'], np.ma.masked_all((1,1)), cmap='Blues')
    return im


def main():
    r.configure()
    r.f.FIG=r.ROOT/'figures/v2_0_revision'
    r.f.TABLES=r.ROOT/'manifests/v2_0_revision'
    for path in sorted((r.f.TABLES/'figures').glob('03_distance_*.json')):
        record=json.loads(path.read_text()); name=record['name']
        csv_path=r.ROOT/record['csv']; csv_before=csv_path.read_bytes()
        with open(csv_path) as stream:rows=list(csv.DictReader(stream))
        r.f.begin();r.f.ROWS.extend(rows);r.f.PANELS.extend(record['panels'])
        fig=plt.figure(figsize=(17,10.5))
        outer=fig.add_gridspec(2,3,height_ratios=[5,1.8],hspace=.42,wspace=.48,
                              top=.87,bottom=.08,left=.065,right=.96)
        for j,source in enumerate(('Gaia','specdis','spdist')):
            upper=outer[0,j].subgridspec(2,2,height_ratios=[4,1],width_ratios=[1,.045],hspace=.045,wspace=.055)
            ax=fig.add_subplot(upper[0,0]);res=fig.add_subplot(upper[1,0],sharex=ax)
            cax=fig.add_subplot(upper[:,1]);hist=fig.add_subplot(outer[1,j])
            selection=record['clipping']['sources'][source];n=selection['retained_n']
            if not n:
                for a in (ax,res,cax,hist):a.set_axis_off()
                ax.text(.5,.5,'No eligible reference',ha='center',transform=ax.transAxes)
                ax.set_title(source+' · N=0');continue
            panels={p['panel']:p for p in record['panels']}
            d=panels[source+'_distance'];e=panels[source+'_Attached residual']
            im=density(ax,rows,source+'_distance',d);rim=density(res,rows,source+'_Attached residual',e)
            norm=LogNorm(1,max(2,np.ma.max(im.get_array()),np.ma.max(rim.get_array())))
            im.set_norm(norm);rim.set_norm(norm)
            cb=fig.colorbar(im,cax=cax,label='Objects/bin')
            if norm.vmax<10:
                cb.set_ticks(np.arange(1,int(norm.vmax)+1))
                cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
                cb.ax.yaxis.set_minor_locator(NullLocator())
            edges=d['x_range']
            ax.plot(edges,edges,'--',c=r.ORANGE,lw=1)
            ax.set(xscale='log',yscale='log',xlim=edges,ylim=d['y_range'],ylabel='Predicted distance [kpc]',
                   title=f'{source} · N={n:,}\n3σ clipped: {selection["clipped_n"]:,}')
            ax.tick_params(axis='x',which='both',labelbottom=False)
            ax.yaxis.set_major_locator(FixedLocator(np.geomspace(*edges,4)[1:]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.3g}'))
            ax.yaxis.set_minor_locator(NullLocator())
            res.set(ylim=(-100,100),yticks=[-100,0,100],xlabel='Reference distance [kpc]')
            res.set_ylabel('Residual [%]',fontsize=12);res.axhline(0,c=r.GRAY,lw=.8)
            res.xaxis.set_major_locator(FixedLocator(np.geomspace(*edges,4)))
            res.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.3g}'))
            res.xaxis.set_minor_locator(NullLocator())
            h=[v for v in rows if v['panel']==source+'_Central 2 percentage-point histogram']
            bins=np.r_[[float(v['lo']) for v in h],float(h[-1]['hi'])];counts=[int(v['n']) for v in h]
            hist.stairs(counts,bins,color=r.BLUE)
            hist.set(xlim=(-100,100),xlabel='Relative distance residual [%]',ylabel='Objects/bin',
                     title=f'Central 2 percentage-point histogram\nOutside ±100%: {n-sum(counts):,}/{n:,}')
            hist.axvline(0,c=r.GRAY,lw=.8);hist.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
            for v in rows:
                if v['panel']==source+'_spread' and v.get('median'):
                    x=np.sqrt(float(v['xlo'])*float(v['xhi']))
                    res.plot(x,float(v['median']),'o',ms=2,c='black')
                    res.vlines(x,float(v['p16']),float(v['p84']),color='black',lw=.6)
        group=name.removeprefix('03_distance_');group='giant' if group=='evolved' else group
        fig.suptitle(r.GROUP_NAMES[group],fontsize=17)
        r.f.save(fig,name,record['caption'],section=record['section'],reference=record['reference'])
        assert csv_path.read_bytes()==csv_before, 'Binned statistics changed during layout-only rendering'
        record.update(png_sha256=r.f.REGISTRY[name]['png_sha256'],pdf_sha256=r.f.REGISTRY[name]['pdf_sha256'],
                      layout_postprocess='Re-rendered exact exported cells and trend quantiles; compact low-count colorbar ticks; CSV byte-identical.')
        r.save_json(path,record)
    print('Re-rendered five plates from unchanged CSV bins')


if __name__=='__main__':main()
