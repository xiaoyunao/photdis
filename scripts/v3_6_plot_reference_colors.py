#!/usr/bin/env python3
"""Compare observable and absolute color-magnitude diagrams on the same stars."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.data import BANDS
from photdis_figure_style import STYLE
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_6_distance_evidence import CLASSES, OUT as EVIDENCE
from v3_6_plot_blue_colors import PLANES, label
from v3_6_plot_distance_atlas import SOURCE_LABELS
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import read_columns


OUT = ROOT/'data/v3_6/figures/reference_colors'
CUTS = ('supported','primary','strict')
CUT_LABELS = ('Before cuts','Primary','Strict')
CLASS_NAMES = ('all',*CLASSES,'unknown')
CMD = (
    ('observed','Observed CMD','Observed $r$ [mag]',(-.35,2.7),(13.,22.)),
    ('reference','Reference-distance CMD','$M_{r}$ from reference [mag]',(-.35,2.7),(-7.,18.)),
    ('prediction','QRF-distance CMD','$M_{r}$ from QRF [mag]',(-.35,2.7),(-7.,18.)),
)
COLOR_RANGE = {'G-R':(-.35,2.7),'R-Z':(-.35,2.2),'Z-W1':(-2.,4.5),
               'W1-W2':(-1.,2.0),'R-I':(-.4,1.6),'I-Z':(-.4,1.7)}
GRID = 56


def binned(x,y,residual,x_edges,y_edges):
    nx,ny = len(x_edges)-1,len(y_edges)-1
    valid = np.isfinite(x)&np.isfinite(y)
    i = np.searchsorted(x_edges,x,side='right')-1
    j = np.searchsorted(y_edges,y,side='right')-1
    draw = valid&(i>=0)&(i<nx)&(j>=0)&(j<ny)
    key = i[draw]*ny+j[draw]
    counts = np.bincount(key,minlength=nx*ny)
    stats = {'count':counts.reshape(nx,ny)}
    d = residual[draw]
    finite = np.isfinite(d)
    key,d = key[finite],d[finite]
    order = np.argsort(key,kind='stable')
    ids,d = key[order],d[order]
    cells,starts = np.unique(ids,return_index=True)
    fields = {k:np.full(nx*ny,np.nan) for k in ('bias','absolute','severe')}
    used = np.zeros(nx*ny,dtype=int)
    for cell,start,end in zip(cells,starts,np.r_[starts[1:],len(ids)]):
        value = d[start:end]
        fields['bias'][cell] = np.median(value)
        fields['absolute'][cell] = np.median(abs(value))
        fields['severe'][cell] = (abs(value)>1.).mean()
        used[cell] = len(value)
    stats.update({k:v.reshape(nx,ny) for k,v in fields.items()})
    stats['residual_count'] = used.reshape(nx,ny)
    return stats, dict(available=int(valid.sum()),outside_frame=int((valid&~draw).sum()),
                       finite_residual=int((valid&np.isfinite(residual)).sum()))


def plots(a,magnitude):
    with np.errstate(divide='ignore',invalid='ignore'):
        reference = 10+5*np.log10(a['reference_kpc'])
        qrf = 10+5*np.log10(a['qrf_quantiles_kpc'][:,2])
    residual = qrf-reference
    y = (a['observed_r'],magnitude[:,1]-reference,magnitude[:,1]-qrf)
    common = np.isfinite(residual)&np.isfinite(magnitude[:,1])
    color = magnitude[:,0]-magnitude[:,1]
    for (key,title,ylabel,xrange,yrange),value in zip(CMD,y):
        yield key,title,label(('G','R')),ylabel,np.where(common,color,np.nan),value,residual,xrange,yrange
    for x,y in PLANES:
        xlabel,ylabel = label(x),label(y)
        slug_x,slug_y = '-'.join(x),'-'.join(y)
        cx,cy = (magnitude[:,BANDS.index(u)]-magnitude[:,BANDS.index(v)] for u,v in (x,y))
        yield f'{slug_x}_vs_{slug_y}',f'{xlabel[:-6]} and {ylabel[:-6]}',xlabel,ylabel,cx,cy,residual, \
            COLOR_RANGE[slug_x],COLOR_RANGE[slug_y]


def prepare(a,magnitude):
    cell_rows,admission,display = [],[],{}
    all_rows = np.ones(len(a['ls_id']),bool)
    cls = a['consensus_class']
    groups = (all_rows,*[cls==i for i in range(len(CLASSES))],cls<0)
    for key,title,xlabel,ylabel,x,y,delta,xrange,yrange in plots(a,magnitude):
        edges = [np.linspace(*r,GRID+1) for r in (xrange,yrange)]
        for cut,select in zip(CUTS,(all_rows,a['primary'],a['strict'])):
            for name,group in zip(CLASS_NAMES,groups):
                use = select&group
                stats,counts = binned(x[use],y[use],delta[use],*edges)
                if counts['outside_frame']+stats['count'].sum()!=counts['available']:
                    raise ValueError('F05 bins do not reconcile with the number of eligible stars')
                scope = dict(plane=key,selection=cut,reference_class=name)
                admission.append(dict(**scope,total=int(use.sum()),**counts))
                for i,j in zip(*np.nonzero(stats['count'])):
                    cell_rows.append(dict(**scope,x_lo=edges[0][i],x_hi=edges[0][i+1],
                        y_lo=edges[1][j],y_hi=edges[1][j+1],**{k:v[i,j] for k,v in stats.items()}))
                if name=='all':
                    display[key,cut] = dict(data=stats,counts=counts,title=title,xlabel=xlabel,
                        ylabel=ylabel,edges=edges)
    state_rows = []
    for cut,select in zip(CUTS,(all_rows,a['primary'],a['strict'])):
        for name,group in zip(CLASS_NAMES,groups):
            for band in ('I','W1','W2'):
                values = a['band_state'][select&group,BANDS.index(band)]
                for i,tag in enumerate(('missing','invalid','contaminated','weak_or_signed','significant','upper_limit')):
                    state_rows.append(dict(selection=cut,reference_class=name,band=band,state=tag,
                                           rows=int((values==i).sum())))
    return cell_rows,admission,state_rows,display


def panel(ax,item,field,limits,cmap,invert,minimum=0):
    x,y = item['edges']
    image = item['data'][field].T
    valid = (item['data'][('count' if field=='count' else 'residual_count')].T>=minimum)&np.isfinite(image)
    norm = limits if isinstance(limits,LogNorm) else None
    kwargs = {'norm':norm} if norm else dict(vmin=limits[0],vmax=limits[1])
    m = ax.pcolormesh(x,y,np.ma.masked_where(~valid,image),cmap=cmap,rasterized=True,**kwargs)
    ax.set(xlabel=item['xlabel'],ylabel=item['ylabel'],xlim=(x[0],x[-1]),ylim=(y[0],y[-1]))
    if invert:
        ax.invert_yaxis()
    if field=='count':
        n,t = (item['counts'][k] for k in ('available','outside_frame'))
        ax.set_title(f"{item['title']}   |   N = {n:,}" + (f'   |   {t:,} off-frame' if t else ''))
    return m


def export(fig,dest,name):
    for extension in ('png','pdf'):
        fig.savefig(dest/f'{name}.{extension}',dpi=300)
    plt.close(fig)


def render(display,dest,source):
    keys = [k[0] for k in CMD]
    color_keys = [key for key,cut in display if key not in keys and cut=='supported']
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(3,3,figsize=(15.4,13.2),layout='constrained')
        upper = max(2,*[display[key,cut]['data']['count'].max() for key in keys for cut in CUTS])
        for row,cut,cut_label in zip(axs,CUTS,CUT_LABELS):
            for ax,key in zip(row,keys):
                m = panel(ax,display[key,cut],'count',LogNorm(1,upper),'viridis',True,1)
            row[0].annotate(cut_label,(-.22,.5),xycoords='axes fraction',rotation=90,ha='center',va='center')
        fig.colorbar(m,ax=axs,label='Matched stars per bin',shrink=.7)
        fig.suptitle(f'{source} | Observed and absolute color–magnitude diagrams')
        export(fig,dest,'cmd')

        fig,axs = plt.subplots(2,3,figsize=(15.1,9.2),layout='constrained')
        upper = max(2,*[display[key,'supported']['data']['count'].max() for key in color_keys])
        for ax,key in zip(axs.flat,color_keys):
            m = panel(ax,display[key,'supported'],'count',LogNorm(1,upper),'viridis',False,1)
        fig.colorbar(m,ax=axs,label='Matched stars per bin',shrink=.7)
        fig.suptitle(f'{source} | Native color planes before cuts')
        export(fig,dest,'colors')

        fig,axs = plt.subplots(2,3,figsize=(15.1,8.5),layout='constrained')
        metrics = (('bias',(-1,1),'RdBu_r','Median distance modulus residual [mag]'),
                   ('absolute',(0,2),'magma','Median absolute modulus residual [mag]'),
                   ('severe',(0,1),'magma','Fraction with $|\u0394\u03bc|>1$ mag'))
        for col,(field,limits,cmap,title) in enumerate(metrics):
            for row,key in enumerate(('observed','reference')):
                m = panel(axs[row,col],display[key,'supported'],field,limits,cmap,True,3)
                axs[row,col].set_title(display[key,'supported']['title'])
            fig.colorbar(m,ax=axs[:,col],label=title,shrink=.8)
        fig.suptitle(f'{source} | Distance errors before cuts (three or more stars per bin)')
        export(fig,dest,'residuals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    source = parser.parse_args().source
    require_compute()
    p,e = PHOTO/'complete.json',EVIDENCE/'complete.json'
    photo,evidence = [json.loads(x.read_text()) for x in (p,e)]
    if photo['inputs'][str(e.relative_to(ROOT))]!=sha(e) or source not in evidence['rows']:
        raise ValueError('Reference source does not belong to the frozen photometry union')
    path = EVIDENCE/f'{source}.npz'
    a = load(verified(path,evidence['outputs'][path.name]))
    t = read_columns(verified(PHOTO/'photometry.npz',photo['outputs']['photometry.npz']),
                     ('ls_id','catalog_row','TYPE','magnitude','band_state'))
    order = np.searchsorted(t['catalog_row'],a['catalog_row'])
    np.testing.assert_array_equal(t['ls_id'][order],a['ls_id'])
    np.testing.assert_array_equal(t['band_state'][order],a['band_state'])
    if not (a['TYPE']=='PSF').all() or not (t['TYPE'][order]=='PSF').all():
        raise ValueError('F05 requires native PSF')
    dest = OUT/source
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        raise FileExistsError(dest/'complete.json')
    cells,counts,states,display = prepare(a,t['magnitude'][order])
    for kind,data in (('cells',cells),('counts',counts),('band_states',states)):
        csv_rows(dest/f'{kind}.csv',data)
    render(display,dest,SOURCE_LABELS.get(source,source))
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),source=source,dpi=300,
        inputs={str(x.relative_to(ROOT)):sha(x) for x in (Path(__file__),e,p)},
        caption=('CMD columns use the same positive-reference and QRF-distance sample. Color panels '
                 'require only their displayed measured bands. Optional-band states, class and cut '
                 'counts, off-frame counts and all occupied plotted bins are exported. Distance '
                 'modulus residuals are QRF minus reference. Residual and reference CMD coordinates '
                 'share the reference distance, inducing correlated errors. A narrow predicted '
                 'sequence by itself gives no validation of distances.'),
        outputs={x.name:sha(x) for x in dest.iterdir() if x.name!='complete.json'}))


if __name__=='__main__':
    main()
