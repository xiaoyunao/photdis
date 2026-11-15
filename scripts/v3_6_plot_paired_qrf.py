#!/usr/bin/env python3
"""Show the fixed-model selection contrast on the paired historic test stars."""
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, ORANGE, STYLE
from v2_1_near import csv_rows
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


SOURCES = ('specdis','spdist')
SHELLS = (('2.0','2–5 kpc'),('5.0','5–10 kpc'))
CUTS = ('before_gates','primary','strict')
MODELS = (('expanded_A','Historic expanded_A',BLUE),('current_grz','Current grz QRF',ORANGE))
LOW_COUNT = 20


def load_comparison(marker):
    receipt = json.loads(marker.read_text())
    path = marker.with_name('performance.csv')
    if sha(path)!=receipt['outputs'][path.name]:
        raise ValueError('Paired QRF plot must use the receipt-verified performance table')
    with path.open() as stream:
        data = [r for r in csv.DictReader(stream)
                if r['source'] in SOURCES and r['lower_kpc'] in dict(SHELLS)
                and r['selection'] in CUTS and r['group']=='dwarf'
                and r['old_split']=='test' and r['current_qrf_training']=='False'
                and r['support']=='both_supported']
    view = {(r['source'],r['lower_kpc'],r['selection'],r['model']):r for r in data}
    if len(view)!=len(data):
        raise ValueError('Duplicate plotted QRF comparison')
    counts = {}
    for source in SOURCES:
        for shell,_ in SHELLS:
            for cut in CUTS:
                n = [int(view.get((source,shell,cut,model),{}).get('n',0)) for model,_,_ in MODELS]
                if n[0]!=n[1]:
                    raise ValueError('Plotted QRF models differ in their paired denominator')
                counts[source,shell,cut] = n[0]
    return view,counts,path


def render(view,counts,caption_method,name):
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(2,2,figsize=(12.4,8.4),sharey='row',layout='constrained')
        for row,(shell,heading) in enumerate(SHELLS):
            for col,(source,label) in enumerate(zip(SOURCES,('SpecDis','SPDist'))):
                ax = axs[row,col]
                n = np.asarray([counts[source,shell,cut] for cut in CUTS])
                x = np.arange(len(CUTS))
                for j,(model,title,color) in enumerate(MODELS):
                    values = [view.get((source,shell,cut,model),{}).get('median_relative') for cut in CUTS]
                    y = np.asarray([100*float(v) if v not in (None,'') else np.nan for v in values])
                    xp = x+(.04 if j else -.04)
                    ample = n>=LOW_COUNT
                    ax.plot(xp,np.where(ample,y,np.nan),'-o',color=color,ms=5,label=title)
                    tiny = (n>0)&~ample
                    ax.plot(xp[tiny],y[tiny],'o',color=color,markerfacecolor='white',ms=5)
                ticks = [f'{text}\nN = {k:,}' for text,k in zip(('Before cuts','Primary','Strict'),n)]
                ax.set(title=f'{label}  |  {heading}',xlim=(-.33,2.33),xticks=x,xticklabels=ticks)
                ax.axhline(0,color=GRAY,ls=':',linewidth=1.)
                ax.xaxis.grid(False)
        fig.supylabel('Median relative distance bias [%]')
        handles,labels = axs[0,0].get_legend_handles_labels()
        fig.suptitle(caption_method)
        fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False,fontsize=11)
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{name}.{ext}',dpi=300)
        plt.close(fig)


def main(mode):
    source,title = {
        'cache':('qrf_cache_comparison','Paired historical dwarfs | Original cached predictions'),
        'same_grz':('qrf_same_photometry','Paired historical dwarfs | Identical native grz'),
    }[mode]
    name = f'F02_paired_qrf_{mode}'
    marker = ROOT/'manifests/v3_6'/source/'complete.json'
    view,counts,path = load_comparison(marker)
    table = TABLE/f'{name}.csv'
    rows = []
    for (source,shell,cut),n in counts.items():
        for model,_,_ in MODELS:
            key = source,shell,cut,model
            rows.append(view.get(key,dict(source=source,lower_kpc=shell,selection=cut,model=model,n=n)))
    csv_rows(table,rows)
    render(view,counts,title,name)
    caption = ('Historic test dwarfs evaluated on fixed stars in each panel and selection, '
               'requiring support in both QRFs and absence from either training set. Both '
               'predictions use the same source-specific reference distance for each plotted '
               'star. A hollow symbol denotes fewer than 20 paired stars. The table gives '
               'uncut distance-error statistics for the plotted cells. '
               + ('Each model uses its original cached photometry.' if mode=='cache'
                  else 'Both frozen models use identical current native optical photometry.'))
    outputs = (table,FIGURE/f'{name}.png',FIGURE/f'{name}.pdf')
    (TABLE/f'{name}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (marker,path)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison',choices=('cache','same_grz'))
    main(parser.parse_args().comparison)
