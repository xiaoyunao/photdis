#!/usr/bin/env python3
"""Show native QRF histograms and the nominal-selection scores of both cohorts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, ORANGE, STYLE
from v3_6_blue_cutouts import anchors
from v3_6_plot_blue_colors import MARKER
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


NAME = 'F03_blue_posteriors_and_scores'
CUTS = ROOT/'configs/v3_4_near_validation.json'
COHORTS = (('faint_blue_near','Faint blue foreground',ORANGE),('blue_2_to_5','Blue 2–5 kpc',BLUE))


def main(path):
    rec = json.loads(MARKER.read_text())
    if sha(path)!=rec['outputs']['blue_objects.npz']:
        raise ValueError('The posterior input differs from the verified blue replay')
    source = MARKER.with_name('blue_objects.csv')
    if sha(source)!=rec['outputs'][source.name]:
        raise ValueError('The object input differs from the verified blue replay')
    with source.open() as f:
        cases = anchors(list(csv.DictReader(f)))
    values = ('sample','ls_id','TYPE','strict','qrf_mu_edges','qrf_mu_mass','reference_kpc',
              'reference_lo_kpc','reference_hi_kpc','qrf_quantiles_kpc',
              'dwarf_selection_score','nominal_qnear','qrf_qnear')
    with np.load(path,allow_pickle=False) as z:
        t = {k:z[k] for k in values}
    if not (t['TYPE']=='PSF').all():
        raise ValueError('Native PSFs required for the fixed blue sample')
    cuts = json.loads(CUTS.read_text())
    ids = {str(k):i for i,k in enumerate(t['ls_id'])}
    selected = [(label,ids[row['ls_id']]) for label,row in cases]
    rows = []
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(2,3,figsize=(15.4,8.0),layout='constrained')
        labels = ('Historical foreground','Typical foreground','Largest underestimate','Typical 2–5 kpc case')
        ax_profile = (axs[0,0],axs[0,1],axs[1,0],axs[1,1])
        for ax,title,(_,j) in zip(ax_profile,labels,selected):
            edges = t['qrf_mu_edges'][j]
            mass = t['qrf_mu_mass'][j]
            if not np.isclose(mass.sum(),1.) or (np.diff(edges)<=0).any():
                raise ValueError('Original PDF is invalid')
            d = 10**((edges-10)/5)
            ax.stairs(mass/np.diff(edges),d,color=BLUE,lw=1.5)
            ax.axvline(t['qrf_quantiles_kpc'][j,2],color=BLUE,ls=':',alpha=.9)
            ax.axvspan(t['reference_lo_kpc'][j],t['reference_hi_kpc'][j],color=ORANGE,alpha=.3)
            ax.axvline(t['reference_kpc'][j],color=ORANGE,ls='--',label='Gaia reference')
            ax.set(title=title,xscale='log',xlabel='Distance [kpc]',ylabel='PDF per modulus',
                   xlim=(min(d[0],t['reference_lo_kpc'][j])*.92,max(d[-1],t['reference_hi_kpc'][j])*1.08))
            for a,b,v in zip(edges[:-1],edges[1:],mass):
                rows.append(dict(panel='QRF histogram',legacy_id=t['ls_id'][j],sample=t['sample'][j],
                    strict=t['strict'][j],mu_lo=a,mu_hi=b,mass=v,gaia_kpc=t['reference_kpc'][j],
                    gaia_lo_kpc=t['reference_lo_kpc'][j],gaia_hi_kpc=t['reference_hi_kpc'][j]))
        axs[0,0].legend(frameon=False,fontsize=9)
        ax_scores,ax_near = axs[:,2]
        for sample,title,color in COHORTS:
            use = t['sample']==sample
            opts = dict(c=color,s=18,alpha=.69,linewidths=0,label=title)
            ax_scores.scatter(t['dwarf_selection_score'][use],t['nominal_qnear'][use],**opts)
            ax_near.scatter(t['nominal_qnear'][use],t['qrf_qnear'][use],**opts)
        for cut,style in (('primary',':'),('strict','--')):
            v = cuts[cut]
            ax_scores.axvline(v['minimum_dwarf_score'],color=GRAY,ls=style,lw=1)
            ax_scores.axhline(v['minimum_near_probability'],color=GRAY,ls=style,lw=1)
        ax_near.plot([0,1],[0,1],color=GRAY,ls='--',lw=1)
        ax_scores.set(title='Photometric selection',xlabel='Dwarf classifier score',
                      ylabel='Nominal mass below 10 kpc',xlim=(.785,1.01),ylim=(.885,1.008))
        ax_near.set(title='Distance assignment',xlabel='Nominal mass below 10 kpc',
                    ylabel='QRF mass below 10 kpc',xlim=(.885,1.007),ylim=(0,1.025))
        ax_near.legend(frameon=False,fontsize=8.5,loc='lower left')
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)
    for j,k in enumerate(t['ls_id']):
        rows.append(dict(panel='Selection and distance probability',legacy_id=k,sample=t['sample'][j],
            strict=t['strict'][j],dwarf_score=t['dwarf_selection_score'][j],
            nominal_near=t['nominal_qnear'][j],qrf_near=t['qrf_qnear'][j]))
    table = TABLE/f'{NAME}.csv'
    with table.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    caption = ('The same four stars as the coadd plate and both full frozen anomaly samples. '
               'Histogram heights are densities per distance modulus and plotted against '
               'logarithmic distance. Every stored QRF histogram bin is retained. The '
               'orange intervals use the high-S/N Gaia parallax interval. Dotted and dashed '
               'cuts indicate primary and strict sample selection. The nominal score panels '
               'include every admitted blue case.')
    outputs = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        native_table_sha256=sha(path),inputs={str(p.relative_to(ROOT)):sha(p) for p in (MARKER,source,CUTS)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    main(parser.parse_args().data)
