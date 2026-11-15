#!/usr/bin/env python3
"""Show both fixed blue distance-failure cohorts and their observed diagnostics."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, ORANGE, STYLE
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


SOURCE = ROOT/'manifests/v3_6/blue_replay'
NAME = 'F03_blue_distance_failures'
GROUPS = (('faint_blue_near','Faint blue foreground',ORANGE),
          ('blue_2_to_5','Blue 2–5 kpc references',BLUE))
VALUES = ('reference_kpc','qrf_median_kpc','distance_ratio','dereddened_g_r','observed_r',
          'M_G_reference','M_G_qrf','gaia_bp_rp','gaia_reduced_proper_motion_G',
          'gaia_pmra','gaia_pmdec','gaia_ruwe')


def main():
    source = SOURCE/'blue_objects.csv'
    marker = SOURCE/'complete.json'
    if sha(source)!=json.loads(marker.read_text())['outputs'][source.name]:
        raise ValueError('Blue-cohort table differs from the NERSC replay receipt')
    with source.open() as stream:
        data = list(csv.DictReader(stream))
    groups = []
    for name,label,color in GROUPS:
        rows = [r for r in data if r['sample']==name]
        metrics = {k:np.asarray([float(r[k]) for r in rows]) for k in VALUES}
        groups.append((label,color,rows,metrics))
    if [len(x[2]) for x in groups]!=[116,189]:
        raise ValueError('Unexpected frozen v3.5 blue counts')
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,2,figsize=(12.6,9),layout='constrained')
        a,b,c,d = axes.ravel()
        for label,color,rows,p in groups:
            mark = dict(s=21,color=color,alpha=.75,linewidths=0)
            a.scatter(p['reference_kpc'],p['qrf_median_kpc'],**mark,label=f'{label} (N={len(rows)})')
            b.scatter(p['dereddened_g_r'],p['M_G_reference'],**mark)
            c.scatter(p['observed_r'],p['distance_ratio'],**mark)
            d.scatter(p['gaia_bp_rp'],p['gaia_reduced_proper_motion_G'],**mark)
        a.plot([.04,9],[.04,9],ls='--',color=GRAY,lw=1.3)
        a.set(xscale='log',yscale='log',xlim=(.04,9),ylim=(.04,9),
              xlabel='Gaia reference distance [kpc]',ylabel='QRF median distance [kpc]',
              title='Distances on the same stars')
        a.legend(loc='lower right',frameon=False,fontsize=10)
        b.set(xlabel='Dereddened $g-r$ [mag]',ylabel='Legacy $M_g$ from Gaia [mag]',
              title='Reference color–magnitude diagram')
        b.invert_yaxis()
        c.axhline(1,ls='--',color=GRAY,lw=1.3)
        c.set(yscale='log',xlabel='Observed $r$ [mag]',ylabel='QRF median / Gaia distance',
              title='Residuals with both tails retained')
        d.set(xlabel='Gaia BP − RP [mag]',ylabel='Gaia reduced proper motion $H_G$ [mag]',
              title='Photometry and proper motion')
        d.invert_yaxis()
        for ext in ('png','pdf'):
            fig.savefig(FIGURE/f'{NAME}.{ext}',dpi=300)
        plt.close(fig)
    table = TABLE/f'{NAME}.csv'
    with table.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['ls_id','source_id','sample','strict',*VALUES],
                                extrasaction='ignore',lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
    summary = {}
    for name,(_,_,rows,p) in zip((g[0] for g in GROUPS),groups):
        summary[name] = dict(n=len(rows),strict=sum(r['strict']=='True' for r in rows),
            low_luminosity_locus=sum(r['low_luminosity_locus']=='True' for r in rows),
            archived_class_known=sum(int(r['archived_class'])>=0 for r in rows),
            old_anomaly_gaia_match=sum(r['old_anomaly_gaia_match']=='True' for r in rows),
            reference_median_kpc=float(np.median(p['reference_kpc'])),
            qrf_median_kpc=float(np.median(p['qrf_median_kpc'])),
            median_distance_ratio=float(np.median(p['distance_ratio'])),
            median_reference_Mg=float(np.median(p['M_G_reference'])),
            median_qrf_Mg=float(np.median(p['M_G_qrf'])))
    caption = ('Both cohorts come from the frozen v3.5 primary Gaia draw and keep every member. '
               'Gaia reference distances use the qualified zero-point-corrected parallaxes. '
               'The reference color–magnitude diagram uses dereddened Legacy grz photometry. '
               'Gaia reduced proper motion uses observed G and proper motion, with no parallax in '
               'the ordinate. A low-luminosity location is evidence for a faint stellar branch '
               'and cannot establish an individual spectral class.')
    outputs = (table,FIGURE/f'{NAME}.png',FIGURE/f'{NAME}.pdf')
    (TABLE/f'{NAME}.json').write_text(json.dumps(dict(caption=caption,dpi=300,groups=summary,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (marker,source)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs}),indent=2)+'\n')


if __name__=='__main__':
    main()
