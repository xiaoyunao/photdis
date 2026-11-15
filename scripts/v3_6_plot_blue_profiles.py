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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from photdis_figure_style import BLUE, GRAY, ORANGE, STYLE
from v3_6_blue_cutouts import anchors
from v3_6_plot_blue_colors import MARKER
from v3_6_plot_forward_validation import deliver
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


NAME = 'F03_blue_posteriors_and_scores'
CUTS = ROOT/'configs/v3_4_near_validation.json'
COHORTS = (('faint_blue_near','Faint blue foreground',ORANGE),('blue_2_to_5','Blue 2–5 kpc',BLUE))


def read_profiles(path,fields):
    with np.load(path,allow_pickle=False) as z:
        t = {k:z[k] for k in fields}
    if not (t['TYPE']=='PSF').all():
        raise ValueError('Native PSFs required for the fixed blue sample')
    return t


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
    t = read_profiles(path,values)
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


def mode_pair(edges,mass):
    """Locate two distinct PDF peaks using integrated local probability."""
    center = (edges[:-1]+edges[1:])*.5
    smooth = gaussian_filter1d(mass,1.5)
    peaks,_ = find_peaks(smooth,prominence=.006)
    weight = {a:mass[np.abs(center-center[a])<=.25].sum() for a in peaks}
    pairs = [(min(weight[a],weight[b]),a,b) for a in peaks for b in peaks
             if center[b]-center[a]>=1. and weight[a]>=.10 and weight[b]>=.10
             and smooth[a:b].min() <= .5*min(smooth[a],smooth[b])]
    return max(pairs,default=(0.,-1,-1))


def illustrate(path):
    rec = json.loads(MARKER.read_text())
    if sha(path)!=rec['outputs']['blue_objects.npz']:
        raise ValueError('The posterior input differs from the verified blue replay')
    keys = ('sample','ls_id','TYPE','strict','qrf_mu_edges','qrf_mu_mass','reference_kpc',
            'reference_lo_kpc','reference_hi_kpc','qrf_quantiles_kpc','band_state','qrf_qnear')
    t = read_profiles(path,keys)
    d,ref,edges,mass,state = (t[k] for k in ('qrf_quantiles_kpc','reference_kpc',
                                            'qrf_mu_edges','qrf_mu_mass','band_state'))
    np.testing.assert_allclose(mass.sum(axis=1),1.,rtol=0.,atol=1e-10)
    error = np.abs(5*np.log10(d[:,2]/ref))
    states = state[:,4:]
    success = (error<=.1)&(ref>=d[:,1])&(ref<=d[:,3])
    low_wise = (states<=2).all(axis=1)
    weak_w2 = (states[:,1]==3)&(t['sample']=='faint_blue_near')
    pairs = [mode_pair(u,m) for u,m in zip(edges,mass)]
    pair_strength = np.asarray([w for w,_,_ in pairs])

    def pick(eligible,value):
        return min(np.flatnonzero(eligible),key=lambda i:(value[i],t['ls_id'][i]))

    examples = (('Close Gaia agreement',pick(success,error)),
                ('Two separated QRF peaks',pick(pair_strength>0,-pair_strength)),
                ('Foreground failure · weak W2',pick(weak_w2,np.abs(error-np.median(error[weak_w2])))),
                ('Blue underestimate · WISE unusable',pick(low_wise&(t['sample']=='blue_2_to_5')&(d[:,2]<ref),d[:,2]/ref)))
    band = {0:'Missing',1:'Invalid',2:'Rejected',3:'Weak',4:'Significant',5:'Upper limit'}
    rows = []
    with plt.rc_context(STYLE):
        fig,axs = plt.subplots(2,2,figsize=(11.8,7.9),layout='constrained')
        for ax,(title,i) in zip(axs.flat,examples):
            u,m = edges[i],mass[i]
            dx = 10**((u-10)/5)
            lo,hi = t['reference_lo_kpc'][i],t['reference_hi_kpc'][i]
            ax.stairs(m/np.diff(u),dx,color=BLUE,label='QRF histogram')
            ax.axvline(d[i,2],color=BLUE,ls=':',label='QRF median')
            ax.axvspan(lo,hi,color=ORANGE,alpha=.25)
            ax.axvline(ref[i],color=ORANGE,ls='--',label='Gaia reference')
            ax.set(title=title,xlabel='Distance [kpc]',ylabel='PDF per modulus',xscale='log',
                   xlim=(min(dx[0],lo)*.96,max(dx[-1],hi)*1.04))
            w1,w2 = (band[k] for k in states[i])
            ax.text(.98,.98,f'W1 {w1.lower()} · W2 {w2.lower()}',transform=ax.transAxes,
                    ha='right',va='top',fontsize=10)
            _,p,q = pairs[i]
            for j,(a,b,v) in enumerate(zip(u[:-1],u[1:],m)):
                rows.append(dict(example=title,legacy_id=t['ls_id'][i],sample=t['sample'][i],strict=t['strict'][i],
                    W1=w1,W2=w2,gaia_kpc=ref[i],gaia_lo_kpc=lo,gaia_hi_kpc=hi,qrf_d50_kpc=d[i,2],
                    qrf_qnear=t['qrf_qnear'][i],mu_lo=a,mu_hi=b,mass=v,mode_peak=j in (p,q)))
        axs[0,0].legend(frameon=False,fontsize=9,loc='upper left')
        caption = ('Illustrative cases from the two frozen blue cohorts. The agreement case minimizes '
                   'the absolute modulus residual among rows within 0.1 mag whose central 68% '
                   'QRF interval contains the Gaia center. The two-peak case maximizes the smaller '
                   'mass within 0.25 mag of each peak, subject to separation of at least 1 mag and '
                   'a clear valley. The weak-W2 case is near the median absolute residual within '
                   'its W2 state. The last case has the strongest underestimate in the 2–5 kpc '
                   'cohort with both WISE bands unusable. QRF uses grz; WISE labels describe the '
                   'independently recorded photometry. These examples do not measure frequency.')
        deliver(fig,'F03_qrf_pdf_examples',rows,{str(MARKER.relative_to(ROOT)):sha(MARKER),
            str(path.relative_to(ROOT)):rec['outputs']['blue_objects.npz']},caption,
            figure_dir=FIGURE,table_dir=TABLE)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('--examples',action='store_true')
    args = parser.parse_args()
    (illustrate if args.examples else main)(args.data.resolve())
