#!/usr/bin/env python3
"""Render the observed selection, distance validation and radial sensitivity."""
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from v2_1_near import ROOT, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import OUT as FORECAST
from v3_4_near_validate import OUT as EVIDENCE
from v3_4_near_validation_inputs import OUT as INPUTS, receipt_payload

OUT = ROOT/'data/v3_4/near_development/validation/figures'
COLORS = {'primary':'#0072B2', 'strict':'#D55E00'}
LABELS = {'primary':'Primary selection', 'strict':'Strict selection'}


def save(fig, name):
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def distance_figure(fresh, predicted, evidence):
    rows = fresh['prediction_row']
    d = evidence['distance_kpc']
    q = predicted['qrf_quantiles_kpc'][rows, 2]
    use = np.isfinite(d) & predicted['primary'][rows]
    strict = predicted['strict'][rows]
    fig, (ax, rat) = plt.subplots(2, 1, figsize=(6.2,7), sharex=True, layout='constrained',
                                gridspec_kw={'height_ratios':[2,1]})
    for tag, sample in [('primary', use & ~strict), ('strict', use & strict)]:
        label = 'Primary only' if tag == 'primary' else LABELS[tag]
        ax.scatter(d[sample], q[sample], c=COLORS[tag], s=8, alpha=.45, label=label)
        rat.scatter(d[sample], 5*np.log10(q[sample]/d[sample]), c=COLORS[tag], s=8, alpha=.45)
    lo, hi = np.min(np.r_[d[use],q[use]]), np.max(np.r_[d[use],q[use]])
    ax.plot([lo,hi], [lo,hi], c='.35', lw=1, ls='--')
    ax.set(xscale='log', yscale='log', ylabel='QRF median distance (kpc)',
           title='Photometric distances against fresh Gaia geometry')
    ax.legend(loc='upper left', frameon=False)
    rat.axhline(0, c='.35', lw=1, ls='--')
    rat.set(xlabel='Gaia geometric distance (kpc)', ylabel='Distance modulus residual (mag)')
    save(fig, 'distance_residual')


def selection_figures(application, photo, predicted):
    rows = application['prediction_row']
    mags, _ = optical_magnitudes({k:v[rows] for k,v in photo.items()})
    r = application['observed_r']
    gr, rz = mags[:,0]-mags[:,1], mags[:,1]-mags[:,2]
    masks = {'envelope':~predicted['primary'][rows], 'primary':predicted['primary'][rows] & ~predicted['strict'][rows],
             'strict':predicted['strict'][rows]}
    colors = dict(COLORS, envelope='#BBBBBB')
    labels = dict(LABELS, envelope='Other PSF sources', primary='Primary only')
    fig, (a,b) = plt.subplots(1, 2, figsize=(10,4.6), layout='constrained')
    for k, m in masks.items():
        style = dict(s=5,c=colors[k],alpha=.32,label=labels[k],rasterized=True)
        a.scatter(gr[m], r[m], **style)
        b.scatter(gr[m], rz[m], **style)
    a.invert_yaxis()
    a.set(xlabel='Dereddened g − r (mag)', ylabel='Observed r (mag)', title='Color and magnitude')
    b.set(xlabel='Dereddened g − r (mag)', ylabel='Dereddened r − z (mag)', title='Optical color space')
    b.legend(frameon=False, loc='best', markerscale=2)
    save(fig, 'observable_selection')

    fig, (a,b) = plt.subplots(2,1,subplot_kw={'projection':'mollweide'},figsize=(9,6),layout='constrained')
    ra = np.deg2rad((180.-application['RA']+180.)%360.-180.)
    dec = np.deg2rad(application['DEC'])
    for ax, cut in [(a,'primary'),(b,'strict')]:
        ax.scatter(ra,dec,c='.8',s=2,alpha=.45,rasterized=True)
        m = predicted[cut][rows]
        ax.scatter(ra[m],dec[m],c=COLORS[cut],s=3,alpha=.5,rasterized=True)
        ax.grid(alpha=.25)
        ax.set_title(LABELS[cut])
        ax.set_xticklabels([f'{h}h' for h in (22,20,18,16,14,12,10,8,6,4,2)])
        ax.set_xlabel('Right ascension')
        ax.set_ylabel('Declination')
    save(fig, 'sky_selection')


def radial_figure(values):
    fig, (a,b) = plt.subplots(1,2,figsize=(10.4,4.1),layout='constrained')
    x = np.arange(3)
    for j, cut in enumerate(('primary','strict')):
        rows = [r for r in values if r['cohort']=='application' and r['selection']==cut
                and r['group']=='all' and float(r['upper_kpc'])<=10]
        mass = np.array([float(r['model_total']) for r in rows])
        excluded = np.array([float(r['model_excluding_confirmed_far']) for r in rows])
        lower, upper = [np.array([float(r[key])/float(r['denominator']) for r in rows])
                        for key in ('geometry_lower','geometry_upper')]
        centers = x+(.19 if j else -.19)
        a.bar(centers,mass/mass.sum(),width=.35,color=COLORS[cut],alpha=.8,label=LABELS[cut])
        if excluded.sum():
            a.scatter(centers,excluded/excluded.sum(),color='black',s=16,marker='_',zorder=3,
                      label='Excluding secure far sources' if j==0 else None)
        b.vlines(centers,lower,upper,color=COLORS[cut],lw=4,label=LABELS[cut])
        b.scatter(centers,[float(r['model_total'])/float(r['denominator']) for r in rows],
                  facecolor='white',edgecolors=COLORS[cut],zorder=3)
    for ax in (a,b):
        ax.set_xticks(x,['0–2','2–5','5–10'])
        ax.set_xlabel('Distance range (kpc)')
        ax.legend(frameon=False,fontsize=8)
    a.set(ylabel='Fraction of predicted near mass',title='Sensitivity to the photometric selection')
    b.set(ylabel='Fraction of selected sources',title='Gaia distance bounds and QRF expectation')
    save(fig, 'radial_sensitivity')


def main():
    require_compute()
    inputs = {}
    photo = receipt_payload(INPUTS, 'photometry.npz', inputs)
    application = receipt_payload(INPUTS, 'application.npz', inputs)
    fresh = receipt_payload(INPUTS, 'fresh_gaia.npz', inputs)
    predicted = receipt_payload(FORECAST, 'forecasts.npz', inputs)
    marker = EVIDENCE/'complete.json'
    outputs = json.loads(marker.read_text())['outputs']
    evidence = load(verified(EVIDENCE/'fresh_gaia.npz', outputs['fresh_gaia.npz']))
    np.testing.assert_array_equal(evidence['identity'],fresh['identity'])
    path = verified(EVIDENCE/'distributions.csv', outputs['distributions.csv'])
    with path.open() as f:
        distributions = list(csv.DictReader(f))
    OUT.mkdir(parents=True,exist_ok=False)
    distance_figure(fresh,predicted,evidence)
    selection_figures(application,photo,predicted)
    radial_figure(distributions)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),evidence=sha(marker), inputs=inputs,
             outputs={p.name:sha(p) for p in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
