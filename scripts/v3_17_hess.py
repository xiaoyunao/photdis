#!/usr/bin/env python3
"""Measure a fixed main-sequence excess against near and distant controls."""
import csv
import json
import os

import numpy as np
from astropy.io import fits
from matplotlib.colors import SymLogNorm

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_8_direct_figures import GRAY, STYLE, plt, save
from v3_10_figures import check_templates
from v3_11_layers import read_receipt
from v3_13_delivery import figure_check
from v3_16_gd1 import EDGES, LIMITS, NSEG, NSTEP, OFFSETS, track
from v3_16_sample import COLOR
from v3_17_members import table
from v3_17_sample import C, CONFIG, DEST, INPUT, NEW, center, read_track

ABS = np.linspace(-2,8,101)
MODULUS_CURVE = 'c2ed28ca8341f40e001946e33c6e2ac513786ac5cad23be516ad6eaec7a1d651'
BHB_POINTS = '5b286259a1874c857b90155210e595fd108ec92fb7aa34f0941a602d30c8ef58'
TITLES = ('r < 20','20 <= r < 21','21 <= r < 22','r < 21','r < 22')
TIERS = ((0,),(1,),(2,),(0,1),(0,1,2))


def read_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def refs(out,inputs):
    x = .5*(EDGES[1:]+EDGES[:-1])
    ds = np.sort(fits.getdata(verified(INPUT/'v4_Figure2_track_jarvis.fits',MODULUS_CURVE),1),order='phi1')
    d = 10**((np.interp(x,ds['phi1'],ds['r-Mr'])-10)/5)
    n = np.sort(table('v4_Table1.fits'),order='phi1')
    dr = np.interp(x,n['phi1'],n['distance'])
    p = ROOT/'manifests/v3_16'
    cf = read_csv(p/'cfht_reference.csv')
    sd = read_csv(p/'sdss_reference.csv')
    xx = [float(a['phi1']) for a in sd]
    sdss = np.interp(x,xx,[float(a['distance_kpc']) for a in sd])
    if not np.array_equal(x,[float(a['phi1']) for a in cf]):
        raise ValueError('Mismatched reference evaluation centers')
    n = read_track()
    k = [np.interp(x,n['phi1'],n[c]) for c in ('mu_phi1','mu_phi2','V_GSR')]
    rows = [dict(segment=i,phi1=l,desi_phi2=center(l,n),koposov_phi2=track(l),
                 desi_mu1=k[0][i],desi_mu2=k[1][i],desi_vgsr=k[2][i],
                 desi_curve_kpc=d[i],desi_node_linear_kpc=dr[i],cfht_kpc=cf[i]['distance_kpc'],
                 cfht_error_kpc=cf[i]['error_kpc'],sdss_interpolated_kpc=sdss[i]) for i,l in enumerate(x)]
    csv_rows(out/'references.csv',rows)
    b = fits.getdata(verified(INPUT/'v4_Figure2_BHBs.fits',BHB_POINTS),1)
    med = np.interp(b['phi1'],ds['phi1'],ds['r-Mr'])
    csv_rows(out/'bhb.csv',[dict(phi1=x,distance_kpc=10**((m-10)/5),distance_modulus=m,
                              desi_track_modulus=z,in_test_range=bool(EDGES[0]<=x<EDGES[-1]))
                            for x,m,z in zip(b['phi1'],b['r-Mr'],med)])
    inputs.update(cfht=sha(p/'cfht_reference.csv'),sdss=sha(p/'sdss_reference.csv'),
                  desi_modulus_curve=MODULUS_CURVE,bhb_points=BHB_POINTS)
    return ds


def weights(exposure,offsets,control_indices,at):
    good = np.zeros(len(offsets),bool)
    good[control_indices] = True
    good &= exposure>0
    neg,pos = good&(offsets<0),good&(offsets>0)
    if not neg.any() or not pos.any():
        return None
    x = np.average(offsets[neg],weights=exposure[neg])
    y = np.average(offsets[pos],weights=exposure[pos])
    w = np.zeros(len(offsets))
    w[neg] = (y-at)/(y-x)/exposure[neg].sum()
    w[pos] = (at-x)/(y-x)/exposure[pos].sum()
    return w


def test(h,exposure,offsets,ics,point):
    x = exposure[point]
    w = weights(exposure,offsets,ics,offsets[point])
    if w is None or x<=0:
        return None
    k = w*x
    contrast = h[point]-np.tensordot(k,h,axes=1)
    var = h[point]+np.tensordot(k*k,h,axes=1)
    return contrast,var


def by_cells(t,offsets,mu_curve,iso):
    lon,mag = t['phi1'],t['r0']
    mu = np.interp(lon,mu_curve['phi1'],mu_curve['r-Mr'])
    absolut = mag-mu
    iso = np.sort(iso,order='Mr')
    pred = np.interp(absolut,iso['Mr'],iso['g-r'])
    lo,hi = C['ridge_Mr_range']
    in_ms = (absolut>=lo)&(absolut<=hi)&(abs(t['color']-pred)<=C['ridge_half_width_mag'])
    base = ((t['block'].astype('i8')//NSTEP)*len(LIMITS)+t['tier'])*len(offsets)+t['ribbon']
    i = np.searchsorted(COLOR,t['color'],'right')-1
    j = np.searchsorted(ABS,absolut,'right')-1
    h = (base*(len(COLOR)-1)+i)*(len(ABS)-1)+j
    good = (j>=0)&(j<len(ABS)-1)
    shp = (NSEG,len(LIMITS),len(offsets))
    cells = (*shp,len(COLOR)-1,len(ABS)-1)
    counts = []
    for cohort in (np.ones(len(j),bool),t['qrf']):
        counts.append((
            np.bincount(h[good&cohort],minlength=np.prod(cells)).reshape(cells),
            np.bincount(base[in_ms&cohort],minlength=np.prod(shp)).reshape(shp),
            np.bincount(base[cohort],minlength=np.prod(shp)).reshape(shp)))
    return counts


def detect(nums,exposure,offsets,ics,blanks,tiers):
    v,var = 0.,0.
    zblank = np.full((len(tiers),len(blanks),2),np.nan)
    background = []
    for j,k in enumerate(tiers):
        n,a = nums[k],exposure[k]
        xc = test(n,a,offsets,ics,0)
        if xc is None:
            return None
        dv,dvar = xc
        v += dv
        var += dvar
        w = weights(a,offsets,ics,0.)
        background.append(np.dot(w*a[0],n))
        for l,b in enumerate(blanks):
            got = test(n,a,offsets,ics,b)
            if got is not None and got[1]>0:
                zblank[j,l] = got
    eligible = np.isfinite(zblank).all(axis=(0,2))
    points = np.asarray(blanks,dtype=int)[eligible]
    blank_data = zblank[:,eligible].sum(axis=0)
    zx = blank_data[:,0]/np.sqrt(blank_data[:,1])
    z = v/np.sqrt(var) if var>0 else np.nan
    high = float(abs(zx).max()) if len(zx) else np.nan
    adequate = (np.sum(offsets[points]<0)>=C['minimum_blank_per_side'] and
                np.sum(offsets[points]>0)>=C['minimum_blank_per_side'])
    passes = bool(adequate and z>=C['minimum_z'] and z>high)
    return dict(excess=float(v),standard_error=float(np.sqrt(var)),z=float(z),independent_blank=len(zx),
                max_abs_blank_z=high,signal=passes,background=float(sum(background))),points,zx


def render(out,name,H,iso,records):
    y = np.array([a for a in iso if 2.5<a['Mr']<7.])
    if not len(y):
        raise ValueError('No external sequence points in the Hess frame')
    limit = max(1.,float(np.abs(H).max()))
    norm = SymLogNorm(linthresh=max(.5,limit*.05),vmin=-limit,vmax=limit)
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,3,figsize=(15,10),layout='constrained',sharex=True,sharey=True)
        for i,group in enumerate(('Native PSF','QRF support')):
            for j in range(3):
                ax = axes[i,j]
                pic = ax.pcolormesh(COLOR,ABS,H[i,j].T,cmap='RdBu_r',norm=norm,rasterized=True)
                ax.plot(y[:,0] if y.ndim==2 else y['g-r'],y[:,1] if y.ndim==2 else y['Mr'],
                        color=GRAY,ls='--',lw=1.2)
                ax.set(title=f'{group} · {TITLES[j]}',xlim=(-.05,1.15),ylim=(7.8,1.5))
                if i==1:
                    ax.set_xlabel(r'Dereddened $g-r$ (mag)')
                if j==0:
                    ax.set_ylabel(r'$M_r$ at DESI distance (mag)')
        fig.colorbar(pic,ax=axes.ravel().tolist(),label='Excess PSFs per square degree per cell',pad=.02)
        save(fig,out,name,records)
    figure_check(out/(name+'.png'))


def run():
    folder = DEST/'sample'
    out = DEST/'hess'
    out.mkdir(parents=True,exist_ok=True)
    inputs = {'config':sha(CONFIG),'style':check_templates()}
    curve = refs(out,inputs)
    iso = table('v4_Figure3_empirical_isochrone.fits')
    inputs['ridge_sha256'] = sha(INPUT/'v4_Figure3_empirical_isochrone.fits')
    fields = ('TYPE','ls_id','phi1','color','r0','ribbon','tier','block','qrf')
    data = read_receipt(folder,'complete.json','area.npz',('legacy','legacy_good','desi','desi_good'),inputs)
    result,null,spatial,audit,stored = [],[],[],[],{}
    for frame,offsets,ics,blanks in [('legacy',OFFSETS,list(range(1,9)),[]),
                                     ('desi',NEW,list(range(1,5)),list(range(5,9)))]:
        t = read_receipt(folder,'complete.json',frame+'.npz',fields,inputs)
        assert all(t['TYPE']=='PSF')
        exp = data[frame].reshape(len(LIMITS),NSEG,NSTEP,len(offsets)).sum(axis=2).transpose(1,0,2)
        exp = np.where(data[frame+'_good'].transpose(1,0,2),exp,0)
        profiles = by_cells(t,offsets,curve,iso)
        plate = np.zeros((2,3,len(COLOR)-1,len(ABS)-1))
        for m,(h,nums,n_all) in enumerate(profiles):
            group = ('native','qrf')[m]
            den = np.zeros(3)
            for s in range(NSEG):
                a = exp[s]
                for k in range(3):
                    for j,x in enumerate(offsets):
                        spatial.append(dict(geometry=frame,cohort=group,segment=s,tier=TITLES[k],offset=x,
                                           native_area=a[k,j],n=int(n_all[s,k,j]),ridge_n=int(nums[s,k,j])))
                    d = test(h[s,k],a[k],offsets,ics,0)
                    if d is not None:
                        plate[m,k] += d[0]
                        den[k] += a[k,0]
                        stored[f'{frame}_{group}_s{s}_t{k}'] = np.asarray(d)
                    if frame=='desi':
                        left,right = [np.array([i for i in ics if offsets[i]*sign>0 and a[k,i]>0],dtype=int) for sign in (-1,1)]
                        if len(left)>=C['minimum_far_per_side'] and len(right)>=C['minimum_far_per_side']:
                            g = [nums[s,k,j].sum()/a[k,j].sum() for j in (left,right)]
                            z = [np.average(offsets[j],weights=a[k,j]) for j in (left,right)]
                            audit.append(dict(cohort=group,segment=s,tier=TITLES[k],negative_density=g[0],
                                                positive_density=g[1],gradient=(g[1]-g[0])/(z[1]-z[0])))
                for name,tiers in zip(TITLES,TIERS):
                    stats = detect(nums[s],a,offsets,ics,blanks,tiers)
                    if stats is None:
                        continue
                    row,points,z = stats
                    result.append(dict(geometry=frame,cohort=group,segment=s,tier=name,
                        observed=int(nums[s,list(tiers),0].sum()),**row))
                    null.extend(dict(cohort=group,segment=s,tier=name,offset=offsets[j],z=x)
                                for j,x in zip(points,z))
            plate[m] = np.divide(plate[m],den[:,None,None],out=plate[m],where=den[:,None,None]>0)
        render(out,'hess_'+frame,plate,iso,[r for r in result if r['geometry']==frame])
        stored['stack_'+frame] = plate
    csv_rows(out/'signal.csv',result)
    csv_rows(out/'blank.csv',null)
    csv_rows(out/'ribbons.csv',spatial)
    csv_rows(out/'gradient.csv',audit)
    atomic_npz(out/'hess.npz',color=COLOR,absolute=ABS,**stored)
    admitted = sorted(r['segment'] for r in result if (r['geometry'],r['cohort'],r['tier'],r['signal'])==
                      ('desi','native','r < 22',True))
    yes = sum(i in C['exploration_segments'] for i in admitted)>=C['minimum_exploration_pass']
    yes &= sum(i in C['validation_segments'] for i in admitted)>=C['minimum_validation_pass']
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,signal_segments=admitted,
        authorize_distance_stage=bool(yes),outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
