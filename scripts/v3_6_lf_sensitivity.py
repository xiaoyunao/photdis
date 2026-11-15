#!/usr/bin/env python3
"""Evaluate luminosity and spatial prior alternatives on one stratified DR11 draw."""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import fitsio
import joblib
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.stage1 import galactic, require_compute
from photdis.v3.data import stable_rank
from photdis.v3.inference import kernel_loglike
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_0_native import RAW, native_payload
from v3_0_response import observation
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified
from v3_4_near_development import DISTANCE_EDGES, MU
from v3_5_dr11_inventory import CONFIG
from v3_5_dr11_validate import join_parts, take_rows
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_lf_priors import empirical_weights, giant_luminosity_weights, radial_profiles
from v3_6_vac_distances import read_columns


OUT = ROOT/'data/v3_6/lf_sensitivity'
CAP = 12
SEED = 360723
NAMES = ('nominal','empirical_galactic','rgb_disk_rich','rgb_halo_rich','more_faint_other')
BOUNDS = 10+5*np.log10(DISTANCE_EDGES[1:-1])
GRID = np.union1d(MU,BOUNDS)
_STATE = None


def draw(data):
    """Stratify by the existing observable cells, gate and coarse Galactic sky."""
    l,b = galactic(data['RA'],data['DEC'])
    sky = ((l%360)//90).astype('i4')+4*(b>=0)
    gate = data['primary'].astype('i4')+data['strict'].astype('i4')
    cell = np.ravel_multi_index((sky,gate,data['stratum']),(8,3,180))
    rank = stable_rank(data['ls_id'],SEED)
    order = np.lexsort((data['ls_id'],rank,cell))
    limits = np.r_[0,np.flatnonzero(np.diff(cell[order]))+1,len(order)]
    ix,weights = [],[]
    for a,b in zip(limits[:-1],limits[1:]):
        n = min(CAP,b-a)
        ix.extend(order[a:a+n])
        weights.extend(np.full(n,(b-a)/n))
    result = take_rows(data,np.asarray(ix,dtype='i8'))
    result.update(weight=np.asarray(weights),sampling_cell=cell[ix])
    np.testing.assert_allclose(result['weight'].sum(),len(cell),rtol=0,atol=1e-7)
    for k in ('primary','strict'):
        np.testing.assert_allclose(result['weight'][result[k]].sum(),data[k].sum(),rtol=0,atol=1e-7)
    return result


def samples(receipt):
    frozen = []
    fields = ('catalog_row','ls_id','identity','source_id','RA','DEC','primary','strict',
              'stratum','observed_r','dereddened_g_r','band_state','parent_row','sweep_number',
              'part_number','part_row','qrf_bins')
    names = []
    for key,expected in receipt['child_receipts'].items():
        marker = verified(ROOT/key,expected)
        child = json.loads(marker.read_text())
        part = verified(marker.with_name('rows.npz'),child['outputs']['rows.npz'])
        frozen.append(read_columns(part,fields))
        names.extend(child['sweeps'])
    return draw(join_parts(frozen)),np.asarray(names)


def raw_photo(sample,names,root):
    def read(i):
        j = np.flatnonzero(sample['sweep_number']==i)
        rows = sample['parent_row'][j]
        order = np.argsort(rows,kind='stable')
        with fitsio.FITS(str(root/names[i])) as f:
            available = set(f[1].get_colnames())
            raw = f[1].read(columns=[k for k in RAW if k in available],rows=rows[order])
        data = native_payload(raw[np.argsort(order)],'DECam_DR11')
        np.testing.assert_array_equal(data['ls_id'],sample['ls_id'][j])
        np.testing.assert_array_equal(data['band_state'],sample['band_state'][j])
        if not (data['TYPE']=='PSF').all():
            raise ValueError('A drawn source lacks native PSF')
        keys = ('flux','flux_error','transmission','band_mask','EBV','photometric_system','RA','DEC')
        return ({k:data[k] for k in keys},j)
    with ThreadPoolExecutor(max_workers=min(12,int(os.getenv('SLURM_CPUS_PER_TASK','1')))) as pool:
        tables,slots = zip(*pool.map(read,np.unique(sample['sweep_number'])))
    return take_rows(join_parts(tables),np.argsort(np.concatenate(slots)))


def component_curves(log_kernel,weights):
    with np.errstate(divide='ignore'):
        return logsumexp(np.log(weights)[:,None]+log_kernel,axis=0)


def normalise(log_joint):
    pdf = np.exp(log_joint-np.max(log_joint))
    cdf = cumulative_trapezoid(pdf,GRID,axis=1,initial=0.)
    z = cdf[:,-1].sum()
    shell_cum = cdf[:,np.searchsorted(GRID,BOUNDS)]
    shell = np.diff(np.column_stack([np.zeros(len(pdf)),shell_cum,cdf[:,-1]]),axis=1)/z
    return shell,(logsumexp(log_joint,axis=0)-np.max(log_joint)-np.log(z)).astype('f4')


def forecast(i):
    photo,model,base,rgb_disk,rgb_halo,other_faint,class_mass,lon,lat = _STATE
    loglike,_ = kernel_loglike(model,observation(photo,i),GRID)
    binary = model.binary_reference_fraction
    log_kernel = np.logaddexp(loglike[0]+np.log1p(-binary),loglike[1]+np.log(binary))
    empirical = np.stack([component_curves(log_kernel,base[:,j]) for j in range(3)])
    disk,halo = radial_profiles(GRID,lon[i],lat[i])
    mass = np.log(class_mass)[:,None]
    mix = np.stack([.98*disk+.02*halo,.5*disk+.5*halo,.98*disk+.02*halo])
    galactic = empirical+mass+np.log(np.maximum(mix,1e-300))
    curve_disk = component_curves(log_kernel,rgb_disk[:,1])
    curve_halo = component_curves(log_kernel,rgb_halo[:,1])
    def changed_giants(halo_weight):
        joint = galactic.copy()
        joint[1] = mass[1]+np.logaddexp(curve_disk+np.log(np.maximum((1-halo_weight)*disk,1e-300)),
                                       curve_halo+np.log(np.maximum(halo_weight*halo,1e-300)))
        return joint
    faint = changed_giants(.5)
    faint[2] += component_curves(log_kernel,other_faint)-empirical[2]
    out = [normalise(joint) for joint in (empirical+mass+norm.logpdf(GRID,15.,5.),
                                          galactic,changed_giants(.2),changed_giants(.8),faint)]
    return np.stack([r[0] for r in out]),np.stack([r[1] for r in out])


def main():
    global _STATE
    require_compute()
    source = PROJECTION/'complete.json'
    receipt = json.loads(source.read_text())
    sample,names = samples(receipt)
    cfg = json.loads(CONFIG.read_text())
    photo = raw_photo(sample,names,Path(cfg['sweep_root']))
    entry = read_frozen()['models']['nominal']
    model_file = verified(ROOT/entry['path'],entry['sha256'])
    model = inference_model(joblib.load(model_file))
    base = empirical_weights(model)
    rgb_disk,disk_support = giant_luminosity_weights(model,.27,-.72,.22)
    rgb_halo,halo_support = giant_luminosity_weights(model,.32,-1.46,.30)
    abs_g_at_8 = 10**((12.5-8.)/2.5)
    faint = (model.means[:,0]>0)&(model.means[:,0]<abs_g_at_8)
    other_faint = base[:,2]*np.where(faint,2.,1.)
    other_faint /= other_faint.sum()
    class_mass = model.weights@model.classes
    lon,lat = galactic(sample['RA'],sample['DEC'])
    _STATE = photo,model,base,rgb_disk,rgb_halo,other_faint,class_mass,lon,lat
    ncpu = int(os.getenv('SLURM_CPUS_PER_TASK','16'))
    print(f'Observed DR11 LF draw: {len(lon)} native PSFs',flush=True)
    with ProcessPoolExecutor(max_workers=ncpu,mp_context=mp.get_context('fork')) as pool:
        joint,curves = zip(*pool.map(forecast,range(len(lon)),chunksize=12))
    joint = np.stack(joint)
    np.testing.assert_allclose(joint.sum(axis=(2,3)),1.,rtol=0,atol=1e-12)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    atomic_npz(OUT/'predictions.npz',**sample,TYPE=np.full(len(lon),'PSF'),
               grid_mu=GRID,scenario=np.asarray(NAMES),class_shell=joint,
               marginal_log_density=np.stack(curves),galactic_l=lon,galactic_b=lat)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),n=len(lon),
        scenarios=NAMES,class_prior=class_mass.tolist(),rgb_disk_support=disk_support,
        rgb_halo_support=halo_support,faint_other_base_mass=float(base[faint,2].sum()),
        rgb_disk=dict(slope=.27,feh_mean=-.72,feh_sigma=.22),
        rgb_halo=dict(slope=.32,feh_mean=-1.46,feh_sigma=.30),
        giant_halo_fractions=[.5,.2,.8,.5],non_giant_halo_fraction=.02,halo_taper_kpc=100.,
        inputs={str(x.relative_to(ROOT)):sha(x) for x in
                (source,model_file,FREEZE,CONFIG,Path(__file__),ROOT/'scripts/v3_6_lf_priors.py')},
        outputs={'predictions.npz':sha(OUT/'predictions.npz')}))


if __name__=='__main__':
    main()
