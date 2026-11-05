"""Read-only diagnostic statistics, separate from frozen prediction code."""
import hashlib
import numpy as np


CLASS_NAMES = {-1: 'unknown', 0: 'dwarf', 1: 'evolved', 2: 'other_transition'}


def distance_kpc(mu):
    return 10.0 ** ((np.asarray(mu) - 10) / 5)


def residual_stats(mu, ref):
    mu, ref = np.asarray(mu), np.asarray(ref)
    good = np.isfinite(ref) & np.isfinite(mu).all(axis=1)
    mu, ref = mu[good], ref[good]
    if not len(ref):
        return {'n': 0}
    delta = mu[:, 2] - ref
    bias = np.median(delta)
    return dict(n=len(ref), bias_mag=float(bias),
                nmad_mag=float(1.4826 * np.median(abs(delta-bias))),
                tail_1=float(np.mean(abs(delta)>1)), tail_count=int((abs(delta)>1).sum()),
                coverage68=float(np.mean((ref>=mu[:,1]) & (ref<=mu[:,3]))),
                coverage95=float(np.mean((ref>=mu[:,0]) & (ref<=mu[:,4]))),
                width68=float(np.median(mu[:,3]-mu[:,1])),
                width95=float(np.median(mu[:,4]-mu[:,0])),
                min_delta=float(delta.min()), max_delta=float(delta.max()),
                distance_ratio_median=float(np.median(10**(delta/5))))


def paired_stats(old, new, cell, repeats=2000, seed=20261105):
    old, new, cell = np.asarray(old), np.asarray(new), np.asarray(cell)
    good = np.isfinite(old) & np.isfinite(new)
    old, new, cell = old[good], new[good], cell[good]
    if not len(old):
        return {'n': 0}
    ob, nb = abs(old)>1, abs(new)>1
    cells, inv = np.unique(cell, return_inverse=True)
    counts = np.bincount(inv)
    delta_tail = nb.astype(float)-ob
    delta_error = abs(new)-abs(old)
    totals = np.column_stack([np.bincount(inv, weights=x) for x in [delta_tail, delta_error]])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cells), size=(repeats,len(cells)))
    values = totals[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
    ci = np.quantile(values,[.025,.975],axis=0) if len(cells)>1 else np.full((2,2),np.nan)
    return dict(n=len(old), cells=len(cells), old_bad=int(ob.sum()), new_bad=int(nb.sum()),
                rescued=int((ob & ~nb).sum()), harmed=int((~ob & nb).sum()),
                both_bad=int((ob & nb).sum()), both_good=int((~ob & ~nb).sum()),
                tail_difference=float(delta_tail.mean()), tail_ci95=ci[:,0].tolist(),
                mean_absolute_error_difference=float(delta_error.mean()), mae_ci95=ci[:,1].tolist(),
                median_absolute_error_difference=float(np.median(delta_error)), replicates=repeats)


def cell_area(cell):
    dec = (np.asarray(cell)%18)*10-90
    return np.deg2rad(10)*(np.sin(np.deg2rad(dec+10))-np.sin(np.deg2rad(dec)))*(180/np.pi)**2


def fixed_examples(identity, masks, per_group=2, seed=20261105):
    used=set();result={}
    for name, mask in masks.items():
        rows = np.flatnonzero(mask)
        order=sorted(rows,key=lambda i:hashlib.sha256(f'{seed}:{int(identity[i])}'.encode()).digest())
        chosen=[]
        for row in order:
            if int(identity[row]) not in used:
                used.add(int(identity[row]));chosen.append(int(row))
            if len(chosen)==per_group:break
        result[name]=chosen
    return result


def wise_joint(flux, ivar, nobs, transmission, wisemask, fracflux, weight):
    flux, ivar, weight = np.asarray(flux), np.asarray(ivar), np.asarray(weight)
    measured=(np.asarray(nobs)>0)&np.isfinite(flux)&np.isfinite(ivar)&(ivar>0)
    measured &= np.isfinite(transmission)&(np.asarray(transmission)>0)
    snr=np.full(len(flux),np.nan);snr[measured]=flux[measured]*np.sqrt(ivar[measured])
    # The failures are independent boolean axes, not sequential categories.
    low=snr<5; mask=np.asarray(wisemask)!=0; blend=~(np.asarray(fracflux)<1)
    total=float(weight.sum());rows=[]
    for low_value in [False, True]:
        for mask_value in [False, True]:
            for blend_value in [False, True]:
                take=measured&(low==low_value)&(mask==mask_value)&(blend==blend_value)
                rows.append(dict(low_snr=low_value,mask_failed=mask_value,fracflux_failed=blend_value,
                                 n=int(take.sum()),weight=float(weight[take].sum()),
                                 fraction=float(weight[take].sum()/total)))
    q=measured&(mask|blend)
    return dict(cells=rows,unmeasured_n=int((~measured).sum()),
                unmeasured_fraction=float(weight[~measured].sum()/total),
                low_snr_total_fraction=float(weight[measured&low].sum()/total),
                quality_failure_fraction=float(weight[q].sum()/total),
                low_snr_and_quality_fraction=float(weight[q&low].sum()/total),
                low_snr_given_quality=float(weight[q&low].sum()/weight[q].sum()) if q.any() else None,
                weighted_total=total)
