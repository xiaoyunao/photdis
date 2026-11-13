"""External reference agreement, kept distinct from physical coverage."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.metrics import normal_reference_score, observed_parallax


def finite_pdf(grid, logpdf):
    grid, logpdf = np.asarray(grid), np.asarray(logpdf)
    if grid.shape != logpdf.shape or grid.ndim != 1 or not np.all(np.diff(grid) > 0):
        raise ValueError('Aligned increasing PDF grid required')
    if np.isnan(logpdf).any() or np.isposinf(logpdf).any():
        raise ValueError('Invalid PDF values')
    keep = np.flatnonzero(np.isfinite(logpdf))
    if len(keep) < 3 or not np.all(np.diff(keep) == 1):
        raise ValueError('Only zero-density endpoint extensions may be trimmed')
    return grid[keep],logpdf[keep]


def reference_center_agreement(quantiles, mu):
    q = np.asarray(quantiles)
    if q.shape != (5,) or not np.isfinite(q).all() or (np.diff(q) < 0).any():
        raise ValueError('Ordered five posterior quantiles required')
    delta = float(q[2]-mu)
    return dict(mu_reference=float(mu),mu_residual=delta,relative_residual=float(np.expm1(np.log(10)*delta/5)),
        severe=bool(abs(delta)>1),reference_agreement68=bool(q[1]<=mu<=q[3]),
        reference_agreement95=bool(q[0]<=mu<=q[4]))


def score_pdf(grid, logpdf, quantiles, probability, evidence, *, common_scale=None):
    g,lp = finite_pdf(grid,logpdf)
    q = np.asarray(quantiles); p = np.asarray(probability)
    reference_center_agreement(q,q[2])
    if p.shape != (3,6) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1)>1e-8:
        raise ValueError('Normalized class-distance probabilities required')
    result = dict(mu=q.tolist(),class_distance=p.tolist(),qnear=float(p[:,:3].sum()),
        width68=float(q[3]-q[1]),width95=float(q[4]-q[0]),references={})
    if evidence['distance_available']:
        distance,error = float(evidence['distance_kpc']),float(evidence['distance_error_kpc'])
        if not np.isfinite([distance,error]).all() or min(distance,error)<=0:
            raise ValueError('Invalid qualified distance observation')
        mu = 10+5*np.log10(distance); sigma = 5/np.log(10)*error/distance
        value = reference_center_agreement(q,mu)
        value.update(log_score=normal_reference_score(g,lp,mu,sigma),sigma_mu=float(sigma),
            precise=bool(evidence['distance_precise']),teacher='Gaia-trained AstroNN',score_unit='inverse_mag')
        result['references']['astronn'] = value
    for name in ['specdis','spdist']:
        if not evidence.get(name+'_valid',False):continue
        distance=float(evidence[name+'_distance']); sigma=float(evidence[name+'_sigma_mu'])
        if not np.isfinite([distance,sigma]).all() or min(distance,sigma)<=0:raise ValueError('Invalid qualified VAC distance')
        mu=10+5*np.log10(distance); value=reference_center_agreement(q,mu)
        value.update(log_score=normal_reference_score(g,lp,mu,sigma),sigma_mu=sigma,
            score_unit='inverse_mag',teacher=name,scope='Source-qualified teacher agreement, not independent absolute truth.')
        result['references'][name]=value
    if common_scale is not None:
        distance,error = common_scale
        if min(distance,error)<=0 or not np.isfinite(common_scale).all():
            raise ValueError('Positive finite common distance scale required')
        logmass = lp+np.log(trapezoid_weights(g)); logmass -= logsumexp(logmass)
        score = logsumexp(logmass+norm.logpdf(distance,10**((g-10)/5),error))
        value = reference_center_agreement(q,10+5*np.log10(distance))
        value.update(log_score=float(score),distance_kpc=float(distance),error_kpc=float(error),
            shared=True,score_unit='inverse_kpc',scope='Single-object marginal only; do not multiply across members.')
        result['references']['common_shared_scale'] = value
    if evidence['parallax_eligible']:
        result['gaia'] = observed_parallax(g,lp,float(evidence['p_corr']),float(evidence['gaia_parallax_error']))
        result['gaia']['tail'] = result['gaia']['pit']<.025 or result['gaia']['pit']>.975
    label = int(evidence['class_label'])
    if label not in (-1,0,1):
        raise ValueError('External class is reliable dwarf/giant or unknown')
    cp = p.sum(axis=1); result['class_probability'] = cp.tolist()
    if label >= 0:
        result['class'] = dict(label=label,predicted=int(cp.argmax()),probability=float(cp[label]),
            log_score=float(np.log(cp[label])) if cp[label]>0 else None,zero_probability=bool(cp[label]==0),
            brier=float(np.square(cp-np.eye(3)[label]).sum()))
    mws_label=int(evidence.get('mws_class_label',-1))
    if mws_label not in (-1,0,1,2):raise ValueError('Invalid independently qualified MWS class')
    if mws_label>=0:
        result['mws_class']=dict(label=mws_label,predicted=int(cp.argmax()),probability=float(cp[mws_label]),
            log_score=float(np.log(cp[mws_label])) if cp[mws_label]>0 else None,zero_probability=bool(cp[mws_label]==0),
            brier=float(np.square(cp-np.eye(3)[mws_label]).sum()))
    return result


def compare_score_levels(previous, selected):
    checks = []
    if set(previous['references']) != set(selected['references']) or ('gaia' in previous) != ('gaia' in selected):
        raise ValueError('Numerical levels require the same external channels')
    for name in selected['references']:
        delta = abs(selected['references'][name]['log_score']-previous['references'][name]['log_score'])
        checks.append(dict(channel=name,logscore_delta=float(delta),passed=bool(np.isfinite(delta) and delta<.02)))
    if 'gaia' in selected:
        delta = abs(selected['gaia']['log_score']-previous['gaia']['log_score'])
        pd = abs(selected['gaia']['pit']-previous['gaia']['pit'])
        checks.append(dict(channel='gaia',logscore_delta=float(delta),pit_delta=float(pd),
            passed=bool(np.isfinite([delta,pd]).all() and delta<.02 and pd<.003)))
    return dict(channels_checked=len(checks),checks=checks,passed=all(r['passed'] for r in checks))
