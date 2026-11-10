"""Historical distance admission conditional on fixed non-parallax covariates."""
import numpy as np


def distance_fit_from_snr(snr, optical_error, clean, astrometry, specdis_valid):
    """Recompute v1.4 fit eligibility under a counterfactual observed SNR.

    Other source qualifications/errors, photometry, class and spatial flags stay
    fixed. This does not model spectroscopic targeting or their correlations.
    Arrays may have a final Monte Carlo draw dimension.
    """
    return clean & (optical_error < .03) & ((astrometry & (snr >= 20)) | specdis_valid)


def selection_upper(optical_error, clean, astrometry, specdis_valid):
    """Upper SNR bound for NOT being distance-fit eligible; empty is NaN."""
    enabled=clean & (optical_error < .03)
    return np.where(enabled & specdis_valid,np.nan,np.where(enabled & astrometry,20.,np.inf))


def draw_observed_parallax(pdf, edges, g, error, rng, repeats=2048):
    """Direct forward histogram-distance + Gaussian-noise draws, no SNR cut."""
    out=np.empty((len(g),repeats))
    for i in range(len(g)):
        cdf=np.cumsum(pdf[i]);cdf/=cdf[-1];cdf[-1]=1
        j=np.searchsorted(cdf,rng.random(repeats))
        absolute=edges[j]+np.diff(edges)[j]*rng.random(repeats)
        out[i]=10**(2-(g[i]-absolute)/5)+error[i]*rng.normal(size=repeats)
    return out
