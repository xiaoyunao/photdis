"""Matched grz/griz distance controls with an explicit four-band interface.

Reuse the established histogram forest, never reinterpret a serialized v1 model.
Both variants receive [g,r,i,z] and share the very same Sobol draws and g shift.
"""
from functools import lru_cache
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from .baseline_v0 import HistogramQRF, summaries
from .baseline_v1 import quantiles_interpolated, shift_pdf
from .reference_v1_4 import optical_features, FEATURE_VERSION, FEATURE_BANDS


@lru_cache(maxsize=1)
def optical_sobol_bits():
    return (qmc.Sobol(4, scramble=True, seed=20261104).random_base2(10)
            * 2**32).astype(np.uint32)


def optical_noise(ids, draw, replicate=0):
    if not 0 <= draw < 1024 or replicate < 0:
        raise ValueError('draw must be in [0,1024); replicate must be nonnegative')
    with np.errstate(over='ignore'):
        x = (np.asarray(ids, dtype=np.uint64)[:, None]
             + np.arange(4, dtype=np.uint64) * np.uint64(0x9e3779b97f4a7c15)
             + np.uint64(replicate + 1) * np.uint64(0xd1b54a32d192ed03))
        x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
        x ^= x >> 31
    u = ((x >> 32).astype(np.uint32) ^ optical_sobol_bits()[draw]).astype(float)
    return ndtri((u + .5) / 2**32)


class OpticalQRF(HistogramQRF):
    """Fixed algorithm; explicit color mapping is saved with each model."""
    def __init__(self, variant, **kwargs):
        if variant not in ('grz', 'griz'):
            raise ValueError(variant)
        self.variant = variant
        self.feature_version = FEATURE_VERSION
        self.band_order = FEATURE_BANDS
        self.color_definitions = (('g-r', 'r-z') if variant == 'grz'
                                  else ('g-r', 'r-i', 'i-z'))
        super().__init__(len(self.color_definitions), **kwargs)

    def fit(self, mags, y, labels):
        mags = np.asarray(mags, float)
        optical_features(mags, self.variant)  # Validate the explicit input order.
        if not np.isfinite(mags).all():
            raise ValueError('Matched optical training requires all four measured bands')
        selected = mags[:, [0, 1, 3]] if self.variant == 'grz' else mags
        return super().fit(selected, y, labels)

    def predict(self, mags, errors, ids, draws=128, replicate=0):
        mags, errors, ids = np.asarray(mags, float), np.asarray(errors, float), np.asarray(ids)
        if (mags.ndim != 2 or mags.shape[1] != 4 or errors.shape != mags.shape
                or ids.ndim != 1 or len(ids) != len(mags)):
            raise ValueError('Expected aligned [g,r,i,z] magnitudes, errors and identities')
        if not np.isfinite(mags).all() or not np.isfinite(errors).all() or (errors < 0).any():
            raise ValueError('Invalid measured magnitudes or errors')
        if draws not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            raise ValueError('Sobol draw count must be a power of two up to1024')
        n = len(mags)
        pdf = np.zeros((n, len(self.centers)), float)
        classes = np.zeros((n, 4), float)
        for draw in range(draws):
            delta = optical_noise(ids, draw, replicate) * errors
            p, c = self.distribution(optical_features(mags + delta, self.variant))
            # M_effective=M-delta_g: use the same g noise as the noisy colors.
            pdf += shift_pdf(p, delta[:, 0], self.edges)
            classes += c
        pdf /= draws
        classes /= draws
        x = optical_features(mags, self.variant)
        dist = self.nn.kneighbors((x - self.mean) / self.scale, n_neighbors=10)[0][:, -1]
        supported = (dist <= self.radius) & np.all((x >= self.lo) & (x <= self.hi), axis=1)
        _, modes, ambiguous = summaries(pdf, self.centers)
        return dict(q_Meffective=quantiles_interpolated(pdf, self.edges),
                    modes_Meffective=modes, ambiguous=ambiguous, supported=supported,
                    support_distance=dist, class_weights=classes, pdf=pdf)
