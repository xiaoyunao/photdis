"""Explicit observation states and independent reference qualifications."""
import numpy as np

BANDS = ('G', 'R', 'I', 'Z', 'W1', 'W2')
ROLES = ('excluded', 'sed_train', 'source_calibration', 'hyperparameters',
         'seen_development', 'prospective')
# Missing, invalid, contaminated, legal weak/signed, significant, upper limit.
MISSING, INVALID, CONTAMINATED, WEAK, SIGNIFICANT, UPPER_LIMIT = range(6)


def specdis_qualification(distance, error, spectral_quality):
    """No binary flag or Gaia residual/SNR is an argument to this gate."""
    d, e = np.asarray(distance, float), np.asarray(error, float)
    good = np.asarray(spectral_quality, bool) & np.isfinite(d) & np.isfinite(e)
    good &= (d > 0) & (e > 0) & (e < d)
    sigma = np.full(d.shape, np.nan)
    sigma[good] = 5/np.log(10)*np.sqrt(np.log1p((e[good]/d[good])**2))
    return good, sigma


def primary_channel(specdis, spdist, gaia):
    """Availability hierarchy, not a fit to reference residuals. Codes 0/1/2/3."""
    return np.where(specdis, 1, np.where(spdist, 2, np.where(gaia, 3, 0))).astype('i1')


def observations(raw, system, optical_clean=None, i_state=None):
    """Keep raw signed nanomaggies; dust enters the forward model only.

    Unknown quality fails admission. Per-band metadata are retained, never filled
    with plausible measurements. Explicit censoring needs an actual limit column.
    """
    n = len(raw['TYPE'])
    system = np.broadcast_to(np.asarray(system), (n,))
    flux = np.full((n, 6), np.nan); error = flux.copy(); trans = flux.copy()
    state = np.full((n, 6), MISSING, 'i1')
    for j, b in enumerate(BANDS):
        keys = [p+'_'+b for p in ('FLUX', 'FLUX_IVAR', 'MW_TRANSMISSION', 'NOBS')]
        if not all(k in raw for k in keys):
            continue
        f, v, t, count = [np.asarray(raw[k]) for k in keys]
        flux[:, j], trans[:, j] = f, t
        exists = count > 0
        legal = exists & np.isfinite(f) & np.isfinite(v) & (v > 0)
        legal &= np.isfinite(t) & (t > 0) & (t <= 1)
        state[exists, j] = INVALID
        with np.errstate(divide='ignore', invalid='ignore'):
            error[legal, j] = 1/np.sqrt(v[legal])
        quality = np.zeros(n, bool)
        if b in ('G', 'R', 'Z') and optical_clean is not None:
            quality = np.asarray(optical_clean, bool).copy()
        elif b == 'I' and i_state is not None:
            quality = np.isin(i_state, [3, 4]) & (system == 'DECam_DR11')
        elif b.startswith('W'):
            if 'FRACFLUX_'+b in raw and 'WISEMASK_'+b in raw:
                ff = np.asarray(raw['FRACFLUX_'+b]); mask = np.asarray(raw['WISEMASK_'+b])
                quality = np.isfinite(ff) & (ff < 1) & (mask == 0)
        state[legal, j] = CONTAMINATED
        use = legal & quality
        state[use, j] = WEAK
        state[use & (f*np.sqrt(np.maximum(v, 0)) >= 5), j] = SIGNIFICANT
        # Northern I is not a DECam band, even if a caller accidentally supplies it.
        if b == 'I':
            state[system != 'DECam_DR11', j] = MISSING
    mask = np.isin(state, [WEAK, SIGNIFICANT, UPPER_LIMIT])
    psf = np.char.strip(np.asarray(raw['TYPE']).astype('U')) == 'PSF'
    return dict(flux=flux, flux_error=error, transmission=trans, band_state=state,
                band_mask=mask, psf=psf, photometric_system=system.astype('U16'))


def stable_rank(identity, salt=301111):
    """SplitMix64; stable across process/platform, all copies share assignment."""
    x = np.asarray(identity, dtype='u8') + np.uint64(salt)
    with np.errstate(over='ignore'):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


def assign_roles(split, cell, historical_fit, reserved):
    """Keep prior training/development separate, split calibration by sky block."""
    split, cell = np.asarray(split), np.asarray(cell)
    role = np.zeros(len(split), 'i1')
    role[split == 0] = 1
    role[(split == 1) & (stable_rank(cell) % 2 == 0)] = 2
    role[(split == 1) & (stable_rank(cell) % 2 == 1)] = 3
    role[split == 2] = 4
    # Previously fitted evidence cannot calibrate or evaluate the new model.
    role[np.asarray(historical_fit, bool)] = 1
    role[np.asarray(reserved, bool)] = 5
    return role
