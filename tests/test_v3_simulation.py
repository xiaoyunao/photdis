import numpy as np
from photdis.v3.simulation import empirical_library, draw_transfer


def test_continuous_transfer_keeps_missing_bands_and_shared_source_realization():
    n = 30
    bank = dict(identity=np.arange(n), flux_at_1kpc=np.full((n, 6), 100.),
        flux_error_at_1kpc=np.full((n, 6), 5.), band_mask=np.ones((n, 6), bool),
        reference_sigma=np.full(n, .1), class_label=np.arange(n)%3,
        corrected_feh=np.full(n, -.8), corrected_feh_valid=np.ones(n, bool))
    bank['band_mask'][:10, 2] = False; bank['flux_at_1kpc'][:, 4] = -1.
    library = empirical_library(bank)
    assert (library['flux'][:, 4] > 0).all()
    assert (library['flux'][:10, 2] == 0).all() and not library['mask'][:10, 2].any()
    contexts = dict(identity=np.arange(n), flux_error=np.full((n, 6), .1),
        transmission=np.full((n, 6), .98), band_mask=np.ones((n, 6), bool),
        EBV=np.full(n, .01), photometric_system=np.full(n, 'DECam_DR11'),
        RA=np.zeros(n), DEC=np.zeros(n), cell=np.zeros(n, int), field=np.zeros(n, int))
    out, truth, meta = draw_transfer(library, contexts, 'narrow_4kpc', n=128)
    assert (out['TYPE'] == 'PSF').all() and len(np.unique(out['identity'])) == 128
    assert np.isfinite(out['flux'][out['band_mask']]).all()
    assert np.isnan(out['flux'][~out['band_mask']]).all()
    assert len(np.unique(truth['mu'])) == 128 and .04 < truth['mu'].std() < .12
    assert np.isin(truth['template_identity'], bank['identity']).all()
    missing, _, _ = draw_transfer(library, contexts, 'missing_WISE', n=32)
    assert not missing['band_mask'][:, 4:].any()
