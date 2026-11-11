"""Continuous independent-SED transfer experiments with explicit truth limitations."""
import numpy as np
from scipy.stats import truncnorm


SCENARIOS = {
    'reference_like': dict(class_weights=[.75, .05, .20]),
    'dwarf_dominated': dict(class_weights=[.94, .01, .05]),
    'evolved_rich': dict(class_weights=[.35, .45, .20]),
    'metal_poor': dict(class_weights=[.75, .05, .20], feh_range=[-5., -1.5]),
    'middle_metallicity': dict(class_weights=[.75, .05, .20], feh_range=[-1.5, -.5]),
    'class_distance_correlated': dict(class_weights=[.55, .20, .25], correlated=True),
    'narrow_4kpc': dict(class_weights=[.75, .05, .20], narrow=True),
    'binary_brightening': dict(class_weights=[.75, .05, .20], binary_fraction=.4),
    'missing_WISE': dict(class_weights=[.75, .05, .20], missing_wise=True),
    'weak_WISE': dict(class_weights=[.75, .05, .20], weak_wise=True),
    'dust_offset': dict(class_weights=[.75, .05, .20], dust_offset=.006),
    'band_offset': dict(class_weights=[.75, .05, .20], band_offset=[.01, -.01, .015, -.01, .02, -.02]),
}


def empirical_library(bank, seed=302501):
    """One shared source-level flux/teacher realization, not new stars per repeat."""
    rng = np.random.default_rng(seed)
    mean = bank['flux_at_1kpc']; sigma = bank['flux_error_at_1kpc']; mask = bank['band_mask']
    sed = np.zeros_like(mean)
    legal = mask & np.isfinite(mean) & np.isfinite(sigma) & (sigma > 0)
    # Weak signed flux remains likelihood evidence; its latent nonnegative flux is
    # sampled under an explicit flat-positive prior, not silently set to zero.
    sed[legal] = truncnorm.rvs(-mean[legal]/sigma[legal], np.inf,
                               loc=mean[legal], scale=sigma[legal], random_state=rng)
    reference_shift = rng.normal(size=len(sed))*bank['reference_sigma']
    sed *= 10**(.4*reference_shift[:, None])
    return dict(flux=sed, mask=legal, reference_shift=reference_shift,
                identity=bank['identity'], class_label=bank['class_label'],
                feh=bank['corrected_feh'], feh_valid=bank['corrected_feh_valid'])


def draw_transfer(library, contexts, scenario, n=128, seed=302502, first_identity=0, known_distance=False):
    """Observed-catalog experiment, with distances continuous before selection."""
    settings = SCENARIOS[scenario]; rng = np.random.default_rng(seed)
    populations = []
    for c in range(3):
        valid = (library['class_label'] == c) & library['mask'][:, [0, 1, 3]].all(axis=1)
        if 'feh_range' in settings:
            lo, hi = settings['feh_range']
            valid &= library['feh_valid'] & (library['feh'] >= lo) & (library['feh'] < hi)
        populations.append(np.flatnonzero(valid))
    class_weights = np.asarray(settings['class_weights'], float)
    absent = np.array([not len(p) for p in populations])
    # An absent empirical class cannot be manufactured. Record the renormalized
    # design so the restricted FeH experiment is not called all-population truth.
    class_weights[absent] = 0
    if class_weights.sum() == 0: raise ValueError('No empirical SED in scenario')
    class_weights /= class_weights.sum()
    output = []; truth = []; attempts = 0; accepted_before_cap = 0
    while sum(len(x['flux']) for x in output) < n:
        size = max(4*n, 512); attempts += size
        if attempts > 200000: raise RuntimeError('Simulation selection too inefficient; inspect design')
        label = rng.choice(3, size=size, p=class_weights)
        source = np.empty(size, int)
        for c in range(3):
            use = label == c
            if use.any(): source[use] = rng.choice(populations[c], size=use.sum())
        context = rng.integers(len(contexts['identity']), size=size)
        # Three overlapping continuous components; never the reporting-bin nodes.
        component = rng.choice(3, size=size, p=[.45, .4, .15])
        mu = rng.normal(np.array([11., 13., 16.5])[component], np.array([1.5, 1., 2.])[component])
        if known_distance:
            from .inference import DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
            component = rng.integers(len(DISTANCE_BASIS_MEAN), size=size)
            mu = rng.normal(DISTANCE_BASIS_MEAN[component], DISTANCE_BASIS_SIGMA[component])
        if settings.get('correlated'): mu += np.where(label == 1, 3., np.where(label == 2, 1., 0.))
        if settings.get('narrow'): mu = rng.normal(10+5*np.log10(4.), .08, size)
        binary = rng.random(size) < settings.get('binary_fraction', .1)
        brightness = np.where(binary, 2., 1.)
        trans = contexts['transmission'][context].copy(); error = contexts['flux_error'][context].copy()
        mask = library['mask'][source] & contexts['band_mask'][context]
        ebv = contexts['EBV'][context]
        coefficient = np.divide(-2.5*np.log10(trans), ebv[:, None], out=np.zeros_like(trans), where=ebv[:, None] > 0)
        shift = np.asarray(settings.get('band_offset', np.zeros(6)))[None]+coefficient*settings.get('dust_offset', 0.)
        true_flux = library['flux'][source]*10**(-.4*(mu[:, None]-10))*brightness[:, None]*trans*10**(-.4*shift)
        if settings.get('missing_wise'): mask[:, 4:] = False
        if settings.get('weak_wise'):
            error[:, 4:] = np.maximum(error[:, 4:], np.maximum(true_flux[:, 4:]/1.5, 1e-6))
        measured = true_flux+rng.normal(size=true_flux.shape)*np.where(mask, error, 0.)
        entry = mask[:, [0, 1, 3]].all(axis=1) & (measured[:, [0, 1, 3]] >= 5*error[:, [0, 1, 3]]).all(axis=1)
        if not entry.any(): continue
        accepted_before_cap += int(entry.sum())
        state = np.where(mask, np.where(measured/error >= 5, 4, 3), 0).astype('i1')
        measured[~mask] = np.nan; error[~mask] = np.nan
        values = dict(flux=measured, flux_error=error, transmission=trans, band_mask=mask,
            band_state=state, EBV=ebv, photometric_system=contexts['photometric_system'][context],
            RA=contexts['RA'][context], DEC=contexts['DEC'][context], cell=contexts['cell'][context],
            field=contexts['field'][context], TYPE=np.full(size, 'PSF'), design_weight=np.ones(size))
        output.append({k: v[entry] for k, v in values.items()})
        truth.append(dict(mu=mu[entry], class_label=label[entry], binary=binary[entry],
            template_identity=library['identity'][source][entry], context_identity=contexts['identity'][context][entry],
            corrected_feh=library['feh'][source][entry], corrected_feh_valid=library['feh_valid'][source][entry]))
    result = {k: np.concatenate([x[k] for x in output])[:n] for k in output[0]}
    truth = {k: np.concatenate([x[k] for x in truth])[:n] for k in truth[0]}
    result['identity'] = -(900000000000000000+first_identity+np.arange(n, dtype='i8'))
    result['ls_id'] = result['identity'].copy()
    return result, truth, dict(attempted=attempts, accepted_before_cap=accepted_before_cap,
        generated_class_weights=class_weights.tolist(), absent_classes=np.flatnonzero(absent).tolist(),
        common_library_seed=302501, known_distance_family=known_distance, model='empirical source likelihood draw; teacher-scaled conditional transfer, not independent absolute truth')


def model_library(model, n=65536, seed=302599):
    """Known generative-model control, including audited Gaussian flux support.

    This mathematical control does not claim that negative intrinsic SED flux is
    physical. Fresh libraries per realization avoid treating repeated templates
    as independent observed calibrators.
    """
    rng = np.random.default_rng(seed)
    component = rng.choice(len(model.weights), size=n, p=model.weights)
    values = np.empty((n, 7)); labels = np.empty(n, 'i1')
    for k in range(len(model.weights)):
        rows = np.flatnonzero(component == k)
        values[rows] = rng.multivariate_normal(model.means[k], model.covariances[k], size=len(rows))
        labels[rows] = rng.choice(3, size=len(rows), p=model.classes[k])
    return dict(flux=values[:, :6], mask=np.ones((n, 6), bool),
        identity=-(800000000000000000+seed*100000+np.arange(n, dtype='i8')),
        class_label=labels, feh=values[:, 6], feh_valid=np.ones(n, bool),
        reference_shift=np.zeros(n))
