"""Low-dimensional class/radial/within-class SED target-population inference."""
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from .compression import quadrature, DISTANCE_MU_BOUNDARIES
from .inference import distance_basis
from scipy.stats import norm


def component_response(compressed, model, order=3):
    """Unselected likelihood/Z_ref integrated in continuous radial bases and bins."""
    q = quadrature(compressed, order)
    k = q['component']; mu = q['mu']
    correction = q['mass']/(model.weights[k]*norm.pdf(mu, 15., 5.))
    radial = distance_basis(mu)
    bins = np.searchsorted(DISTANCE_MU_BOUNDARIES, mu, side='right')
    result = np.zeros((len(model.weights), len(radial), 6))
    for d in range(len(radial)):
        result[:, d] = np.bincount(k*6+bins, correction*radial[d],
                                   minlength=len(model.weights)*6).reshape(-1, 6)
    return result


def sed_tilt_features(model):
    mean = model.means
    valid = (mean[:, 0] > 0) & (mean[:, 1] > 0)
    color = np.zeros(len(mean)); color[valid] = -2.5*np.log10(mean[valid, 0]/mean[valid, 1])
    values = np.c_[mean[:, 6], color]
    center = model.weights@values
    scale = np.sqrt(model.weights@((values-center)**2)); scale = np.maximum(scale, .1)
    return np.clip((values-center)/scale, -3, 3), dict(center=center, scale=scale,
        names=['component_mean_corrected_FeH', 'component_mean_g_minus_r'], valid_color=valid)


def mixture_response(response, selection, sed_weights, classes, features, beta):
    logprior = np.log(sed_weights)[:, None]+np.log(classes)+(features@beta)[:, None]
    prior = softmax(logprior, axis=0)
    likelihood = np.einsum('ikd,kc->icd', response, prior, optimize=True)
    alpha = np.einsum('ikd,kc->icd', selection, prior, optimize=True)
    return likelihood, alpha, prior


def population_objective(parameters, response, selection, sed_weights, classes,
                         features, prior_center, prior_sigma=.7, beta_sigma=.3,
                         case_weights=None):
    n, _, d = response.shape; c = classes.shape[1]; size = c*d
    theta = softmax(np.r_[parameters[:size-1], 0.]).reshape(c, d)
    beta = parameters[size-1:]
    likelihood, alpha, prior = mixture_response(response, selection, sed_weights, classes, features, beta)
    total = np.einsum('icd,cd->i', likelihood, theta)
    entry = np.einsum('icd,cd->i', alpha, theta)
    if (total <= 0).any() or (entry <= 0).any(): raise ValueError('Zero population support')
    w = np.ones(n) if case_weights is None else np.asarray(case_weights)
    posterior = likelihood*theta[None]/total[:, None, None]
    selected = alpha*theta[None]/entry[:, None, None]
    delta = parameters[:size-1]-prior_center
    value = -w@(np.log(total)-np.log(entry))+.5*np.sum((delta/prior_sigma)**2)+.5*np.sum((beta/beta_sigma)**2)
    gradient_theta = -(w@(posterior-selected).reshape(n, size))[:-1]+delta/prior_sigma**2
    mean_feature = prior.T@features
    gradient_beta = []
    for j in range(features.shape[1]):
        derivative = prior*(features[:, j, None]-mean_feature[None, :, j])
        derivative_l = np.einsum('ikd,kc,cd->i', response, derivative, theta, optimize=True)
        derivative_a = np.einsum('ikd,kc,cd->i', selection, derivative, theta, optimize=True)
        gradient_beta.append(-w@(derivative_l/total-derivative_a/entry)+beta[j]/beta_sigma**2)
    return float(value), np.r_[gradient_theta, gradient_beta]


def fit_target(response, selection, model, *, case_weights=None, prior_sigma=.7,
               beta_sigma=.3, reference_theta=None, shrinkage_center=None):
    response = np.asarray(response, float)
    selection = np.broadcast_to(selection, response.shape)
    features, feature_definition = sed_tilt_features(model)
    c, d = model.classes.shape[1], response.shape[2]
    if reference_theta is None:
        frequency = model.weights@model.classes
        reference_theta = frequency[:, None]*np.full((1, d), 1/d)
    reference_theta = np.asarray(reference_theta).ravel()
    center = (np.log(reference_theta[:-1]/reference_theta[-1]) if shrinkage_center is None
              else np.asarray(shrinkage_center))
    initial = np.r_[center, np.zeros(features.shape[1])]
    arguments = (response, selection, model.weights, model.classes, features, center,
                 prior_sigma, beta_sigma, case_weights)
    fit = minimize(population_objective, initial, args=arguments, jac=True,
                   method='BFGS', options=dict(maxiter=400, gtol=1e-6))
    if not fit.success and abs(fit.jac).max() > 1e-4: raise RuntimeError('Joint target fit failed: '+fit.message)
    step = 1e-4; hessian = np.empty((len(initial), len(initial)))
    for j in range(len(initial)):
        offset = np.zeros(len(initial)); offset[j] = step
        hessian[:, j] = (population_objective(fit.x+offset, *arguments)[1]-population_objective(fit.x-offset, *arguments)[1])/(2*step)
    hessian = (hessian+hessian.T)/2
    curvature = np.linalg.eigvalsh(hessian)
    if curvature.min() <= 0: raise RuntimeError('Joint target curvature not identified')
    return dict(parameters=fit.x, covariance=np.linalg.inv(hessian), hessian=hessian,
        theta=softmax(np.r_[fit.x[:c*d-1], 0.]).reshape(c, d), beta=fit.x[c*d-1:],
        prior_center=center, prior_sigma=prior_sigma, beta_sigma=beta_sigma,
        features=features, feature_definition=feature_definition, objective=float(fit.fun),
        gradient_max=float(abs(fit.jac).max()), minimum_curvature=float(curvature.min()))


def joint_distance_probability(response_bins, model, theta, beta):
    features, _ = sed_tilt_features(model)
    prior = softmax(np.log(model.weights)[:, None]+np.log(model.classes)+(features@beta)[:, None], axis=0)
    joint = np.einsum('ikdj,kc,cd->icj', response_bins, prior, theta, optimize=True)
    total = joint.sum(axis=(1, 2))
    if (total <= 0).any(): raise ValueError('Zero joint target support')
    return joint/total[:, None, None]
