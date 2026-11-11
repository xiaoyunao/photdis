"""Small selection-normalized population model, independent of QRF posteriors."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

DISTANCES = np.array([1., 3.5, 7.5, 15., 30.])
DISTANCE_PRIOR = np.array([.25, .30, .25, .12, .08])
COLOR_EDGES = (np.array([-np.inf, 0., .4, .8, 1.2, 1.6, np.inf]),
               np.array([-np.inf, 0., .3, .6, .9, np.inf]),
               np.array([-np.inf, 12., 14., 16., 17., 18., 19., 20., 22., np.inf]))
NCELL = int(np.prod([len(e)-1 for e in COLOR_EDGES]))


def observation_cell(mags):
    m=np.asarray(mags);x=(m[:,0]-m[:,1],m[:,1]-m[:,2],m[:,1])
    return np.ravel_multi_index(tuple(np.searchsorted(e,v,side='right')-1 for e,v in zip(COLOR_EDGES,x)),tuple(len(e)-1 for e in COLOR_EDGES))


def component_weights(theta, base):
    theta=np.atleast_2d(theta)
    group=np.log(np.asarray(base))[None,:]+np.column_stack((np.zeros(len(theta)),theta[:,:3]))
    group=np.exp(group-logsumexp(group,axis=1)[:,None])
    dist=np.log(DISTANCE_PRIOR)[None,:]+theta[:,3,None]*np.log(DISTANCES/5)
    dist=np.exp(dist-logsumexp(dist,axis=1)[:,None])
    return (group[:,:,None]*dist[:,None,:]).reshape(len(theta),20)


def selected_probabilities(weights, response):
    """Response rows are P(observation cell AND selection | component)."""
    mass=np.asarray(weights)@np.asarray(response)
    normalizer=mass.sum(axis=1)
    if np.any(normalizer<=0):raise ValueError('Selection has zero support')
    return mass/normalizer[:,None]


def posterior_grid(counts, response, base, power=14, seed=20261111):
    theta=6*qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)-3
    eta=component_weights(theta,base);alpha=response.sum(axis=1)
    prob=selected_probabilities(eta,response)
    loglike=np.atleast_2d(counts)@np.log(np.maximum(prob,1e-300)).T
    posterior=np.exp(loglike-logsumexp(loglike,axis=1)[:,None])
    selected=eta*alpha;selected/=selected.sum(axis=1)[:,None]
    return theta,selected,posterior


def weighted_quantile(values, weights, levels=(.025,.5,.975)):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return np.interp(levels,np.cumsum(w)/w.sum(),v)


def summarize_posterior(theta, selected, weight):
    mean=weight@selected
    groups=selected.reshape(-1,4,5).sum(axis=2)
    distances=selected.reshape(-1,4,5).sum(axis=1)
    values=np.column_stack((groups,distances))
    intervals=np.array([weighted_quantile(v,weight) for v in values.T])
    center=theta-weight@theta;cov=(center.T*weight)@center
    sd=np.sqrt(np.maximum(np.diag(cov),1e-30));cor=cov/sd[:,None]/sd[None,:]
    return mean,intervals,dict(ess=float(1/np.sum(weight**2)),boundary=float(weight@(np.max(abs(theta),axis=1)>2.8)),
        max_abs_correlation=float(np.max(abs(cor-np.eye(4)))),correlation=cor)


def response_from_draws(cells, component, selected, ncomponent=20, floor=1e-4):
    response=np.zeros((ncomponent,NCELL));alpha=np.zeros(ncomponent)
    for j in range(ncomponent):
        u=component==j;n=u.sum()
        if not n:raise ValueError('Empty component')
        alpha[j]=np.mean(selected[u])
        response[j]=np.bincount(cells[u&selected],minlength=NCELL)/n
        # Conditional contamination floor preserves each measured selection rate.
        response[j]=(1-floor)*response[j]+floor*alpha[j]/NCELL
    return response
