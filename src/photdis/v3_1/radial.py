"""Selection-normalized fixed-SED background plus one regularized local structure."""
import numpy as np
from scipy.special import expit, logit, softmax
from scipy.stats import norm
from scipy.optimize import minimize
from photdis.v3.inference import distance_basis, DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from photdis.v3.likelihood import trapezoid_weights


class RadialPopulation:
    """Single-field radial control with matched fixed class-conditional SEDs.

    Curves are object x class x mu. Likelihood curves may each be divided by a
    constant independent of the parameters. Entry curves are probabilities.
    """
    def __init__(self, mu, likelihood, entry, class_frequency, *, local=False,
                 case_weights=None):
        self.mu = np.asarray(mu,float); self.weights = trapezoid_weights(self.mu)
        omitted = (norm.cdf((self.mu[0]-DISTANCE_BASIS_MEAN)/DISTANCE_BASIS_SIGMA)
                   +norm.sf((self.mu[-1]-DISTANCE_BASIS_MEAN)/DISTANCE_BASIS_SIGMA))
        if omitted.max() > 1e-8: raise ValueError('Broad radial prior tail outside integration support')
        self.response = np.asarray(likelihood,float); self.entry = np.asarray(entry,float)
        if self.response.shape != self.entry.shape or self.response.ndim != 3 or self.response.shape[1:] != (3,len(self.mu)):
            raise ValueError('Expected matching object x three classes x mu curves')
        if not np.isfinite(self.response).all() or (self.response < 0).any():
            raise ValueError('Finite nonnegative native likelihood required')
        if not np.isfinite(self.entry).all() or (self.entry < 0).any() or (self.entry > 1+1e-12).any():
            raise ValueError('Conditional entry probabilities required')
        frequency = np.asarray(class_frequency,float)
        if frequency.shape != (3,) or (frequency <= 0).any() or not np.isclose(frequency.sum(),1):
            raise ValueError('Positive normalized class frequencies required')
        base = (frequency[:,None]*np.ones((1,5))/5).ravel()
        self.center = np.log(base[:-1]/base[-1]); self.theta_sigma = np.hypot(.7,.35)
        self.local = bool(local); self.dimension = 19 if self.local else 14
        self.case_weights = np.ones(len(self.response)) if case_weights is None else np.asarray(case_weights,float)
        if self.case_weights.shape != (len(self.response),) or not np.isfinite(self.case_weights).all() or (self.case_weights <= 0).any():
            raise ValueError('Positive per-object case weights required')
        self.case_weights = self.case_weights/self.case_weights.mean()
        basis = distance_basis(self.mu)*self.weights
        self.broad = self.response@basis.T; self.broad_entry = self.entry@basis.T
        self.response_matrix = self.response.reshape(-1,len(self.mu))
        self.entry_matrix = self.entry.reshape(-1,len(self.mu))

    def decode(self,z):
        z = np.asarray(z,float)
        if z.shape != (self.dimension,) or not np.isfinite(z).all(): raise ValueError('Invalid radial coordinates')
        theta = softmax(np.r_[self.center+self.theta_sigma*z[:14],0.]).reshape(3,5)
        if not self.local: return theta,np.zeros(3),14.,.25
        eta = expit(-2+1.5*z[14:17])
        location = 8+12*expit(z[17])
        width = .04+1.96*expit(logit((.25-.04)/1.96)+z[18])
        return theta,eta,float(location),float(width)

    def local_terms(self,location,width):
        if norm.cdf((self.mu[0]-location)/width)+norm.sf((self.mu[-1]-location)/width) > 1e-8:
            raise ValueError('Local prior tail outside integration support')
        x = (self.mu-location)/width; density = norm.pdf(x)/width
        weighted = density*self.weights
        if abs(weighted.sum()-1) > 1e-6: raise ValueError('Local density grid is unresolved')
        columns = np.stack([weighted,weighted*x/width,weighted*(x*x-1)],axis=1)
        return ((self.response_matrix@columns).reshape(-1,3,3),
                (self.entry_matrix@columns).reshape(-1,3,3))

    def objective(self,z):
        theta,eta,location,width = self.decode(z)
        if self.local:
            loc,alpha = self.local_terms(location,width)
            r = (1-eta)[None,:,None]*self.broad+eta[None,:,None]*loc[:,:,0,None]
            a = (1-eta)[None,:,None]*self.broad_entry+eta[None,:,None]*alpha[:,:,0,None]
        else: r,a = self.broad,self.broad_entry
        total = np.einsum('ncd,cd->n',r,theta); selected = np.einsum('ncd,cd->n',a,theta)
        if (total <= 0).any() or (selected <= 0).any(): raise ValueError('Zero population/entry support')
        wr = self.case_weights/total; wa = self.case_weights/selected
        value = .5*np.dot(z,z)-self.case_weights@(np.log(total)-np.log(selected))
        gradient = np.array(z,copy=True,dtype=float)
        gtheta = theta*(np.einsum('n,ncd->cd',wa,a)-np.einsum('n,ncd->cd',wr,r))
        gtheta -= theta*gtheta.sum()
        gradient[:14] += self.theta_sigma*gtheta.ravel()[:-1]
        if self.local:
            dr = np.einsum('cd,ncd->nc',theta,loc[:,:,0,None]-self.broad)
            da = np.einsum('cd,ncd->nc',theta,alpha[:,:,0,None]-self.broad_entry)
            gradient[14:17] += 1.5*eta*(1-eta)*(wa@da-wr@dr)
            amplitude = theta.sum(axis=1)*eta
            dl = (wa@alpha[:,:,1]-wr@loc[:,:,1])@amplitude
            ds = (wa@alpha[:,:,2]-wr@loc[:,:,2])@amplitude
            position_fraction = (location-8)/12; width_fraction = (width-.04)/1.96
            gradient[17] += dl*12*position_fraction*(1-position_fraction)
            gradient[18] += ds*1.96*width_fraction*(1-width_fraction)/width
        return float(value),gradient

    def starts(self):
        output = [np.zeros(self.dimension)]
        if self.local:
            for position in [9.,11.,13.,15.,17.,19.]:
                for width in [.08,.30]:
                    z = np.zeros(self.dimension); z[17] = logit((position-8)/12)
                    z[18] = logit((width-.04)/1.96)-logit((.25-.04)/1.96)
                    output.append(z)
        return output

    def fit(self):
        results = []
        for initial in self.starts():
            fit = minimize(self.objective,initial,jac=True,method='BFGS',options=dict(maxiter=600,gtol=1e-5))
            covariance = (fit.hess_inv+fit.hess_inv.T)/2
            eigen,vectors = np.linalg.eigh(covariance)
            metric = (vectors*np.clip(eigen,1e-5,10.))@vectors.T
            results.append(dict(map=fit.x,objective=float(fit.fun),metric=metric,
                gradient_max=float(abs(fit.jac).max()),success=bool(fit.success),message=str(fit.message)))
        # Retain failures and every local optimum; caller decides numerical acceptance.
        return sorted(results,key=lambda x:x['objective'])
