"""Fixed balanced classification and prediction-only distance decisions."""
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .baseline_v0 import features, distance_quantiles
from .baseline_v1 import quantiles_interpolated
from .baseline_v1_1 import transfer_unsupported


def balanced_weights(labels):
    labels=np.asarray(labels)
    if np.any((labels<0)|(labels>2)):raise ValueError('Reliable three-class labels required')
    count=np.bincount(labels,minlength=3)
    if np.any(count==0):raise ValueError('All three classes required')
    return len(labels)/(3*count[labels])


class DirectClassifier:
    def __init__(self,dim,kind,config):self.dim=dim;self.kind=kind;self.config=config
    def fit(self,mags,labels,weight):
        x=features(mags,self.dim)
        if self.kind=='quadratic':
            self.scaler=StandardScaler().fit(x,sample_weight=weight)
            self.poly=PolynomialFeatures(2,include_bias=False)
            x=self.poly.fit_transform(self.scaler.transform(x))
            self.model=LogisticRegression(C=self.config['logistic_C'],max_iter=2000,random_state=self.config['seed'])
        elif self.kind=='forest':
            self.model=RandomForestClassifier(**self.config['forest'],random_state=self.config['seed'],n_jobs=8)
        else:raise ValueError(self.kind)
        self.model.fit(x,labels,sample_weight=weight)
        assert np.array_equal(self.model.classes_,np.arange(3))
        return self
    def predict_proba(self,mags):
        x=features(mags,self.dim)
        if self.kind=='quadratic':x=self.poly.transform(self.scaler.transform(x))
        return self.model.predict_proba(x)


def conservative_threshold(scores,budget):
    """Lowest inclusive threshold admitting <=floor(budget*N) dwarf scores.

    The nextafter step rejects a tied boundary in its entirety. Empty controls
    cannot calibrate a threshold. Infinity produces an explicitly empty group.
    """
    x=np.sort(np.asarray(scores)[np.isfinite(scores)])[::-1]
    if not len(x):return np.inf
    allowed=int(np.floor(budget*len(x)+1e-10))
    if allowed>=len(x):return -np.inf
    return float(np.nextafter(x[allowed],np.inf))


def distance_summary(pdf,edges,g):
    return distance_quantiles(g,quantiles_interpolated(pdf,edges))[0]


def mix_cached(branch_pdf,raw,branch_supported):
    if raw.shape[1]==3:raw=np.column_stack([raw,np.zeros(len(raw))])
    weights,lost=transfer_unsupported(raw,branch_supported)
    pdf=np.einsum('nk,nkb->nb',weights,branch_pdf)
    pdf/=pdf.sum(axis=1)[:,None]
    return pdf,weights,lost


def main_component(branch_pdf,weights):
    index=np.argmax(weights,axis=1)
    return branch_pdf[np.arange(len(weights)),index],index


def single_mask(weights,branch_supported,broad_supported,confidence,margin):
    index=np.argmax(weights,axis=1);ordered=np.sort(weights,axis=1)
    supports=np.column_stack([branch_supported,broad_supported])
    return (broad_supported&supports[np.arange(len(index)),index]&
            (ordered[:,-1]>=confidence)&(ordered[:,-1]-ordered[:,-2]>=margin))


def fraction_interval(k,n):
    """Wilson descriptive 95% interval; no independent-sky calibration claim."""
    if not n:return [None,None]
    p=k/n;z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def nominal_old_weights(model,mags,variant):
    """Unmarginalized old scores for a common observed-color ranking control."""
    p=model.gate.predict_proba(features(mags,model.dim))
    if variant=='balanced':p=p/np.maximum(model.frequencies,1e-10)
    p=p/p.sum(axis=1)[:,None]
    return np.column_stack([p,np.zeros(len(p))])
