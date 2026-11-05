"""Histogram-compressed QRF and shared-band propagation for baseline v0.

Leaf member weights are the existing QuantileForest convention. Compression
changes only the representation (0.1mag bins), never fits a separate regressor.
"""
import numpy as np
from scipy.special import ndtri
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

Q=np.array([.025,.16,.5,.84,.975])
BANDS=('g0','r0','z0','w10','w20');ERR=('sigma_g','sigma_r','sigma_z','sigma_w1','sigma_w2')

def features(mags,dim):
    return (mags[:,:-1]-mags[:,1:])[:,:dim]

def band_noise(ids,draw,nband=5):
    """Counter-based normals: identical identities/bands across models/chunks."""
    with np.errstate(over='ignore'):
        x=np.asarray(ids,dtype=np.uint64)[:,None]+np.arange(nband,dtype=np.uint64)[None,:]*np.uint64(0x9e3779b97f4a7c15)+np.uint64(draw+1)*np.uint64(0xd1b54a32d192ed03)
        x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>27))*np.uint64(0x94d049bb133111eb);x=x^(x>>31)
    u=((x>>11).astype(float)+.5)/(2.**53)
    return ndtri(np.clip(u,1e-15,1-1e-15))

def quantiles(pdf,centers):
    c=np.cumsum(pdf,axis=1);c/=c[:,-1,None]
    return centers[np.column_stack([np.argmax(c>=q,axis=1) for q in Q])]

def summaries(pdf,centers):
    q=quantiles(pdf,centers)
    smooth=gaussian_filter1d(pdf,2,axis=1)
    peaks=(smooth>=np.roll(smooth,1,axis=1))&(smooth>=np.roll(smooth,-1,axis=1));peaks[:,0]=peaks[:,-1]=False
    first=np.argmax(smooth,axis=1);a=np.where(peaks,smooth,-1.)
    a[np.abs(centers[None,:]-centers[first,None])<1.0]=-1
    second=np.argmax(a,axis=1);exists=a[np.arange(len(a)),second]>0
    mass1=np.sum(pdf*(abs(centers[None,:]-centers[first,None])<=.5),axis=1)
    mass2=np.sum(pdf*(abs(centers[None,:]-centers[second,None])<=.5),axis=1);mass2[~exists]=0
    modes=np.column_stack([centers[first],np.where(exists,centers[second],np.nan),mass1,mass2])
    ambiguity=(mass2>=.2)&(abs(modes[:,0]-modes[:,1])>=2)
    return q,modes,ambiguity

class HistogramQRF:
    def __init__(self,dim,seed=20261102,n_estimators=128,min_samples_leaf=32,max_depth=20,bin_width=.1,feature_schema='adjacent_colors'):
        if feature_schema not in ('adjacent_colors', 'grz_r0') or (feature_schema == 'grz_r0' and dim != 3):
            raise ValueError('grz_r0 requires three features')
        self.feature_schema=feature_schema
        self.dim=dim;self.bin_width=bin_width
        self.forest=RandomForestRegressor(n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=.75,bootstrap=False,n_jobs=1,random_state=seed)
    def feature_matrix(self,mags):
        # Earlier serialized forests predate the explicit schema.
        if getattr(self,'feature_schema','adjacent_colors') == 'grz_r0':
            return np.column_stack((features(mags,2),mags[:,1]))
        return features(mags,self.dim)

    def fit(self,mags,y,labels):
        x=self.feature_matrix(mags);self.forest.fit(x,y)
        self.edges=np.arange(np.floor(np.min(y))-3,np.ceil(np.max(y))+3+self.bin_width,self.bin_width);self.centers=(self.edges[1:]+self.edges[:-1])/2
        bins=np.searchsorted(self.edges,y,side='right')-1;self.tables=[];self.types=[];self.maps=[]
        nbin=len(self.centers);classes=np.where(labels<0,3,labels)
        for tree in self.forest.estimators_:
            leaf=tree.apply(x);unique,inv=np.unique(leaf,return_inverse=True);mapping=np.zeros(tree.tree_.node_count,np.int32);mapping[unique]=np.arange(len(unique))
            hist=np.bincount(inv*nbin+bins,minlength=len(unique)*nbin).reshape(-1,nbin).astype('f4');hist/=hist.sum(axis=1)[:,None]
            cl=np.bincount(inv*4+classes,minlength=len(unique)*4).reshape(-1,4).astype('f4');cl/=cl.sum(axis=1)[:,None]
            self.tables.append(hist);self.types.append(cl);self.maps.append(mapping)
        self.mean=np.mean(x,axis=0);self.scale=np.std(x,axis=0);self.scale[self.scale<1e-8]=1
        z=(x-self.mean)/self.scale;self.nn=NearestNeighbors(n_neighbors=11,n_jobs=1).fit(z)
        self.radius=float(np.percentile(self.nn.kneighbors(z)[0][:,-1],99));self.lo=x.min(axis=0);self.hi=x.max(axis=0)
        return self
    def distribution(self,x):
        n=len(x);pdf=np.zeros((n,len(self.centers)),np.float32);cl=np.zeros((n,4),np.float32)
        for tree,mapping,h,t in zip(self.forest.estimators_,self.maps,self.tables,self.types):
            i=mapping[tree.apply(x)];pdf+=h[i];cl+=t[i]
        pdf/=len(self.tables);cl/=len(self.tables)
        return pdf,cl
    def predict(self,mags,errors,ids,draws=8):
        n=len(mags);pdf=np.zeros((n,len(self.centers)),np.float32);cl=np.zeros((n,4),np.float32)
        for d in range(draws):
            delta=band_noise(ids,d)*np.nan_to_num(errors,nan=0,posinf=0)
            p,c=self.distribution(self.feature_matrix(mags+delta))
            # Effective M=M-delta_g preserves the same g error in colors and mu.
            # Interpolate the CDF instead of PDF; preserve mass at finite grid ends.
            cumulative=np.column_stack([np.zeros(n),np.cumsum(p,axis=1)])
            # CDF at output bin edges shifted by +delta_g.
            e=(self.edges[None,:]+delta[:,0,None]-self.edges[0])/self.bin_width
            l=np.floor(e).astype(int);f=e-l;l0=np.clip(l,0,len(self.centers));l1=np.clip(l+1,0,len(self.centers))
            cc=np.take_along_axis(cumulative,l0,axis=1)*(1-f)+np.take_along_axis(cumulative,l1,axis=1)*f
            cc[:,0]=0;cc[:,-1]=1
            shifted=np.maximum(np.diff(cc,axis=1),0);shifted/=shifted.sum(axis=1)[:,None]
            pdf+=shifted;cl+=c
        pdf/=draws;cl/=draws
        x=self.feature_matrix(mags);dist=self.nn.kneighbors((x-self.mean)/self.scale,n_neighbors=10)[0][:,-1]
        supported=(dist<=self.radius)&np.all((x>=self.lo)&(x<=self.hi),axis=1)
        q,modes,amb=summaries(pdf,self.centers)
        return dict(q_Meffective=q,modes_Meffective=modes,ambiguous=amb,supported=supported,support_distance=dist,class_weights=cl,pdf=pdf)

def distance_quantiles(g0,q):
    mu=g0[:,None]-q[:,::-1]
    return mu,10**((mu-10)/5)

def metrics(mu,truth):
    good=np.isfinite(truth)&np.all(np.isfinite(mu),axis=1);mu=mu[good];truth=truth[good]
    if not len(truth):return {'n':0}
    r=mu[:,2]-truth;bias=np.median(r);rel=10**(r/5)-1
    return dict(n=len(r),bias_mag=float(bias),nmad_mag=float(1.4826*np.median(abs(r-bias))),median_relative=float(np.median(rel)),nmad_relative=float(1.4826*np.median(abs(rel-np.median(rel)))),tail_0p5=float(np.mean(abs(r)>.5)),tail_1=float(np.mean(abs(r)>1)),coverage68=float(np.mean((truth>=mu[:,1])&(truth<=mu[:,3]))),coverage95=float(np.mean((truth>=mu[:,0])&(truth<=mu[:,4]))),width68=float(np.median(mu[:,3]-mu[:,1])),width95=float(np.median(mu[:,4]-mu[:,0])))
