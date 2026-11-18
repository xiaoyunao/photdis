"""Hard native CMD acceptance and explicitly overlapping modulus channels."""
import numpy as np


def envelope(magnitude, color, width=.5):
    grid=np.arange(-5,17.0001,.01)
    low=np.full(len(grid),np.inf);high=np.full(len(grid),-np.inf)
    for i in range(len(magnitude)-1):
        a,b=magnitude[i:i+2];c,d=color[i:i+2]
        lo,hi=min(a,b),max(a,b)
        use=(grid+width/2>=lo)&(grid-width/2<=hi)
        if not use.any():continue
        left=np.maximum(grid[use]-width/2,lo)
        right=np.minimum(grid[use]+width/2,hi)
        if abs(b-a)<1e-10:
            v0=np.full(len(left),min(c,d));v1=np.full(len(left),max(c,d))
        else:
            v0=c+(d-c)*(left-a)/(b-a);v1=c+(d-c)*(right-a)/(b-a)
        low[use]=np.minimum(low[use],np.minimum(v0,v1))
        high[use]=np.maximum(high[use],np.maximum(v0,v1))
    return grid,low,high


def prepare_templates(template):
    result={}
    for native,key in enumerate(('south','north')):
        mag=template[key]
        valid=np.ones(len(mag),bool) if native==0 else template['north_support']
        curves=[]
        for phase in (0,2):
            use=valid&(template['phase']==phase)
            # Preserve discontinuities created by finite native-transfer support.
            indices=np.flatnonzero(use)
            splits=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            for segment in splits:
                if len(segment)>1:
                    curves.append(envelope(mag[segment,0],mag[segment,0]-mag[segment,1]))
        result[native]=curves
    return result


def accepted(mag,observed_g,native,mu,templates,error_curve):
    out=np.zeros((len(mag),len(mu)),bool)
    gr=mag[:,0]-mag[:,1]
    for code in (0,1):
        rows=np.flatnonzero((native==code)&(gr>=-.2)&(gr<1.2))
        if not len(rows):continue
        e=np.interp(observed_g[rows],error_curve['g_centers'],error_curve['sigma_g'][code])
        delta=mag[rows,0,None]-np.asarray(mu)[None,:]
        for grid,lo,hi in templates[code]:
            lower=np.interp(delta.ravel(),grid,lo,left=np.inf,right=np.inf).reshape(delta.shape)
            upper=np.interp(delta.ravel(),grid,hi,left=-np.inf,right=-np.inf).reshape(delta.shape)
            out[rows]|=(gr[rows,None]>=lower-.05-2*e[:,None])&(gr[rows,None]<=upper+.1+2*e[:,None])
    return out


def channel_weights(mu,intervals):
    mu=np.asarray(mu)
    edges=np.r_[mu[0],(mu[:-1]+mu[1:])/2,mu[-1]]
    result=[]
    for a,b in intervals:
        lo,hi=10+5*np.log10([a,b])
        overlap=np.maximum(0,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
        result.append(overlap/overlap.sum())
    return np.array(result)
