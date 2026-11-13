"""Row-local coordinate qualification before vectorized sky transformations."""
import numpy as np
from photdis.stage1 import galactic


def qualified_sky(ra,dec):
    ra,dec=np.asarray(ra,float),np.asarray(dec,float)
    if ra.ndim!=1 or ra.shape!=dec.shape:raise ValueError('Matching one-dimensional coordinates required')
    valid=np.isfinite(ra)&np.isfinite(dec)&(ra>=0)&(ra<360)&(abs(dec)<=90)
    latitude=np.full(len(ra),np.nan);cell=np.full(len(ra),-1,'i8')
    if valid.any():
        # healpy's rotation can collapse an entire input array to a scalar NaN
        # when any vector is invalid; never let a bad row contaminate valid rows.
        latitude[valid]=galactic(ra[valid],dec[valid])[1]
        cell[valid]=np.floor(ra[valid]/10).astype('i8')*18+np.minimum(17,np.floor((dec[valid]+90)/10).astype('i8'))
    return dict(valid_coordinates=valid,latitude=latitude,cell=cell)
