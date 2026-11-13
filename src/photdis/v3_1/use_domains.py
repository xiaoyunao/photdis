"""Observable availability cells, not automatic calibration certificates."""
import numpy as np
from photdis.v3.controls import optical_magnitudes
from photdis.v3.data import WEAK,SIGNIFICANT

R_EDGES=np.array([16.,18.,20.,22.])
COLOR_EDGES=np.array([0.,.3,.6,.9])
DISTANCE_EDGES=np.array([2.,5.,10.,20.,50.])
WISE_LABELS=('unavailable','weak_signed','significant')
N_CELLS=75


def observable_cells(table,admission):
    np.testing.assert_array_equal(table['identity'],admission['identity'])
    if not (table['TYPE']=='PSF').all() or not (admission['TYPE']=='PSF').all():
        raise ValueError('Verified PSF required before use-domain export')
    mag,_=optical_magnitudes(table); r=mag[:,1]-2.5*np.log10(table['transmission'][:,1]); color=mag[:,0]-mag[:,1]
    if not np.isfinite(r).all() or not np.isfinite(color).all():raise ValueError('Finite native grz required')
    np.testing.assert_allclose(r,admission['observed_r'],atol=1e-12,rtol=0)
    available=np.isin(table['band_state'][:,4],[WEAK,SIGNIFICANT])
    flux=table['flux'][:,4]; error=table['flux_error'][:,4]
    available &= np.isfinite(flux) & np.isfinite(error) & (error>0)
    snr=np.divide(flux,error,out=np.full(len(r),np.nan),where=available)
    wise=np.where(available,np.where(snr>=5,2,1),0).astype('i1')
    rb=np.searchsorted(R_EDGES,r,side='right'); cb=np.searchsorted(COLOR_EDGES,color,side='right')
    cell=((rb*5+cb)*3+wise).astype('i2')
    return dict(identity=table['identity'],TYPE=table['TYPE'],cell=cell,r_bin=rb,color_bin=cb,wise_bin=wise,
        observed_r=r,g0_minus_r0=color,candidate=admission['candidate'])


def cell_descriptors():
    rows=[]
    rb=np.r_[-np.inf,R_EDGES,np.inf]; cb=np.r_[-np.inf,COLOR_EDGES,np.inf]
    for r in range(5):
        for c in range(5):
            for w,label in enumerate(WISE_LABELS):
                rows.append(dict(cell=(r*5+c)*3+w,r_bin=r,color_bin=c,wise_bin=w,wise_label=label,
                    r_lower=None if r==0 else float(rb[r]),r_upper=None if r==4 else float(rb[r+1]),
                    color_lower=None if c==0 else float(cb[c]),color_upper=None if c==4 else float(cb[c+1])))
    return rows


def source_distances(evidence):
    """Separate channels with their own precision; never invert signed Gaia."""
    result={'astronn':(evidence['distance_kpc'],evidence['distance_available'],evidence['distance_precise'])}
    for source in ['specdis','spdist']:
        sigma=evidence[source+'_sigma_mu']; valid=evidence[source+'_valid']
        with np.errstate(over='ignore',invalid='ignore'):
            relative=np.sqrt(np.expm1((sigma*np.log(10)/5)**2))
        result[source]=(evidence[source+'_distance'],valid,valid & np.isfinite(relative) & (relative<.2))
    return result


def distance_availability(cells,evidence,role,*,admitted=None):
    """Number of reference pairs by source; correlated channels are not pooled N."""
    cells=np.asarray(cells); rows=[]; available=np.zeros((N_CELLS,6),bool)
    role_mask=evidence['role']==role
    if admitted is not None:role_mask &= np.asarray(admitted,bool)
    for source,(distance,valid,precise) in source_distances(evidence).items():
        if np.any(valid & (~np.isfinite(distance) | (distance<=0))):raise ValueError('Invalid qualified reference distance')
        bins=np.searchsorted(DISTANCE_EDGES,distance,side='right')
        for quality,qmask in [('qualified',valid),('precise',precise)]:
            use=role_mask & qmask
            count=np.bincount(cells[use]*6+bins[use],minlength=N_CELLS*6).reshape(N_CELLS,6)
            if quality=='precise':available |= count>0
            for cell in range(N_CELLS):
                for b in range(6):rows.append(dict(role=role,source=source,quality=quality,cell=cell,distance_bin=b,n=int(count[cell,b])))
    return available,rows


def missing_reference_mass(probability,cells,available):
    p=np.asarray(probability,float); cells=np.asarray(cells); available=np.asarray(available,bool)
    if p.shape!=(len(cells),3,6) or available.shape!=(N_CELLS,6):raise ValueError('Expected joint probabilities and observable distance availability')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=(1,2)),1,atol=1e-8,rtol=0):
        raise ValueError('Normalized joint probabilities required')
    return (p.sum(axis=1)*~available[cells]).sum(axis=1)


def computational_status(candidate,has_prediction,cohort_passed):
    candidate,has_prediction=np.asarray(candidate,bool),np.asarray(has_prediction,bool)
    if candidate.shape!=has_prediction.shape:raise ValueError('Aligned admission and prediction flags required')
    result=np.full(candidate.shape,'outside_candidate_domain',dtype='U32')
    result[candidate & ~has_prediction]='candidate_prediction_missing'
    result[candidate & has_prediction]='conditional_model_available' if cohort_passed else 'numerical_flags_retained'
    return result
