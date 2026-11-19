"""Scientific invariants of the fixed stream/branch experiments."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from halo_filtering import channel_weights
from stream_common import profile,C,track
from stream_sequence import polyline,posterior,count_median
from stream_local import boundaries,accept,curves
from stream_conditioning import propensity
from stream_fold_audit import native_bounds


def test_rgb_exact_modulus_partition():
    mu=np.round(np.arange(14,20.001,.1),1)
    a=channel_weights(mu,[[26,50],[50,100],[26,100]])
    f=np.log(50/26)/np.log(100/26)
    np.testing.assert_allclose(a[2],f*a[0]+(1-f)*a[1],atol=2e-15)
    for lo,hi in ((10**.8,10),(10,15),(15,26),(26,50),(50,100)):
        e=lo*(hi/lo)**(np.arange(4)/3)
        w=channel_weights(mu,np.column_stack((e[:-1],e[1:])))
        np.testing.assert_allclose(w.mean(axis=0),channel_weights(mu,[[lo,hi]])[0],atol=3e-15)


def test_track_geometry_and_placeholder():
    for name in ('Pal5_tail','Sgr_north','Sgr_south'):
        a,_=track(C['local'][name]['track']);lo,hi=C['local'][name]['ra_range'];use=(a['ra']>lo)&(a['ra']<hi)
        offset,along=profile(name,a['ra'][use],a['dec'][use]);np.testing.assert_allclose(offset,0,atol=1e-10)
        assert ((along>=0)&(along<=1)).all()
    a,_=track('Pal5.ibata2024');assert np.allclose(a['distance'],1)


def test_polyline_keeps_both_luminosity_branches():
    # Same adjacent colors at two actual evolutionary luminosities.
    m=np.array([[4.2,4.,3.9],[5.2,5.,4.9],[5.3,5.1,5.]])
    t=dict(south=m,phase=np.zeros(3))
    nodes,weights=polyline(t,'south')
    assert nodes[:,0].min()<4.5 and nodes[:,0].max()>5
    np.testing.assert_allclose(weights.sum(),1)
    assert (weights>0).all()
    q,_=posterior(np.array([[19.2,19.,18.9]]),np.array([[.02,.02,.02]]),np.array([1357]),(nodes,weights))
    assert q[0,4]/q[0,0]>1.4  # Both distance alternatives remain in the interval.


def test_full_plotted_boundary_is_the_count_boundary():
    m=np.array([[4.4,4.1,4.],[4.7,4.3,4.2]])
    cc=curves(m,np.zeros(2),'MS_TO');ec=(np.array([15,25]),np.array([.07,.07]));g=np.array([20.5])
    lo,hi=boundaries(g,16,cc,ec,'MS_TO')
    inside=np.array([[g[0],g[0]-lo[0,0]-.001,19.]])
    outside=inside.copy();outside[:,1]=g[0]-lo[0,0]+.001
    assert accept(inside,[16],cc,ec,'MS_TO')[0,0]
    assert not accept(outside,[16],cc,ec,'MS_TO')[0,0]
    # The former dotted boundary lacked 2sigma; actual acceptance is wider.
    lo0,_=boundaries(g,16,cc,(ec[0],np.zeros(2)),'MS_TO')
    np.testing.assert_allclose(lo0-lo,.14)


def test_weighting_support_is_a_fixed_intersection():
    rng=np.random.default_rng(777);x=rng.normal(size=(400,7));y=(x[:,0]+rng.normal(size=400)>0).astype(int)
    p=propensity(x,y,np.ones(400));q=propensity(x[:,[0,1,2,3,4,6]],y,np.ones(400))
    fixed=(p>=.05)&(p<=.95)&(q>=.05)&(q<=.95)
    mult=rng.integers(0,4,size=400);b=propensity(x,y,mult);w=mult*np.where(y==1,1-b,b)*fixed
    assert (w[~fixed]==0).all()


def test_common_member_resample_median():
    rng=np.random.default_rng(625)
    for _ in range(20):
        value=rng.normal(size=50);weight=rng.integers(0,5,size=50)
        np.testing.assert_allclose(count_median(value,weight),np.median(np.repeat(value,weight)))


def test_fold_audit_uses_each_native_curve():
    m=np.array([[8.5,8,7.8],[6.3,6,5.8],[4.15,4,3.8],[3.68,3.5,3.3]])
    north=m.copy();north[:,0]+=.02
    t=dict(south=m,north=north,phase=np.zeros(4),north_support=np.ones(4,bool))
    np.testing.assert_allclose(native_bounds(t,'south'),[.15,.18])
    np.testing.assert_allclose(native_bounds(t,'north'),[.17,.20])
