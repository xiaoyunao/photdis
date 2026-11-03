import numpy as np
from photdis.baseline_v1_1 import gate_objective,oracle_weights,integrate
from photdis.baseline_v1 import PopulationModel,integrate as old_integrate
from photdis.baseline_v0 import HistogramQRF


def test_joint_likelihood_gradient_and_rescaling():
    rng=np.random.default_rng(4);x=rng.normal(size=(17,5));x[:,0]=1
    li=np.exp(rng.normal(size=(17,4)));labels=np.arange(17)%4-1
    w=rng.uniform(.1,2,17);theta=rng.normal(size=20)*.1
    _,g=gate_objective(theta,x,li,labels,w);fd=[]
    for j in range(len(theta)):
        d=np.zeros_like(theta);d[j]=1e-6
        fd.append((gate_objective(theta+d,x,li,labels,w)[0]-gate_objective(theta-d,x,li,labels,w)[0])/2e-6)
    assert np.allclose(g,fd,atol=1e-8)
    _,g2=gate_objective(theta,x,li*rng.uniform(.1,10,(17,1)),labels,w)
    assert np.allclose(g,g2,atol=1e-12)


def test_oracle_fallback_mass_and_unknown_labels():
    bs=np.array([[1,1,1],[1,0,1],[1,1,1]],bool)
    a,lost=oracle_weights(np.array([1,1,-1]),bs,np.array([.8,.2,.7]),'full')
    assert np.allclose(a,[[0,.2,0,.8],[0,0,0,1],[0,0,0,1]])
    assert np.allclose(lost,[0,.8,0])
    a,_=oracle_weights(np.array([1,1,-1]),bs,np.zeros(3),'direct')
    assert np.allclose(a,[[0,1,0,0],[0,1,0,0],[0,0,0,1]])


def test_original_variants_reproduced_and_diagnostic_labels_isolated():
    rng=np.random.default_rng(8);m=rng.normal(size=(450,5));labels=np.r_[np.repeat(np.arange(3),120),np.full(90,-1)]
    y=np.where(labels<0,0,labels)*3.
    wide=HistogramQRF(2,n_estimators=2,min_samples_leaf=8).fit(m,y,labels)
    model=PopulationModel(wide).fit(m,y,labels)
    mags=m[:8];errors=np.full((8,5),.03);ids=np.arange(8)
    old=('wide','natural','balanced','oracle');allv=old+('oracle_support','oracle_full')
    a=old_integrate(model,mags,errors,ids,draws=8,variants=old,oracle_labels=np.ones(8,int))
    b=integrate(model,mags,errors,ids,allv,labels=np.ones(8,int),draws=8)
    c=integrate(model,mags,errors,ids,allv,labels=np.zeros(8,int),draws=8)
    for k in old:
        assert np.allclose(a[k]['pdf'],b[k]['pdf'],atol=2e-7)
        assert np.allclose(a[k]['mu'],b[k]['mu'],atol=2e-6)
        if k!='oracle':assert np.array_equal(b[k]['pdf'],c[k]['pdf'])
    for p in b.values():
        assert np.allclose(p['weights'].sum(axis=1),1,atol=2e-6)
        assert np.all(p['weights']>=0) and np.allclose(p['pdf'].sum(axis=1),1)


def test_observed_parallax_truncation_and_negative_measurement():
    from scipy.stats import truncnorm,norm
    from photdis.baseline_parallax import parallax_check
    # Effectively point-mass M=0, g=15 gives a true parallax of 0.1 mas.
    # Observed negative parallax remains a legitimate negative-SNR test object.
    edges=np.array([-1e-8,1e-8]);g=np.array([15.,15.]);y=np.array([-.1,.3]);s=np.array([.2,.2])
    lo=np.array([-np.inf,0.]);hi=np.array([0.,5.])
    p=parallax_check(np.ones((2,1)),edges,g,y,s,lo,hi)
    for i in range(2):
        a=lo[i]-.5;b=hi[i]-.5;dist=truncnorm(a,b,loc=.1,scale=.2)
        assert abs(p['pit'][i]-dist.cdf(y[i]))<1e-9
        assert abs(p['conditional_mean'][i]-dist.mean())<1e-9
        assert abs(p['conditional_variance'][i]-dist.var())<1e-9
        assert abs(p['conditional_density'][i]-dist.pdf(y[i]))<1e-9
    allp=parallax_check(np.ones((2,1)),edges,g,y,s,np.full(2,-np.inf),np.full(2,np.inf))
    assert np.allclose(allp['pit'],norm.cdf((y-.1)/s))
    assert np.allclose(allp['selection_probability'],1.)


def test_selection_simulation_respects_negative_snr_and_expected_center():
    from photdis.baseline_parallax import simulate_selected,parallax_check
    args=(np.ones((1,1)),np.array([-1e-8,1e-8]),np.array([15.]),np.array([.2]),np.array([-np.inf]),np.array([0.]))
    sim=simulate_selected(*args,np.random.default_rng(919),repeats=20000)
    assert np.all(sim<0)
    mean=parallax_check(args[0],args[1],args[2],np.array([-.1]),*args[3:])['conditional_mean'][0]
    assert abs(sim.mean()-mean)<.003


def test_parent_coverage_uses_field_size_weights():
    from baseline_v1_1_risk import estimate
    t=dict(field=np.array([0,0,1,1]),sample_weight=np.array([50.,50.,5.,5.]))
    result=estimate(t,np.array([1,1,0,0],bool))
    assert np.isclose(result['fraction'],100/110)
    assert result['se']==0  # both SRS strata are constant


def test_cal_adoption_requires_gain_without_dwarf_damage():
    from baseline_v1_1_score import adoption
    classes=('class_dwarf','class_evolved','class_transition','all')
    old={k:dict(n=100,tail_1=.2,coverage95=.94,width95=3.,mean_log_density=-1.) for k in classes}
    equal={k:dict(v) for k,v in old.items()}
    assert not adoption(old,equal,{'success':True})['adopt']
    good={k:dict(v,tail_1=.1) for k,v in old.items()}
    assert adoption(old,good,{'success':True})['adopt']
    good['class_dwarf']['tail_1']=.25
    assert not adoption(old,good,{'success':True})['adopt']
    good['class_dwarf']['tail_1']=.1
    assert not adoption(old,good,{'success':False})['adopt']
