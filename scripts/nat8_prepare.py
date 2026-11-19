#!/usr/bin/env python3
"""Seal original identities, stratified folds and an actual-data gray invariance."""
from nat8_common import *
from photdis.stage1 import require_compute
from seq7_math import mixture
import hashlib


def main():
    require_compute();d=folder('prepare');inputs={};t=old('prepare','roster.npz',inputs);assert len(t['ls_id'])==933 and (t['TYPE']=='PSF').all();fold=np.full(933,-1,'i1');eligible=np.zeros(933,bool);bounds=[];invariance=[]
    for s in C['systems']:
        a=old('predict',f'{s}_E_32.npz',inputs);z=old('prepare',f'{s}_E_32.npz',inputs);ii=a['ii'];assert (t['system'][ii]==s).all();x=z['nodes'][:,0]-z['nodes'][:,1]-C['pivot'];npz(d/f'{s}_E.npz',**a,x=x,nodes=z['nodes'],prior=z['prior'],node_post=z['post'])
        if s=='M5':eligible[ii]=a['support']&np.isin(t['phase'][ii],('lower_MS','turnoff'))
        sample=np.arange(min(8,len(ii)));params=np.array([-.4,.8]);node=z['nodes']+params[0]+params[1]*x[:,None];mm,lw,sg,chi=mixture(t['m'][ii[sample]],t['err'][ii[sample]],node,z['prior']);expected=a['means'][sample]-params[0]-params[1]*x
        err=float(abs(mm-expected).max());we=float(abs(np.exp(lw)-np.exp(a['logw'][sample])).max());ce=float(abs(chi-a['chi'][sample]).max());assert err<1e-11 and we<1e-10 and ce<1e-9
        np.testing.assert_allclose((np.exp(lw)*z['post']).sum(axis=1),a['post'][sample],atol=1e-10);np.testing.assert_array_equal(chi<=25,a['support'][sample]);invariance.append(dict(system=s,N=len(sample),means_error=err,weight_error=we,color_chi_error=ce,support_unchanged=True,post_mass_unchanged=True))
    # No fitted coefficient, reference residual or QRF outcome enters this split.
    def key(i):return hashlib.sha256(f"{C['seed']}:{int(t['ls_id'][i])}".encode()).hexdigest()
    gr=t['m'][:,0]-t['m'][:,1];edges=C['fold_color_edges']
    for lo,hi in zip(edges[:-1],edges[1:]):
        ii=np.flatnonzero(eligible&(gr>=lo)&(gr<hi));ii=np.array(sorted(ii,key=key),dtype=int);fold[ii]=np.arange(len(ii))%2
    assert (fold[eligible]>=0).all()
    for i in np.flatnonzero((t['system']=='M5')&~eligible):fold[i]=int(key(i),16)%2
    for test in (0,1):
        train=eligible&(fold!=test);ii=np.flatnonzero(train);bounds.append(dict(prediction_fold=test,train_N=len(ii),test_N=int(sum(eligible&(fold==test))),gr_min=float(gr[ii].min()),gr_max=float(gr[ii].max()),training_ids=[int(v) for v in t['ls_id'][ii]],test_ids=[int(v) for v in t['ls_id'][eligible&(fold==test)]]))
    ii=np.flatnonzero(eligible);bounds.append(dict(prediction_fold='external',train_N=len(ii),gr_min=float(gr[ii].min()),gr_max=float(gr[ii].max()),training_ids=[int(v) for v in t['ls_id'][ii]]));t.update(fold=fold,calibration_eligible=eligible)
    rng=np.random.default_rng(C['seed']+2)
    for s in C['systems']:
        sigma=np.std(5*np.log10(t['scale_'+s]),ddof=1);t['shift_'+s]=rng.normal(0,sigma,C['bootstrap']);t['reference_sigma_'+s]=np.array(sigma)
    npz(d/'roster.npz',**t)
    csv_rows(d/'identities.csv',[dict(TYPE=t['TYPE'][i],ls_id=int(t['ls_id'][i]),source_id=int(t['source_id'][i]),system=t['system'][i],native=int(t['native'][i]),phase=t['phase'][i],v5=bool(t['v5'][i]),fold=int(fold[i]),calibration_eligible=bool(eligible[i])) for i in range(933)])
    save_json(d/'fit_design.json',bounds);csv_rows(d/'actual_gray_invariance.csv',invariance);complete(d,inputs,registered_before_fit=True,parent=933,eligible_M5=int(eligible.sum()));print('Sealed',933,'M5 calibration',eligible.sum(),flush=True)


if __name__=='__main__':main()
