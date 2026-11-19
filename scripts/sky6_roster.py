#!/usr/bin/env python3
"""Freeze all available low-crowding members before any model-support cut."""
from photdis.stage1 import require_compute
from sky6_common import *
from stream_sequence import SIGMA


def main():
    require_compute();d=folder('ridge_inputs');inputs={};pieces=[];rows=[]
    for name in C['ridge']['systems']:
        z=read(ROOT/f'data/v3_14/audit/{name}',('native_join.npz','TYPE','ls_id','source_id','RA','DEC','target','member','training_or_calibration','low_crowding','cluster_mask','g0','r0','z0','r','error_grz','qrf_kpc','phase_box'),inputs)
        keep=np.ones(len(z['ls_id']),bool)
        for label,cut in [('verified_PSF',z['TYPE']=='PSF'),('target',z['target']),('external_member',z['member']),('exclude_fit_cal',~z['training_or_calibration']),('low_crowding',z['low_crowding']),('finite_photometry',np.isfinite(np.column_stack((z['g0'],z['r0'],z['z0']))).all(axis=1)&np.isfinite(z['error_grz']).all(axis=1)&(z['error_grz']>0).all(axis=1))]:
            before=int(keep.sum());keep&=cut;rows.append(dict(system=name,cut=label,before=before,after=int(keep.sum()),excluded=before-int(keep.sum())))
        old=read(STREAM/'sequence',('predictions.npz','TYPE','ls_id','system','ref'),inputs);ref=float(np.median(old['ref'][old['system']==name]));n=int(keep.sum())
        pieces.append(dict(TYPE=z['TYPE'][keep],ls_id=z['ls_id'][keep],source_id=z['source_id'][keep],RA=z['RA'][keep],DEC=z['DEC'][keep],
            system=np.full(n,name),native=np.full(n,int(name=='M13'),'u1'),mask=z['cluster_mask'][keep],m=np.column_stack((z['g0'],z['r0'],z['z0']))[keep],err=z['error_grz'][keep],r=z['r'][keep],q=z['qrf_kpc'][keep],ref=np.full(n,ref),sigma_ref=np.full(n,SIGMA[name]),original_phase=z['phase_box'][keep]))
    t={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};order=np.argsort(t['ls_id']);t={k:v[order] for k,v in t.items()}
    assert len(np.unique(t['ls_id']))==len(t['ls_id']) and (t['TYPE']=='PSF').all()
    npz(d/'roster.npz',**t);csv_rows(d/'admission.csv',rows)
    complete(d,inputs,scope='all available frozen external low-crowding member photometry, before QRF/theory/Mr support; primary unmasked and masked supplementary; earlier native admission and membership limits remain')


if __name__=='__main__':main()
