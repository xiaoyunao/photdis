#!/usr/bin/env python3
"""Seal unchanged identities and native model quadratures before inference."""
from seq7_common import *
from seq7_math import imf_integral
from photdis.stage1 import require_compute
from sv18_common import index_join
from sky6_ridge import model_tables
from stream_sequence import polyline


def build(model,side,q,imf=False,extended=False):
    m=model[side];p=model['phase'];valid=(m[:,1]>=3)&(m[:,1]<=8)&np.isin(p,(0,2) if extended else (0,))
    if side=='north':valid&=model['north_support']
    left=np.flatnonzero(valid[:-1]&valid[1:]);right=left+1
    if extended:left=left[(p[left]==0)|((p[left]==2)&(p[left+1]==2))];right=left+1
    u=(np.arange(q)+.5)/q;nodes=(m[left,None]*(1-u[None,:,None])+m[right,None]*u[None,:,None]).reshape(-1,3)
    post=np.repeat(p[right]==2,q);eep=np.repeat(model.get('EEP',np.arange(len(m)))[left],q)
    if imf:
        mass=model['mass'];assert (np.diff(mass)>0).all()
        boundaries=mass[left,None]+(mass[right]-mass[left])[:,None]*np.arange(q+1)[None]/q
        raw=imf_integral(boundaries[:,:-1],boundaries[:,1:]).ravel()
        np.testing.assert_allclose(raw.sum(),imf_integral(mass[left],mass[right]).sum(),rtol=2e-13)
    else:raw=np.repeat(np.linalg.norm(m[right]-m[left],axis=1)/q,q)
    good=raw>1e-12;assert good.all();weights=raw/raw.sum()
    return dict(nodes=nodes,prior=weights,post=post,EEP=eep,left=left,right=right,raw=raw)


def main():
    require_compute();d=folder('prepare');inputs={}
    t=read(OLD/'ridge_inputs',('roster.npz','TYPE','ls_id','source_id','RA','DEC','system','native','mask','m','err','r','q','ref','sigma_ref','original_phase'),inputs)
    keep=~t['mask'];t={k:v[keep] for k,v in t.items()};assert (t['TYPE']=='PSF').all() and len(np.unique(t['ls_id']))==len(t['ls_id'])
    np.testing.assert_array_equal([sum(t['system']==s) for s in C['systems']],C['parent_N'])
    old=read(STREAM/'sequence',('predictions.npz','ls_id','mask','new_common','new','calibrated','reference_phase'),inputs)
    j=index_join(t['ls_id'],old['ls_id']);found=j>=0;t['v5']=found&old['new_common'][np.maximum(j,0)]&~old['mask'][np.maximum(j,0)]
    for field,out in [('new','legacy_new'),('calibrated','legacy_calibrated')]:
        t[out]=np.full((len(j),5),np.nan);t[out][found]=old[field][j[found]]
    refmu=10+5*np.log10(t['ref']);gr=t['m'][:,0]-t['m'][:,1];mr=t['m'][:,1]-refmu
    phase=np.full(len(j),'other',dtype='U12');phase[(gr>=.3)&(gr<=.85)&(mr>=4.5)&(mr<=8)]='lower_MS';phase[(gr>=.15)&(gr<.3)&(mr>=3)&(mr<4.5)]='turnoff';t['phase']=phase
    np.testing.assert_array_equal(phase[t['v5']],old['reference_phase'][j][t['v5']])
    b=read(STREAM/'sequence',('shared_bootstrap.npz','ls_id','multiplicity',*['scale_'+s for s in C['systems']]),inputs)
    j=index_join(t['ls_id'],b['ls_id']);t['v5_mult']=np.zeros((256,len(j)),'i2');found=j>=0;t['v5_mult'][:,found]=b['multiplicity'][:,j[found]]
    rng=np.random.default_rng(C['seed']);mult=np.zeros((256,len(j)),'i2')
    for s in C['systems']:
        ii=np.flatnonzero(t['system']==s);mult[:,ii]=rng.multinomial(len(ii),np.full(len(ii),1/len(ii)),size=256)
        t['scale_'+s]=b['scale_'+s]
        for p,n in [('lower_MS',{'M13':166,'M5':397,'M30':32}[s]),('turnoff',{'M13':37,'M5':85,'M30':32}[s])]:assert sum((t['system']==s)&t['v5']&(phase==p))==n
    t['mult']=mult;t['qrf_mag']=np.zeros((len(j),3));t['qrf_err']=np.zeros((len(j),3));t['qrf_identity']=np.zeros(len(j),'i8')
    for s in C['systems']:
        parent=ROOT/('data/v3_14/audit/M5' if s=='M5' else 'data/v3_13/clusters/'+s)
        raw=read(parent,('native_join.npz' if s=='M5' else 'sources.npz','TYPE','ls_id','mag_grz','error_grz','identity'),inputs)
        ii=np.flatnonzero(t['system']==s);jj=index_join(t['ls_id'][ii],raw['ls_id']);assert (jj>=0).all() and (raw['TYPE'][jj]=='PSF').all()
        assert abs(t['m'][ii]-raw['mag_grz'][jj]).max()<2e-5
        t['qrf_mag'][ii]=raw['mag_grz'][jj];t['qrf_err'][ii]=raw['error_grz'][jj];t['qrf_identity'][ii]=raw['identity'][jj]
    npz(d/'roster.npz',**t)
    template=read(HALO/'native',('template.npz','south','north','phase','north_support','hull','beta','scatter'),inputs);models=model_tables(template,inputs)
    for key,model in models.items():
        if key=='FOS10':continue
        path=ROOT/'configs/v6_models'/(key+'.txt');cols=next(l.lstrip('# ').split() for l in path.read_text().splitlines() if l.startswith('# EEP '));x=np.loadtxt(path)
        model.update(mass=x[:,cols.index('initial_mass')],EEP=x[:,cols.index('EEP')]);assert (np.diff(model['mass'])>0).all()
    audit=[];curves={}
    for s in C['systems']:
        side='north' if s=='M13' else 'south'
        for method in METHODS[:-1]:
            key={'A':'FOS10','B':'version_control'}.get(method,'matched_'+s);model=models[key]
            for q in (4,8,16,32):
                z=build(model,side,q,method in ('D','E'),method=='E');npz(d/f'{s}_{method}_{q}.npz',**z)
                if method=='A' and q==4:
                    nodes,w=polyline(template,side);np.testing.assert_array_equal(z['nodes'],nodes);np.testing.assert_array_equal(z['prior'],w)
                if q==4:
                    left=z['left'];right=z['right'];phase2=(model['phase'][right]==2);mass=model.get('mass');eep=model.get('EEP',np.arange(len(model['phase'])))
                    audit.append(dict(system=s,method=method,locus=key,side=side,segments=len(left),nodes=len(z['nodes']),sum_prior=float(z['prior'].sum()),post_prior=float(z['prior'][z['post']].sum()),raw_measure=float(z['raw'].sum()),EEP_post_min=float(eep[left[phase2]].min()) if phase2.any() else None,EEP_post_max=float(eep[right[phase2]].max()) if phase2.any() else None,mass_post_min=float(mass[left[phase2]].min()) if phase2.any() else None,mass_post_max=float(mass[right[phase2]].max()) if phase2.any() else None,Mr_min=float(z['nodes'][:,1].min()),Mr_max=float(z['nodes'][:,1].max()),nonmonotonic_mass=0 if mass is not None else 'not used'))
            curves[s+'_'+method]=dict(mag=model[side].tolist(),phase=model['phase'].tolist(),supported=model['north_support'].tolist() if side=='north' else np.ones(len(model['phase']),bool).tolist())
    csv_rows(d/'model_measures.csv',audit);save_json(d/'curves.json',curves)
    csv_rows(d/'admission.csv',[dict(system=s,parent=int(sum(t['system']==s)),v5=int(sum((t['system']==s)&t['v5'])),nonPSF=0,missingTYPE=0,selection='frozen role-excluded low-crowding unmasked native PSFs, no new distance/model filter') for s in C['systems']])
    csv_rows(d/'identities.csv',[dict(system=t['system'][i],ls_id=int(t['ls_id'][i]),source_id=int(t['source_id'][i]),TYPE=t['TYPE'][i],native=int(t['native'][i]),v5=bool(t['v5'][i]),phase=phase[i]) for i in range(len(j))])
    complete(d,inputs,registered_before_predictions=True,no_reference_in_prediction=True,parent=len(j));print('Sealed roster/models',len(j),flush=True)


if __name__=='__main__':main()
