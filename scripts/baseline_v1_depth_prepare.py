#!/usr/bin/env python3
"""Freeze flux-level templates and actual test-field noise donors before scoring."""
from baseline_v1_common import *
from photdis.stage1 import require_compute
from photdis.noise import simulate_shared_flux
from photdis.photometry import derive_photometry
from cold_v4_policy_noise import raw_for_ids

def main():
    require_compute();OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'depth_input.npz').exists():raise FileExistsError('Depth simulation exists')
    t=load(OLD/'references.npz');fields=json.load(open(OLD/'prepared.json'))['fields']
    rng=np.random.default_rng(CFG['seed']+77);parts=[];census=[]
    for field in (5,6,10):
        ix=[]
        for c in (-1,0,1,2):
            pool=np.flatnonzero((t['field']==field)&t['fit_eligible']&(t['class_label']==c))
            chosen=rng.choice(pool,min(len(pool),CFG['depth_templates_per_class_field']),replace=False)
            ix.extend(chosen);census.append(dict(field=field,kind=c,n_available=len(pool),n_chosen=len(chosen)))
        ix=np.sort(ix);path=fields[field]['config'];raw=raw_for_ids(t['ls_id'][ix],path)
        template=np.empty(len(ix),dtype=raw.dtype.descr+[('r0','f8')])
        for k in raw.dtype.names:template[k]=raw[k]
        template['r0']=t['r0'][ix]
        parent=load(OLD/f'parent_{field:02}.npz');targets=[];metadata=[]
        source_index=np.repeat(np.arange(len(ix)),CFG['depth_repeats'])
        for depth,(lo,hi) in enumerate(CFG['depth_r_bins']):
            pool=np.flatnonzero((parent['r']>=lo)&(parent['r']<hi));donor=rng.choice(pool,len(source_index),replace=True)
            targets.extend(parent['ls_id'][donor]);metadata.append((depth,donor))
        unique,inverse=np.unique(targets,return_inverse=True);donor_raw=raw_for_ids(unique,path)[inverse]
        for d,(depth,donor) in enumerate(metadata):
            target=donor_raw[d*len(source_index):(d+1)*len(source_index)]
            simulated,shift,ratio=simulate_shared_flux(template[source_index],target,rng)
            phot=derive_photometry(simulated,**json.load(open('configs/stage1.json'))['photometry'])
            base=ix[source_index];p={k:np.asarray(phot[k]) for k in [*BANDS,*ERR,'r','grz_base','grz_clean','w1_clean','w2_clean','snr_w1','snr_w2']}
            p.update(identity=t['identity'][base],source_row=base,field=t['field'][base],class_label=t['class_label'][base],
                label_source=t['label_source'][base],target_depth=np.full(len(base),depth),
                truth_mu=truth_mu(t,'chosen')[base]+shift,shift=shift,
                donor_ls_id=parent['ls_id'][donor],template_noise_ratio=ratio,
                template_wise_finite=np.isfinite(t['w10'][base])&np.isfinite(t['w20'][base]))
            p['B']=p['w1_clean']&(p['snr_w1']>=5);p['C']=p['B']&p['w2_clean']&(p['snr_w2']>=5)
            p['optical_error']=np.maximum.reduce([p['sigma_g'],p['sigma_r'],p['sigma_z']])
            p['accepted']=p['grz_base']&p['grz_clean'];parts.append(p)
        print('Frozen flux experiment field',field,len(ix),'templates',flush=True)
    data=concat(parts);data['simulation_id']=np.arange(len(data['identity']),dtype=np.int64)+1000000000
    np.savez(OUT/'depth_input.npz',**data)
    write_json(OUT/'depth_input.json',dict(templates=census,n_rows=len(data['identity']),
        n_sources=len(np.unique(data['identity'])),accepted=int(data['accepted'].sum()),
        input_sha256=sha(OUT/'depth_input.npz'),template_identity_sha256=identity_hash(np.unique(data['identity'])),
        note='Source identities retained for clustered resampling; simulation IDs distinguish noisy repeats. Existing parent already conditions morphology/sky; no new image morphology simulated.'))
if __name__=='__main__':main()
