#!/usr/bin/env python3
"""Build the optical parent and broad references in an allocation; no model scores."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import fitsio
from photdis.stage1 import require_compute,galactic
from photdis.class_evidence import desi_class_evidence_v3
from giant_vac_inventory import SPEC,SP,exact_pair_join
from kgiant_distance_inventory import write_json,sha
CFG=json.load(open('configs/baseline_v0.json'));OUT=Path(CFG['output_dir'])
COLS=['ls_id','source_id','ra','dec','ebv','g0','r0','z0','w10','w20','r','sigma_g','sigma_r','sigma_z','sigma_w1','sigma_w2','grz_base','grz_clean','w1_clean','w2_clean','snr_w1','snr_w2','astro_quality','zpt_valid','anchor_astrometry','contaminant_candidate','p_corr','gaia_parallax_error','gaia_bp_rp','desi_match','desi_match_ambiguous','desi_object_id','desi_snr','desi_spectype','desi_quality_reason','desi_teff','desi_teff_err','desi_logg','desi_logg_err','desi_feh','desi_feh_err']

def enrich(t):
    n=len(t['ls_id']);e=desi_class_evidence_v3(t)
    text=e['class_evidence'].astype('U')
    t['class_label']=np.where(text=='conservative_dwarf',0,np.where(text=='conservative_giant',1,np.where(text=='other_star_or_transition',2,-1))).astype('i1')
    t['nonstellar_risk']=e['redrock_nonstellar_evidence']|t['contaminant_candidate']
    t['release']=(t['ls_id']>>40).astype('i4')
    t['identity']=np.where(t['source_id']>0,t['source_id'],-t['ls_id'])
    t['B']=t['w1_clean']&(t['snr_w1']>=5)
    t['C']=t['B']&t['w2_clean']&(t['snr_w2']>=5)
    t['optical_error']=np.maximum.reduce([t['sigma_g'],t['sigma_r'],t['sigma_z']])
    return t

def main():
    require_compute();OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'prepared.json').exists():raise FileExistsError('Prepared baseline is frozen')
    fields=[e for r in CFG['registries'] for e in json.load(open(r))['fields']]
    labels=[];keys=[];counts=[]
    for i,e in enumerate(fields):
        task=json.load(open(e['config']));p=Path(task['output_dir'])/'field_diagnostics.fits';parts=[]
        with fitsio.FITS(str(p)) as h:
            for start in range(0,h[1].get_nrows(),100000):
                raw=h[1].read(rows=np.arange(start,min(start+100000,h[1].get_nrows())),columns=COLS)
                b=galactic(raw['ra'],raw['dec'])[1];keep=raw['grz_base']&raw['grz_clean']&(abs(b)>30)&(raw['ebv']>=0)&(raw['ebv']<.03)
                parts.append(raw[keep])
        raw=np.concatenate(parts);t=enrich({k:raw[k] for k in raw.dtype.names});n=len(raw);del raw,parts
        t['field']=np.full(n,i,dtype='i2');keys.append(t['identity']);np.savez(OUT/f'parent_{i:02}.npz',**t)
        ref=t['anchor_astrometry']|t['desi_match']|(t['astro_quality']&t['zpt_valid'])
        labels.append({k:v[ref] for k,v in t.items()})
        counts.append(dict(index=i,name=task['name'],config=e['config'],input=str(p),input_size=p.stat().st_size,parent_records=n))
        print('parent',i,n,'reference candidates',int(ref.sum()),flush=True)
    # Deterministic first field/first row owns each identity, independent of labels.
    ids=np.concatenate(keys);_,first=np.unique(ids,return_index=True);unique=np.zeros(len(ids),bool);unique[first]=True
    offset=0;owner_ids=[]
    for i,k in enumerate(keys):
        own=unique[offset:offset+len(k)];np.save(OUT/f'owner_{i:02}.npy',own);owner_ids.append(k[own]);counts[i]['unique_parent']=int(own.sum());offset+=len(k)
    t={k:np.concatenate([r[k] for r in labels]) for k in labels[0]};owner={int(s):i for i,a in enumerate(owner_ids) for s in a if s>0}
    # Reference subset can be smaller than optical parent, never reassign its owner.
    take=np.array([owner.get(int(s),int(f))==f for s,f in zip(t['identity'],t['field'])]);_,idx=np.unique(t['identity'][take],return_index=True);idx=np.flatnonzero(take)[idx]
    t={k:v[idx] for k,v in t.items()};n=len(idx)
    target=np.zeros(n,np.int64);ok=t['desi_match']&~t['desi_match_ambiguous']
    target[ok]=np.char.strip(t['desi_object_id'][ok].astype('U')).astype('i8')
    for kind,path,ext,cols in [('specdis',SPEC,'SPECDIS',['TARGETID','SOURCE_ID','DIST','DISTERR','BINARY_FLAG']),('spdist',SP,'SPDIST',['TARGETID','SOURCE_ID','FLAG_GOOD','QFLAG_INPUT','G0','E_GMAG','MG_16','MG_50','MG_84'])]:
        cat=fitsio.read(str(path),ext=ext,columns=cols);join=exact_pair_join(target,t['source_id'],cat);valid=join>=0;v=cat[np.maximum(join,0)]
        if kind=='specdis':
            d=v['DIST'].astype(float);de=v['DISTERR'].astype(float)
            valid&=(d>0)&(de>0)&(de<d)&np.isfinite(d)&np.isfinite(de)&(v['BINARY_FLAG']==1)
            sigma=5/np.log(10)*np.sqrt(np.log1p((de/d)**2))
        else:
            g=v['G0'];mg=v['MG_50'];lo=v['MG_16'];hi=v['MG_84'];em=v['E_GMAG']
            valid&=v['FLAG_GOOD']&v['QFLAG_INPUT']&np.isfinite(g)&np.isfinite(mg)&np.isfinite(lo)&np.isfinite(hi)&np.isfinite(em)&(em>=0)&(g>10)&(g<30)&(mg>-8)&(mg<15)&(hi>=lo)&(hi-lo<4)
            d=10.**((g-mg-10)/5);sigma=np.sqrt(em**2+((hi-lo)/2)**2+.167**2)
        t[kind+'_valid']=valid;t[kind+'_distance']=np.where(valid,d,np.nan);t[kind+'_sigma_mu']=np.where(valid,sigma,np.nan)
        del cat,v
    ga=t['anchor_astrometry']&~t['nonstellar_risk']&(t['p_corr']>0)
    sp=t['specdis_valid']&~t['nonstellar_risk']
    t['label_source']=np.where(ga,1,np.where(sp,2,0)).astype('i1');t['ref_distance']=np.where(ga,1/np.maximum(t['p_corr'],1e-10),t['specdis_distance'])
    t['Mg']=t['g0']-10-5*np.log10(t['ref_distance']);t['sigma_mu']=np.where(ga,5/np.log(10)*t['gaia_parallax_error']/np.maximum(t['p_corr'],1e-10),t['specdis_sigma_mu'])
    # Same DESI target with conflicting Gaia IDs cannot seed different label objects.
    q=target>0;u,c=np.unique(target[q],return_counts=True);bad=u[c>1];t['identity_conflict']=np.isin(target,bad)&q
    t['fit_eligible']=(t['label_source']>0)&~t['identity_conflict']&(t['optical_error']<.03)&np.isfinite(t['Mg'])
    split=np.full(n,3,'i1')
    for k,fields0 in enumerate([CFG['train_fields'],CFG['cal_fields'],CFG['test_fields']]):split[np.isin(t['field'],fields0)]=k
    t['split']=split;assert len(np.unique(t['identity']))==n
    assert np.all((t['release'][split<3]==11010))
    np.savez(OUT/'references.npz',**t)
    result=dict(fields=counts,n_reference=n,n_unique_parent=len(first),n_parent_records=len(ids),label_counts={str(k):int((t['label_source']==k).sum()) for k in (0,1,2)},split_counts={str(k):int((split==k).sum()) for k in range(4)},train_core_counts={tier:int((t['fit_eligible']&(split==0)&(True if tier=='A' else t[tier])).sum()) for tier in ('A','B','C')},identity_conflicts=int(t['identity_conflict'].sum()),config_sha256=sha('configs/baseline_v0.json'),code_sha256=sha(__file__))
    write_json(OUT/'prepared.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
