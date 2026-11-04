#!/usr/bin/env python3
"""Refine Gaia numerical scores in new sidecars; retain every original result."""
from baseline_v1_4_experiments import *
from photdis.gaia_numerics_v1_4 import refined_gaia
_RT=None;_RM=None;_RP=None;_OPTICAL=False


def refine_chunk(path):
    output=_RP/path.name;marker=output.with_suffix('.json')
    if marker.exists():
        assert sha(output)==json.loads(marker.read_text())['sha256'];return output
    primary=load(path);rows=primary['rows'];good=_RT['parallax_eligible'][rows];rr=rows[good]
    if len(rr):
        m,e,ids=arrays(_RT,rr,_OPTICAL)
        if _OPTICAL:
            pred=_RM.predict(m,e,ids,draws=128);mu=distance_quantiles(m[:,0],pred['q_Meffective'])[0]
        else:pred=integrate(_RM,m,e,ids,draws=128)['wide'];mu=pred['mu']
        assert np.allclose(mu,primary['mu'][good],atol=1e-6,rtol=0)
        r=refined_gaia(pred['pdf'],_RM.edges,_RT['g0'][rr],_RT['p_corr'][rr],_RT['gaia_parallax_error'][rr])
        r['primary_log_difference']=r['full_log_density']-primary['full_log_density'][good]
        r['primary_pit_difference']=r['full_pit']-primary['full_pit'][good]
    else:
        r={k:np.empty(0) for k in ['full_log_density','full_pit','log_order','log_initial_difference','pit_initial_difference','pit_adaptive','primary_log_difference','primary_pit_difference']}
    atomic_npz(output,rows=rr,identity=_RT['identity'][rr],supported=primary['supported'][good],ambiguous=primary['ambiguous'][good],**r)
    write_json(marker,dict(sha256=sha(output),primary_sha256=sha(path)));return output


def main(stage):
    global _RT,_RM,_RP,_OPTICAL
    require_compute()
    completion=json.loads((OUT/'experiments'/(stage+'_complete.json')).read_text())
    assembly=json.loads((OUT/'assembly.json').read_text());reference_hash=assembly['reference_sha256']
    assert completion['reference_sha256']==reference_hash and sha(OUT/'references.npz')==reference_hash
    with np.load(OUT/'references.npz') as z:_RT={k:z[k] for k in KEYS}
    report={}
    for name in (['grz','griz'] if stage=='matched' else ['A','B','C']):
        prefix=stage+'_'+name;source=OUT/'experiments'/(prefix+'_distance')
        _OPTICAL=stage=='matched';modelpath=OUT/'experiments'/(prefix+'_distance.joblib')
        _RM=joblib.load(modelpath);_RP=OUT/'gaia_refined'/prefix;_RP.mkdir(parents=True,exist_ok=True)
        fingerprint=dict(reference_sha256=reference_hash,model_sha256=sha(modelpath),primary_score_sha256=sha(source/'scores.json'),
            code_sha256=sha(__file__),numerics_sha256=sha(ROOT/'src/photdis/gaia_numerics_v1_4.py'))
        freeze=_RP/'freeze.json'
        if freeze.exists():assert json.loads(freeze.read_text())==fingerprint
        else:write_json(freeze,fingerprint)
        paths=sorted(source.glob('[0-9]*.npz'));outputs=[];start=time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
            for j,p in enumerate(pool.map(refine_chunk,paths)):
                outputs.append(p)
                if j%25==0:print(prefix,'Gaia refine',j,'/',len(paths),'seconds',round(time.time()-start),flush=True)
        chunks=[load(p) for p in outputs];r={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        checks=dict(identity_aligned=bool(np.array_equal(r['identity'],_RT['identity'][r['rows']])),
            finite=bool(np.isfinite(r['full_log_density']).all() and np.isfinite(r['full_pit']).all()),
            pit_range=bool(np.all((r['full_pit']>=0)&(r['full_pit']<=1))),negative_retained=int((_RT['p_corr'][r['rows']]<0).sum()))
        assert checks['identity_aligned'] and checks['finite'] and checks['pit_range']
        groups_out={};t=subset(_RT,r['rows'])
        layers=groups(t);layers.update(supported=r['supported'],outside_support=~r['supported'],ambiguous=r['ambiguous'])
        for group,mask in layers.items():
            use=mask&(t['split']==2);pit=r['full_pit'][use];log=r['full_log_density'][use]
            groups_out[group]=dict(n=int(use.sum()),negative=int((use&(t['p_corr']<0)).sum()),
                mean_log_density=float(np.mean(log)) if len(log) else None,
                mean_pit=float(np.mean(pit)) if len(pit) else None,low05=float(np.mean(pit<.05)) if len(pit) else None,
                high95=float(np.mean(pit>.95)) if len(pit) else None,pit_histogram=np.histogram(pit,np.linspace(0,1,11))[0].tolist())
        delta=r['primary_log_difference'];pitdelta=r['primary_pit_difference']
        summary=dict(n=len(delta),log_abs_difference_max=float(np.max(abs(delta))) if len(delta) else 0,
            log_abs_difference_gt_1e3=int((abs(delta)>.001).sum()),log_difference_mean=float(np.mean(delta)) if len(delta) else 0,
            pit_abs_difference_max=float(np.max(abs(pitdelta))) if len(pitdelta) else 0,
            log_orders={str(k):int(v) for k,v in zip(*np.unique(r['log_order'],return_counts=True))},pit_adaptive=int(r['pit_adaptive'].sum()))
        atomic_npz(_RP/'evaluation.npz',**r)
        save_json(_RP/'scores.json',dict(fingerprint=fingerprint,checks=checks,numerical_change=summary,groups=groups_out,evaluation_sha256=sha(_RP/'evaluation.npz')))
        report[prefix]=summary;print(prefix,summary,flush=True)
    save_json(OUT/'gaia_refined'/(stage+'_complete.json'),dict(reference_sha256=reference_hash,models=report))


if __name__=='__main__':main(sys.argv[1])
