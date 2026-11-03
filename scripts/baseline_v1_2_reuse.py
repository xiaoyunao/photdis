#!/usr/bin/env python3
"""Reuse frozen v1.1 mixtures exactly; audit and retain recomputed cache versions."""
from baseline_v1_2_run import *
import shutil


def main():
    require_compute();backup=OUT/'pre_frozen_reuse'
    if backup.exists():raise FileExistsError('Reuse already performed; inspect saved audit')
    backup.mkdir();audit={};t=load(OLD/'references.npz')
    for name in MODELS:
        for kind in (('reference','parent','parallax') if name in 'ABC' else ('reference',)):
            cache=load(OUT/f'{kind}_{name}_cache.npz');table=t if kind=='reference' else load(PREV/f'{kind}_sample.npz')
            for v in (('natural','balanced','evidence') if kind=='reference' else ('evidence',)):
                target=OUT/f'{kind}_{name}_{v}.npz';p=load(target)
                oldpath=PREV/(f'{name}_{v}_evaluation.npz' if kind=='reference' else f'{kind}_{name}_{v}.npz')
                old=load(oldpath);assert np.array_equal(p['rows'],old['rows'])
                cdf=float(np.max(abs(np.cumsum(p['pdf'],axis=1,dtype=float)-np.cumsum(old['pdf'],axis=1,dtype=float))))
                diff=abs(p['mu']-old['mu'])
                # Sub-microprobability changes can move inverse CDFs in empty gaps.
                assert cdf<1e-6,(name,kind,v,cdf)
                audit[f'{kind}_{name}_{v}']=dict(cdf_max_difference=cdf,quantile_max_difference=float(diff.max()),
                    quantile_p99_difference=float(np.quantile(diff,.99)),median_max_difference=float(diff[:,2].max()),
                    frozen_source_sha256=sha(oldpath),recomputed_sha256=sha(target))
                for key in ('pdf','mu','weights','supported','branch_support'):
                    if key in old:p[key]=old[key]
                main,index=main_component(cache['branch_pdf'],p['weights'])
                p['main_index']=index;p['main_mu']=distance_summary(main,p['edges'],table['g0'][p['rows']])
                shutil.move(str(target),str(backup/target.name));np.savez_compressed(target,**p)
                assert np.array_equal(p['mu'],old['mu']) and np.array_equal(p['pdf'],old['pdf'])
    write_json(OUT/'frozen_reuse_audit.json',audit)
    print('Frozen mixtures reused exactly:',len(audit),flush=True)
if __name__=='__main__':main()
