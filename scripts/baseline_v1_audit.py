#!/usr/bin/env python3
"""Every evolved/transition reference, regardless of v0 ambiguity flag."""
from baseline_v1_common import *
from photdis.stage1 import require_compute

def main():
    require_compute();t=load(OLD/'references.npz');result={}
    for name in ('A','B','C'):
        p=load(OLD/f'{name}_evaluation.npz');rows=p['rows'];centers=p['centers'];layers={}
        for source in ('chosen','Gaia','specdis','spdist'):
            mu=truth_mu(t,source)[rows];mg=t['g0'][rows]-mu
            mass=np.sum(p['pdf']*(abs(p['centers'][None,:]-mg[:,None])<=.5),axis=1)
            cdf=np.sum(p['pdf']*(p['centers'][None,:]<=mg[:,None]),axis=1)
            bad=abs(p['mu'][:,2]-mu)>1
            for label in (0,1,2):
                for split in (1,2):
                    use=(t['class_label'][rows]==label)&(t['split'][rows]==split)&np.isfinite(mu)
                    if not use.any():continue
                    layers[f'{source}:class{label}:split{split}']=dict(n=int(use.sum()),
                        true_Mg_mass_pm0p5_quantiles=np.quantile(mass[use],[.1,.5,.9]).tolist(),
                        fraction_mass_below_0p05=float(np.mean(mass[use]<.05)),
                        reference_cdf_quantiles=np.quantile(cdf[use],[.1,.5,.9]).tolist(),
                        catastrophic=int((use&bad).sum()),
                        catastrophic_supported_unambiguous=int((use&bad&p['supported']&~p['ambiguous']).sum()))
        result[name]=layers;print('Audited complete v0',name,flush=True)
    write_json(OUT/'v0_branch_audit.json',result)
if __name__=='__main__':main()
