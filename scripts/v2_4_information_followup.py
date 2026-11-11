#!/usr/bin/env python3
"""Cal-only prior support and explicitly matched class-composition stress."""
import shutil
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,csv_rows,save_json
from v2_4_audit import DEST,TABLES
from photdis.stage1 import require_compute
from photdis.science_v2_4 import conditional_prior,weighted_scores,cluster_mean_interval


def main():
    require_compute();t=load(ROOT/'data/v2_1/information_eval.npz');p=load(ROOT/'data/v2_1/information_probabilities.npz')
    assert (t['TYPE']=='PSF').all() and np.array_equal(t['identity'],p['identity'])
    archive=DEST/'information_initial';archive.mkdir(exist_ok=True)
    for name in ('information_population','information_reliability','information_uncertainty'):
        if not (archive/(name+'.csv')).exists():shutil.copy2(TABLES/(name+'.csv'),archive/(name+'.csv'))
    if not (archive/'information_complete.json').exists():shutil.copy2(DEST/'information_complete.json',archive/'information_complete.json')
    cal=t['split']==1;dev=t['split']==2;y=t['class_label'].astype(int);rbin=np.digitize(t['r'],[18,20,22,25])
    prior,rprior=conditional_prior(y[cal],rbin[cal],rbin)
    sky=(t['cell']//18//6)*2+((t['cell']%18)>=9);cells=sky*10+rbin;coarse=rprior.copy()
    for cell in np.unique(cells):
        use=cal&(cells==cell);target=cells==cell
        coarse[target]=(np.bincount(y[use],minlength=3)+30*rprior[target])/(use.sum()+30)
    global_prior=np.tile(prior,(len(y),1));results=[];gains=[];reliability=[]
    groups={'all':np.ones(len(y),bool),'r_lt18':t['r']<18,'r18_20':(t['r']>=18)&(t['r']<20),'r20plus':t['r']>=20}
    for bands in ('grz','griz','grzW1','grzW1W2'):
        methods={'cal_global_prior':global_prior,'cal_r_only_prior':rprior,'cal_r_sky_prior':coarse,
            'frozen_partitions':p[bands+'_frozen_partitions_same_labels'],'fixed_color_probe':p[bands+'_fixed_forest']}
        for group,mask in groups.items():
            u=dev&mask
            if not u.any():continue
            count=np.bincount(y[u],minlength=3)
            for population in ('natural_reference','equal_class_label_shift'):
                weights=np.ones(u.sum()) if population=='natural_reference' else 1/np.maximum(count[y[u]],1)
                def adapt(prob):
                    q=prob[u].copy()
                    if population=='equal_class_label_shift':q/=prior;q/=q.sum(axis=1)[:,None]
                    return q
                base=weighted_scores(adapt(coarse),y[u],weights)
                for method,prob in methods.items():
                    score=weighted_scores(adapt(prob),y[u],weights)
                    results.append(dict(bands=bands,group=group,population=population,method=method,n=int(u.sum()),**score,gain_r_sky=score['log_score']-base['log_score'],dwarf=int(count[0]),evolved=int(count[1]),other=int(count[2])))
                    if population=='natural_reference':
                        diff=np.log(np.maximum(prob[u,y[u]],1e-12))-np.log(np.maximum(coarse[u,y[u]],1e-12))
                        gains.append(dict(bands=bands,group=group,method=method,**cluster_mean_interval(diff,t['cell'][u])))
                        for c in range(3):
                            bins=np.minimum((prob[u,c]*10).astype(int),9)
                            for b in range(10):
                                v=bins==b;reliability.append(dict(bands=bands,group=group,method=method,class_label=c,bin=b,n=int(v.sum()),mean_probability=float(prob[u,c][v].mean()) if v.any() else None,frequency=float((y[u][v]==c).mean()) if v.any() else None))
    csv_rows(TABLES/'information_population.csv',results);csv_rows(TABLES/'information_uncertainty.csv',gains);csv_rows(TABLES/'information_reliability.csv',reliability)
    meta=json.loads((archive/'information_complete.json').read_text());meta.update(coarse_unseen_fraction=float(np.mean(~np.isin(cells[dev],cells[cal]))),
        fine_unseen_fraction=float(np.mean(~np.isin(t['cell'][dev],t['cell'][cal]))),r_unseen_fraction=float(np.mean(~np.isin(rbin[dev],rbin[cal]))),
        sky='60degreeRA×north/southDec; empty cells fallback r-only; fixed30shrinkage',
        class_stress='All methods retargeted by uniform/global-cal class prior; label-shift assumption explicit; original raw macro scores archived',
        amendment_sha256=sha(ROOT/'docs/v2_4_information_amendment.md'),source_sha256=sha(ROOT/'scripts/v2_4_information_followup.py'),job=os.getenv('SLURM_JOB_ID'))
    save_json(DEST/'information_complete.json',meta);print('INFORMATION SUPPORT CORRECTED',meta['coarse_unseen_fraction'],flush=True)


if __name__=='__main__':main()
