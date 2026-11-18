#!/usr/bin/env python3
"""Condition frozen forest leaf histograms on known external luminosity classes."""
import copy
import joblib
from photdis.stage1 import require_compute
from photdis.baseline_v0 import HistogramQRF
from photdis.v3_3 import frozen_qrf
from sv18_common import *
from sv18_benchmark import training
from v3_3_catalog import read_frozen
from v3_9_catalog import from_north


class ConditionalForest(HistogramQRF):
    def __init__(self,base,t,code):
        self.__dict__=copy.copy(base.__dict__)
        self.base=base
        self.conditional=[]
        x=base.feature_matrix(t['m'])
        bins=np.searchsorted(base.edges,t['y'],side='right')-1
        for tree,mapping,h in zip(base.forest.estimators_,base.maps,base.tables):
            leaf=mapping[tree.apply(x)]
            den=np.bincount(leaf,minlength=len(h))
            j=t['class_label']==code
            counts=np.bincount(leaf[j]*h.shape[1]+bins[j],minlength=h.size).reshape(h.shape).astype(float)
            counts=np.divide(counts,den[:,None],out=np.zeros_like(counts),where=den[:,None]>0)
            self.conditional.append(counts)
        self.fallback_n=0;self.evaluation_n=0

    def distribution(self,x):
        joint=np.zeros((len(x),len(self.centers)))
        for tree,mapping,h in zip(self.forest.estimators_,self.maps,self.conditional):
            joint+=h[mapping[tree.apply(x)]]
        total=joint.sum(axis=1)
        good=total>0
        joint[good]/=total[good,None]
        if (~good).any():
            joint[~good]=self.base.distribution(x[~good])[0]
        self.fallback_n+=int((~good).sum());self.evaluation_n+=len(good)
        return joint,np.zeros((len(x),4))


def main():
    require_compute()
    d=out('oracle');inputs={}
    config=CONFIG.with_name('science_validation_20261118_oracle.json')
    if not config.exists():
        config=ROOT/'configs/science_validation_20261118_oracle.json'
    cfg=json.loads(config.read_text());inputs['oracle_config']=sha(config)
    t=cached(inputs,('TYPE','ls_id','source_id','RA','DEC','region','heldout','state','m','err','pop',
        'q','specdis_stage','specdis_d','spdist_stage','spdist_d'))
    select=t['heldout']&(t['state']==3)&np.isin(t['pop'],(1,2,3,4,5))
    t={k:v[select] for k,v in t.items()}
    rng=np.random.default_rng(cfg['seed'])
    rows=[];predictions=[]
    for side,north in (('south',False),('north',True)):
        train,offset=training(side,inputs)
        if north:
            model,hh=from_north();base=model['qrf'];inputs['north_model']=hh
        else:
            f=read_frozen()['models']['qrf'];base=joblib.load(verified(ROOT/f['path'],f['sha256']))
            inputs['south_model']=f['sha256']
        for name,codes,code in (('dwarf',(1,2,3),0),('intermediate',(4,),2),('giant',(5,),1)):
            eligible=((t['region']==3)==north)&np.isin(t['pop'],codes)
            eligible&=(t['specdis_stage']==3)|(t['spdist_stage']==3)
            ix=np.flatnonzero(eligible)
            if len(ix)>cfg['max_per_native_class']:
                ix=np.sort(rng.choice(ix,cfg['max_per_native_class'],replace=False))
            if not len(ix):continue
            oracle=ConditionalForest(base,train,code)
            q=frozen_qrf(oracle,t['m'][ix],t['err'][ix],t['source_id'][ix])['distance']*10**(offset/5)
            for ref in ('specdis','spdist'):
                keep=t[ref+'_stage'][ix]==3
                for method,value in (('Frozen',t['q'][ix]),('Known class',q)):
                    rows.append(dict(native=side,population=name,reference=ref,method=method,
                        fallback_fraction=oracle.fallback_n/max(oracle.evaluation_n,1),
                        **block_metrics(value[keep],t[ref+'_d'][ix][keep],sky(t)[ix][keep])))
            predictions.append(dict(ls_id=t['ls_id'][ix],TYPE=t['TYPE'][ix],oracle=q,original=t['q'][ix]))
    csv_rows(d/'class_oracle.csv',rows)
    from v3_9_references import join
    atomic_npz(d/'oracle_rows.npz',**join(predictions))
    complete(d,inputs,scope='known-class information upper bound; original leaves and labels, no tree refit')


if __name__=='__main__':
    main()
