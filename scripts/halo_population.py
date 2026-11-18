#!/usr/bin/env python3
"""Trace score losses in the same actual CMD-filter reference population."""
import healpy as hp
import joblib
from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature
from halo_common import *
from halo_filtering import accepted,prepare_templates
from sv18_common import cached,index_join
from paper_v2_scan import CLASSES


def main():
    require_compute();d=folder('selection');inputs={};rows=[];examples=[]
    ref=cached(inputs,('TYPE','ls_id','pop','heldout','physical','region'))
    theory=read(OUT/'native',('template.npz','south','north','north_support','phase'),inputs)
    curves=prepare_templates(theory);error=read(OUT/'parent',('error_curve.npz','g_centers','sigma_g'),inputs)
    excluded=read(OUT/'parent',('exclusions.npz','TYPE','ls_id'),inputs)['ls_id']
    models={code:joblib.load(verified(ROOT/f'data/v3_10/selection/{side}/model.joblib',
        json.loads((ROOT/f'data/v3_10/selection/{side}/complete.json').read_text())['outputs']['model.joblib']))
        for code,side in enumerate(('south','north'))}
    for model in models.values():model['model'].n_jobs=1
    threshold=(2.5/np.log(10)/.1/10**((22.5-20)/2.5))**2
    for path in cache_folders():
        t=read(path,('photometry.npz','TYPE','ls_id','m','observed','dep','native','features'),inputs)
        ii=index_join(t['ls_id'],ref['ls_id']);use=(ii>=0)&~np.isin(t['ls_id'],excluded)
        t={k:v[use] for k,v in t.items()};ii=ii[use]
        if not len(ii):continue
        cmd=accepted(t['m'],t['observed'][:,0],t['native'],MU,curves,error).any(axis=1)
        score=np.full(len(ii),np.nan)
        for code in (0,1):
            j=np.flatnonzero(t['native']==code)
            if len(j):
                model=models[code];score[j]=apply_temperature(model['model'].predict_proba(t['features'][j,:3 if code else 16]),model['temperature'])[:,0]
        depth=(t['dep']>=threshold).all(axis=1)
        gr=t['m'][:,0]-t['m'][:,1]
        for v,limit in enumerate((20.,23.)):
            parent=depth&(t['observed']<limit).all(axis=1)
            for label,history in (('all_references',np.ones(len(ii),bool)),('heldout_references',ref['heldout'][ii])):
                for pop in np.unique(ref['pop'][ii]):
                    for low,high in ((-.2,.3),(.3,.5),(.5,.8),(.8,1.2)):
                        select=parent&history&(ref['pop'][ii]==pop)&(gr>=low)&(gr<high)
                        for code in (0,1):
                            use=select&(t['native']==code)
                            rows.append(dict(native=code,version=('bright','deep')[v],history=label,population=CLASSES[pop],
                               gr_lo=low,gr_hi=high,parent=int(use.sum()),cmd_union=int((use&cmd).sum()),
                               cmd_score=int((use&cmd&(score>=.8)).sum())))
    # Sum across disjoint source shards without confusing totals with rates.
    grouped={}
    for r in rows:
        keys=tuple(r[k] for k in ('native','version','history','population','gr_lo','gr_hi'))
        q=grouped.setdefault(keys,{k:v for k,v in r.items() if k not in ('parent','cmd_union','cmd_score')})
        for k in ('parent','cmd_union','cmd_score'):q[k]=q.get(k,0)+r[k]
    for q in grouped.values():q['score_retention_in_CMD']=q['cmd_score']/q['cmd_union'] if q['cmd_union'] else None
    csv_rows(d/'chemical_phase_score_retention.csv',list(grouped.values()))
    complete(d,inputs,scope='actual reference-labelled subset of the same strict high-latitude low-EBV atlas parent; no population-contamination inference',
        labels='spectroscopic FeH/gravity population classes retained from the prior reference census, never assigned from photometric distance',
        selection='same observed grz limits, source depth20 and fixed hard-CMD union; expanded native classifier score>=.8 is an extra selection')


if __name__=='__main__':main()
