#!/usr/bin/env python3
"""Reuse pre-existing case identities; show full gray-remapped distributions."""
from nat8_common import *
from nat8_statsmath import hist_curves,sparse_CDF_components
from seq7_math import log_density_grid
from photdis.stage1 import require_compute


def main():
    require_compute();d=folder('examples');t=load(OUT/'prepare/roster.npz');params=json.loads((OUT/'fit/parameters.json').read_text());draws=load(OUT/'fit/parameter_draws.npz');oldcases=json.loads((OLD/'examples/cases.json').read_text());oldcurves=load(OLD/'examples/example_curves.npz');x=oldcurves['mu'];arrays={'mu':x};cases=[];atoms={}
    chosen=[r for r in oldcases if r['kind']=='QRF_decline' or r['kind']=='sequence_turnoff' or (r['system']=='M5' and r['kind']=='sequence_lower_MS')]
    assert len(chosen)==7
    for c in chosen:
        i=c['index'];s=c['system'];k=len(cases);z=load(OUT/'prepare'/f'{s}_E.npz');at=int(np.flatnonzero(z['ii']==i)[0]);label=str(t['fold'][i]) if s=='M5' else 'full';a=np.array(params[label]['point']['a']);p=load(OUT/'predict'/f'{s}_calibrated.npz');mm=z['means'][at];lw=z['logw'][at];sg=z['sigma'][at];xx=z['x'];mapped=mm-a[0]-a[1]*xx
        item={key:value for key,value in c.items() if key!='methods'};item.update(case=k,previous_case=c['case'],fit=label,rule='same sealed v7 identity, no ranking by new results',methods={})
        for m,means in [('E',mm),('CalE',mapped)]:
            density=np.exp(log_density_grid(x,means,lw,sg));active,w=sparse_CDF_components(means,lw);gx,cf,_,_=hist_curves(active,w,sg,.00125);cdf=np.interp(x,gx,cf[0],left=0,right=1);arrays[f'{k}_{m}_density']=density;arrays[f'{k}_{m}_CDF']=cdf;item['methods'][m]=dict(q=(z['q'][at] if m=='E' else p['q'][at]).tolist(),supported=bool(z['support'][at]),post=float(z['post'][at]));atoms[f'{k}_{m}_means']=means
        atoms[f'{k}_logweights']=lw;atoms[f'{k}_sigma']=np.array(sg);atoms[f'{k}_node_gr']=xx+C['pivot'];atoms[f'{k}_post_nodes']=z['node_post']
        arrays[f'{k}_QRF_density']=oldcurves[f'{c["case"]}_QRF_density'];arrays[f'{k}_QRF_CDF']=oldcurves[f'{c["case"]}_QRF_CDF'];item['methods']['QRF']=c['methods']['QRF']
        if c['kind']=='QRF_decline':
            for field in ('QRF_centered_density','QRF_centered_CDF','QRF_center'):arrays[f'{k}_{field}']=oldcurves[f'{c["case"]}_{field}']
        elif s!='M5':
            raw=load(OUT/'marginal_raw'/f'{i:05d}.npz');arrays[f'{k}_CalE_marg_density']=np.interp(x,raw['mu'],raw['density'],left=0,right=0);arrays[f'{k}_CalE_marg_CDF']=np.interp(x,raw['mu'],raw['cdf'],left=0,right=1);item['methods']['CalE_marg']=dict(q=p['marginal_q'][at].tolist(),supported=bool(z['support'][at]));atoms[f'{k}_full_parameter_draws']=draws['full_a']
        cases.append(item);print('Example',s,c['kind'],c['ls_id'],flush=True)
    save_json(d/'cases.json',cases);npz(d/'curves.npz',**arrays);npz(d/'component_parameters.npz',**atoms);csv_rows(d/'selected.csv',[{key:value for key,value in c.items() if key!='methods'} for c in cases]);complete(d,{'fit':sha(OUT/'fit/complete.json'),'predict':sha(OUT/'predict/complete.json'),'original_examples':sha(OLD/'examples/complete.json')},case_selection='seven existing v7 identities, including unchanged fixed three QRF tails',conditional_PDFs=True,full_factorized_parameters=True)


if __name__=='__main__':main()
