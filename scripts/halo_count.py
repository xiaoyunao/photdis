#!/usr/bin/env python3
"""Count hard CMD selections at every modulus in real Nside512 pixels."""
import argparse
import healpy as hp
import joblib
from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature
from halo_common import *
from halo_filtering import accepted, prepare_templates, channel_weights


def run(number):
    inputs={};d=folder('counts')/f'{number:02}';d.mkdir(exist_ok=True)
    if (d/'complete.json').exists():
        r=json.loads((d/'complete.json').read_text())
        assert r['config']==CONFIG_HASH and r['code']['halo_count.py']==sha(Path(__file__))
        for name,h in r['outputs'].items():verified(d/name,h)
        return
    cache=list(cache_folders())[number]
    t=read(cache,('photometry.npz','TYPE','ls_id','RA','DEC','native','m','err','observed','dep','features','pdf','mass'),inputs)
    ex=read(OUT/'parent',('exclusions.npz','TYPE','ls_id'),inputs)
    keep=~np.isin(t['ls_id'],ex['ls_id'])
    t={k:v[keep] for k,v in t.items()}
    assert (t['TYPE']=='PSF').all()
    theory=read(OUT/'native',('template.npz','south','north','north_support','phase'),inputs)
    curves=prepare_templates(theory)
    error_curve=read(OUT/'parent',('error_curve.npz','g_centers','sigma_g'),inputs)
    classifiers={}
    for code,side in enumerate(('south','north')):
        home=ROOT/f'data/v3_10/selection/{side}'
        r=json.loads((home/'complete.json').read_text())
        classifiers[code]=joblib.load(verified(home/'model.joblib',r['outputs']['model.joblib']))
        classifiers[code]['model'].n_jobs=1
        inputs[side+'_classifier']=sha(home/'complete.json')
    npix=hp.nside2npix(C['nside'])
    counts=np.zeros((2,3,len(MU),npix),'i4')
    probability=np.zeros((2,7,npix))
    channel_w2=np.zeros((2,5,npix))
    weights=channel_weights(MU,C['distance_bins_kpc'])
    stats=[]
    depth_threshold=(2.5/np.log(10)/.1/10**((22.5-20)/2.5))**2
    for start in range(0,len(t['ls_id']),50000):
        a={k:v[start:start+50000] for k,v in t.items()}
        pix=hp.ang2pix(C['nside'],a['RA'],a['DEC'],lonlat=True,nest=True)
        match=accepted(a['m'],a['observed'][:,0],a['native'],MU,curves,error_curve)
        union=match.any(axis=1)
        score=np.full(len(pix),np.nan)
        for code in (0,1):
            j=np.flatnonzero(union&(a['native']==code))
            if len(j):
                model=classifiers[code]
                x=a['features'][j,:3 if code else 16]
                score[j]=apply_temperature(model['model'].predict_proba(x),model['temperature'])[:,0]
        eligible=(a['dep']>=depth_threshold).all(axis=1)
        weighted=match@weights.T
        for v,limit in enumerate((20.,23.)):
            selected=eligible&(a['observed']<limit).all(axis=1)
            common=selected&union&a['pdf']
            assert np.isfinite(a['mass'][common]).all()
            for gate,mask in enumerate((selected,selected&(score>=C['score_cut']),selected&a['pdf'])):
                for node in range(len(MU)):
                    use=mask&match[:,node]
                    np.add.at(counts[v,gate,node],pix[use],1)
            for b in range(7):
                np.add.at(probability[v,b],pix[common],a['mass'][common,b])
            for b in range(5):
                np.add.at(channel_w2[v,b],pix[selected],weighted[selected,b]**2)
            stats.append(dict(version=('bright','deep')[v],parent=len(pix),depth_eligible=int(eligible.sum()),
                selected_parent=int(selected.sum()),cmd_union=int((selected&union).sum()),
                cmd_pdf_common=int(common.sum()),cmd_score_selected=int((selected&(score>=C['score_cut'])).sum()),
                qrf_mass_total=a['mass'][common].sum(),node_count=int((match&selected[:,None]).sum())))
    for v in (0,1):
        expected=sum(r['node_count'] for r in stats if r['version']==('bright','deep')[v])
        assert counts[v,0].sum()==expected
        n=sum(r['cmd_pdf_common'] for r in stats if r['version']==('bright','deep')[v])
        np.testing.assert_allclose(probability[v].sum(),n,atol=2e-5,rtol=0)
    assert (counts[:,1:]<=counts[:,:1]).all()
    atomic_npz(d/'cube.npz',counts=counts,probability=probability,channel_sumw2=channel_w2)
    csv_rows(d/'census.csv',stats)
    complete(d,inputs,cache=str(cache.relative_to(OUT)),TYPE='PSF',
        counts_axis='bright/deep, all/score>=0.8/PDF-common, 61 mu nodes, NESTED celestial pixels',
        probability_axis='same CMD-union source parent with PDF, seven exclusive true PDF integrals including below/above plotted range',
        node_sumw2='equals counts for binary CMD acceptance; broad-channel sumw2 accumulated per source with overlapping nodes retained')


if __name__=='__main__':
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('number',type=int)
    run(parser.parse_args().number)
