#!/usr/bin/env python3
"""One actual-source near/far replay and original v3.8 pixel regression."""
from photdis.stage1 import require_compute
from sky6_common import *
from sky6_channels import model,MU,NEAR,W
from stream_local import accept
from v3_10_figures import regression
from photdis_figure_style import STYLE


def main():
    require_compute();d=folder('validation');checks=[]
    for marker in sorted((OUT/'cache').glob('*/complete.json')):
        r=json.loads(marker.read_text())
        if 1000<=r['counts']['valid_gr']<=10000:break
    else:raise ValueError('No bounded actual source part for smoke replay')
    t=read_columns(verified(marker.parent/'photometry.npz',r['outputs']['photometry.npz']),('TYPE','ls_id','m','old_domain'))
    with np.load(OUT/'identities/alias_exclusions.npz') as z:keep=~np.isin(t['ls_id'],z['ls_id'])
    t={k:v[keep] for k,v in t.items()};assert (t['TYPE']=='PSF').all();cc,ec=model(int(marker.parent.name.startswith('north_')));a=accept(t['m'],MU,cc,ec,'MS_TO')
    direct=a@W.T;split=a[:,:len(NEAR)]@W[:,:len(NEAR)].T+a[:,len(NEAR):]@W[:,len(NEAR):].T;np.testing.assert_allclose(direct,split,atol=1e-12)
    with np.load(OUT/'nodes'/marker.parent.name/'counts.npz') as z:
        for view,flag in enumerate((np.ones(len(a),bool),t['old_domain'])):np.testing.assert_array_equal(a[flag,len(NEAR):].sum(axis=0),z['totals'][view])
    checks.append(dict(check='Actual native PSF201-node replay equals split near+frozen61far response, and exact far counts reproduce original part in both domains',part=marker.parent.name,N=len(a),passed=True))
    for stage in ('atlas','band','rgb','old_domain','no_extinction','prior_display','brightness','ridge','width','channel_trials'):
        root=OUT/stage;r=json.loads((root/'complete.json').read_text())
        for name,h in r['outputs'].items():verified(root/name,h)
        checks.append(dict(check='source-bound stage output hashes',stage=stage,N=len(r['outputs']),passed=True))
    target=d/'v38_pixels';target.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):pixels=regression(target)
    save_json(d/'validation.json',dict(checks=checks,pixel_regression=pixels,style=check_templates(),scope='bounded actual source replay; exact baseline pixels; completed scientific stage hashes. All raw-source identity/count/geometry conservation already checked during actual extraction.'))
    complete(d,{},actual_source_smoke=True,original_style_pixels=True)


if __name__=='__main__':main()
