#!/usr/bin/env python3
"""Validate source-bound scientific outputs independently of acceptance outcomes."""
from PIL import Image
import time
from photdis.stage1 import require_compute
from sv18_common import *


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    require_compute()
    d=out('delivery');marks={};plates=[]
    stages=('definitions','observable','filter','provenance','cohorts','benchmark','numerics','populations','populations_clean',
            'systems','noise','predictive','oracle','figures','style_regression')
    deadline=time.monotonic()+600
    while not (DEST/'numerics/complete.json').exists():
        if time.monotonic()>deadline:
            raise RuntimeError('The concurrent numerical-integration check has not completed')
        time.sleep(5)
    reports={}
    for stage in stages:
        folder=DEST/stage
        marker=folder/'complete.json'
        r=json.loads(marker.read_text())
        reports[stage]=r
        for name,digest in r['outputs'].items():
            verified(folder/name,digest)
        assert r['style']==check_templates()
        marks[stage]=sha(marker)
        for path in sorted(folder.glob('*.png')):
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                assert image.width>=2000 and abs(image.info['dpi'][0]-300)<.02
            assert path.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
            assert path.with_suffix('.csv').is_file()
            if stage!='style_regression':
                plates.append(str(path.relative_to(DEST)))
    u=table(DEST/'definitions/gaia_reference_uncertainty.csv')
    assert len(u)==404 and all(r['centers_unchanged']=='True' for r in u)
    ratios=np.array([float(r['median_halfwidth95_to_linear_sigma']) for r in u])
    assert (ratios>=1.95).all() and (ratios<2.1).all()
    selection=table(DEST/'definitions/working_selection.csv')
    anchors=[r for r in selection if float(r['ref_lo'])==2 and r['population'] in
             ('ordinary_dwarf','metal_poor_dwarf','intermediate_gravity')]
    assert [int(r['selected_2_5']) for r in anchors]==[15827,219,61,14584,177,53]
    for r in selection:
        n=int(r['passed_probability'])
        assert n>=int(r['selected_2_5'])+int(r['selected_5_10'])
        assert int(r['applicable'])>=int(r['passed_score'])>=n
    checks=table(DEST/'filter/template_self_recovery.csv')
    assert len(checks)>0 and all(r['passed']=='True' for r in checks if r['method']=='selected')
    for r in table(DEST/'filter/empirical_injections.csv'):
        assert np.isfinite(float(r['profile_lo'])) and np.isfinite(float(r['profile_hi']))
    with np.load(DEST/'observable/selection.npz') as a:
        assert np.isfinite(a['response']).all()
        assert (a['response']>=-1e-10).all() and (a['response']<=1+1e-10).all()
    p=table(DEST/'populations/matched_metallicity.csv')
    for ref,weights,cells in (('specdis',3474,50),('spdist',3223,49)):
        row=next(r for r in p if r['reference']==ref and r['variant']=='original' and int(r['lo'])==2)
        assert int(float(row['sum_weights']))==weights and int(row['common_cells'])==cells
        assert abs(float(row['shift_mu'])-(.377044 if ref=='specdis' else .382353))<2e-5
    for ref,n in (('specdis',9355),('spdist',8927)):
        rows=table(DEST/'benchmark'/f'{ref}_metrics.csv')
        group=[r for r in rows if r['group']=='headline']
        assert len(group)==3 and all(int(r['n'])==n for r in group)
    q=table(DEST/'systems/track_version_audit.csv')
    assert all(r['byte_identical']=='True' for r in q if r['v3_sha256'] and r['v4_sha256'])
    save_json(d/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),stages=marks,plates=plates,
       filter_gate=reports['filter']['all_sky_search_authorized_by_evidence'],
       original_models_and_release='87069408',baseline_evidence='f4dc28f5',
       scientific_scope='acceptance flags record experimental limits; numerical checks are independent'))


if __name__=='__main__':
    main()
