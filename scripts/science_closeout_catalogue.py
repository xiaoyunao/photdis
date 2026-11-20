#!/usr/bin/env python3
"""Build science-domain marginals, paired reference migration and GD-1 evidence."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests'
OUT = MANIFEST / 'science_closeout_20261120'
REGIONS = ('south_v35', 'south_faint', 'south_low', 'north')
DISTANCE = np.array([0.,2.,5.,10.,np.inf])
SHELLS = ('0_2','2_5','5_10','10_plus')


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def core_marginals(destination):
    """Read archived selected rows only (no sweep or full PDF replay)."""
    count, bindings = Counter(), {}
    archive = ROOT/'data/final_science/scan'
    for marker in sorted(archive.glob('[0-9][0-9].json')):
        record = json.loads(marker.read_text())
        npz = marker.with_suffix('.npz')
        if npz.name not in record['outputs']:
            continue
        assert digest(npz) == record['outputs'][npz.name]
        bindings[str(npz.relative_to(ROOT))] = record['outputs'][npz.name]
        with np.load(npz) as p:
            assert (p['TYPE'] == 'PSF').all()
            use = (p['science']>0)&(p['science']<=3)
            r_bin = np.searchsorted([16,18,20,21,22,23],p['r'][use],side='right')
            cells = np.column_stack((p['region'][use],p['science'][use],r_bin))
            keys,n = np.unique(cells,axis=0,return_counts=True)
            count.update({(REGIONS[i],s,r):int(x) for (i,s,r),x in zip(keys,n)})
    assert sum(v for (_,s,_),v in count.items() if s in (1,2)) == 2281023
    assert sum(v for (_,s,_),v in count.items() if s==3) == 106196
    destination.mkdir(parents=True,exist_ok=True)
    write(destination/'core_counts.csv',[dict(region=k[0],science=k[1],r_bin=k[2],n=n)
                                         for k,n in sorted(count.items())])
    (destination/'core_inputs.json').write_text(json.dumps(bindings,indent=2)+'\n')


def domains(inputs):
    raw = read(inputs/'final_science/evidence/catalog_census.csv')
    depths = {(x['side'],x['latitude']):x for side in ('north','south')
              for x in read(inputs/f'final_science/area/{side}/area_totals.csv')}
    core = read(OUT/'core_counts.csv')
    config = json.loads((ROOT/'configs/final_science.json').read_text())
    rows = []
    features = dict(r_bin='magnitude',color_bin='color',lat_bin='abs_latitude',ebv_bin='ebv',quality=None)
    for region in REGIONS:
        sample = [x for x in raw if x['region'] == region]
        system = 'BASS/MzLS' if region=='north' else 'DECam'
        area = depths['north' if region=='north' else 'south','low' if region=='south_low' else 'high']
        selected = [x for x in core if x['region']==region]
        for key,dim in (('all',None),*features.items()):
            keys = [None] if key=='all' else sorted({int(x[key]) for x in sample})
            for code in keys:
                subset = sample if code is None else [x for x in sample if int(x[key])==code]
                n = sum(int(x['n']) for x in subset)
                pdf = sum(int(x['n']) for x in subset if x['state']=='3')
                strict = sum(int(x['n']) for x in subset if x['state']=='3' and x['quality']=='7')
                x = selected if key=='all' else [q for q in selected if int(q['r_bin'])==code]
                core_n = sum(int(q['n']) for q in x if q['science'] in ('1','2'))
                far_n = sum(int(q['n']) for q in x if q['science']=='3')
                edges = config[f'{dim}_edges'] if dim else ()
                lo,hi = edges[code:code+2] if dim and code < len(edges)-1 else ('','')
                area_label = 'shared south high-latitude footprint' if region in ('south_v35','south_faint') else 'native footprint'
                rows.append(dict(region=region,native=system,dimension=dim or key,bin=code,lo=lo,hi=hi,
                    owned_psf=n,qrf_pdf=pdf,strict_grz_pdf=strict,
                    core_work=core_n if key in ('all','r_bin') else '',
                    far_work=far_n if key in ('all','r_bin') else '',
                    unmasked_footprint_deg2=area['unmasked'],footprint_definition=area_label,
                    **{k+'_area_deg2': v for k,v in area.items() if k.startswith('grz_')}))
    tot = [x for x in rows if x['dimension']=='all']
    assert [sum(x[k] for x in tot) for k in ('owned_psf','qrf_pdf','core_work','far_work')] == [
        72289386,43347612,2281023,106196]
    write(OUT/'production_marginals.csv',rows)

    refs = read(inputs/'final_science/evidence/heldout_distance.csv')
    # Keys retain native system, tier, reference and actual denominator.
    for row in refs:
        row['interpretation'] = ('reference-conditional support' if int(row['n']) >= 30
                                  else 'limited reference support')
        row['role'] = 'archive held-out reference census; source-dependent scales'
    write(OUT/'reference_marginals.csv',refs)


def migrations(inputs):
    all_rows = read(inputs/'release_readiness/population/retention_migration.csv')
    rows, scen = [], []
    for row in all_rows:
        n = int(row['n'])
        hard = [int(row[f'median_{x}']) for x in SHELLS]
        middle = [float(row[f'pdf_{x}']) for x in SHELLS[1:3]]
        assert sum(hard)==n and sum(middle)<=n+1e-5
        rows.append(dict(row, qrf_selection_fraction=n/int(row['qualified']) if int(row['qualified']) else '',
            remaining_pdf_outside_2_10=np.clip(n-sum(middle),0,n),
            hard_2_5_fraction=hard[1]/n if n else '', pdf_2_5_fraction=middle[0]/n if n else '',
            hard_5_10_fraction=hard[2]/n if n else '', pdf_5_10_fraction=middle[1]/n if n else ''))
    write(OUT/'reference_migration.csv',rows)
    radii={'fixed_0_2':(0,2),'fixed_2_5':(2,5),'fixed_5_10':(5,10)}
    classes = ('dwarf_feh_ge_minus1p5','dwarf_feh_lt_minus1p5','intermediate_gravity','giant')
    for system in ('north','south_low'):
        for ref in ('SpecDis','SPDist'):
            for name,(lo,hi) in radii.items():
                matched = [x for x in rows if (x['region'],x['source'],x['observable'])==(system,ref,'all')
                           and x['population'] in classes and float(x['ref_lo'])==lo and float(x['ref_hi'])==hi]
                for scenario, multiplier in [('fewer_metalpoor',.5),('observed_reference',1.),('more_metalpoor',2.)]:
                    for stage in ('supported','selected'):
                        q = [x for x in matched if x['stage']==stage]
                        w = lambda x: multiplier if x['population']==classes[1] else 1.
                        denominator = sum(w(x)*int(x['n']) for x in q)
                        for shell in ('2_5','5_10'):
                            count = [sum(w(x)*float(x[key+'_'+shell]) for x in q) for key in ('median','pdf')]
                            metalpoor = [sum(w(x)*float(x[key+'_'+shell]) for x in q if x['population']==classes[1])
                                         for key in ('median','pdf')]
                            scen.append(dict(region=system,source=ref,truth_shell=name,scenario=scenario,
                                stage=stage,prediction_shell=shell,weighted_reference_count=denominator,
                                hard_count=count[0],pdf_count=count[1],hard_metalpoor=metalpoor[0],
                                pdf_metalpoor=metalpoor[1],hard_metalpoor_fraction=metalpoor[0]/count[0] if count[0] else '',
                                pdf_metalpoor_fraction=metalpoor[1]/count[1] if count[1] else ''))
    write(OUT/'population_scenarios.csv',scen)


def gd1(inputs):
    root = inputs/'v3_18/members'
    cohort = {int(r['ls_id']):r for r in read(root/'roster.csv')}
    with np.load(root/'posteriors.npz') as p:
        j = np.flatnonzero(p['science'])
        assert (p['TYPE'][j]=='PSF').all()
        keys = p['ls_id'][j]
        t = [cohort[k] for k in keys]
        assert all(x['TYPE']=='PSF' and x['science']=='True' and x['train_cal']=='False' for x in t)
        edges, m = p['mu_edges'][j],p['mass'][j]
        median = p['quantiles_kpc'][j,2]
        def cdf(distance):
            with np.errstate(divide='ignore'):
                cut = np.atleast_1d(10+5*np.log10(distance))[:,None]
            return np.sum(m*np.clip((cut-edges[:,:-1])/np.diff(edges,axis=1),0,1),axis=1)
        pdf=np.diff([cdf(k) for k in DISTANCE],axis=0).T
        np.testing.assert_allclose(pdf.sum(axis=1),1,atol=3e-6)
        ref = np.array([float(x['desi_kpc']) for x in t])
        phi = np.array([float(x['phi1']) for x in t])
        tight = cdf(1.1*ref)-cdf(.9*ref)
        per_member = [dict(ls_id=k,group=x['group'],segment=x['segment'],phase=x['phase'],phi1=angle,
                      d_ref=d,d_qrf=med,in_10percent=float(.9*d<=med<=1.1*d),pdf_10percent=w,
                      **{f'pdf_{s}':p0 for s,p0 in zip(SHELLS,mass)})
                      for k,x,angle,d,med,w,mass in zip(keys,t,phi,ref,median,tight,pdf)]
        write(OUT/'gd1_members.csv',per_member)
        rows=[]
        for label, group in [('all',per_member), *[(g,[x for x in per_member if x['group']==g]) for g in ('thin','cocoon')]]:
            for a,b in ((-180,180),(-90,-50),(-50,-30),(-30,0)):
                ix = [x for x in group if a<=x['phi1']<b]
                if not ix:
                    continue
                n=len(ix)
                row = dict(group=label,phi1_lo=a,phi1_hi=b,n=n,ratio_median=np.median([x['d_qrf']/x['d_ref'] for x in ix]),
                           hard_10pct=sum(x['in_10percent'] for x in ix),pdf_10pct=sum(x['pdf_10percent'] for x in ix))
                for s,lo,hi in zip(SHELLS,DISTANCE[:-1],DISTANCE[1:]):
                    row['hard_'+s] = sum(lo<=x['d_qrf']<hi for x in ix)
                    row['pdf_'+s] = sum(x['pdf_'+s] for x in ix)
                    row['ref_'+s] = sum(lo<=x['d_ref']<hi for x in ix)
                rows.append(row)
        write(OUT/'gd1_counts.csv',rows)


if __name__ == '__main__':
    import sys
    if '--core' in sys.argv:
        from photdis.stage1 import require_compute
        require_compute()
        core_marginals(ROOT/'data/science_closeout_20261120/catalogue')
    else:
        domains(MANIFEST)
        migrations(MANIFEST)
        gd1(MANIFEST)
