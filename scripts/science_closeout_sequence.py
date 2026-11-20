#!/usr/bin/env python3
"""Compare known stellar sequences with the labels routed through frozen QRF leaves."""
import json

import numpy as np
from scipy.spatial import cKDTree

import sv18_benchmark as empirical
from environment_dust import modulus_quantiles
from environment_results import histogram_scores
from photdis.stage1 import require_compute
from photdis.v3_3 import frozen_qrf
from v3_13_clusters import SYSTEMS
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_9_catalog import OUT as NATIVE
from v3_13_clusters import model_and_history

CONFIG = ROOT / 'configs/science_closeout_20261120.json'
D = ROOT / 'data/science_closeout_20261120/sequence'
SETTINGS = json.loads(CONFIG.read_text())['sequence']


def training(side, inputs):
    table, offset = empirical.training(side, inputs)
    if side == 'south':
        root = ROOT / 'data/v3_0'
        ledger = load(root / 'controls/training.npz')
        keys = ('identity', 'primary_channel', 'rvs_feh', 'rvs_logg', 'rvs_parameter_quality')
        ref = empirical.read_columns(root / 'references.npz', keys)
        rows = ledger['rows']
        np.testing.assert_array_equal(ref['identity'][rows], ledger['identity'])
        table.update(identity=ledger['identity'], mu=ledger['primary_mu'],
                     source=np.array(['missing', 'specdis', 'spdist', 'gaia'])[ref['primary_channel'][rows]],
                     feh=np.where(ref['rvs_parameter_quality'][rows], ref['rvs_feh'][rows], np.nan),
                     logg=np.where(ref['rvs_parameter_quality'][rows], ref['rvs_logg'][rows], np.nan))
    else:
        role = load(NATIVE / 'north/native_model/roles.npz')
        take = np.isin(role['role'], ['gaia_fit', 'specdis_fit'])
        match = empirical.index_join(table['ls_id'], role['ls_id'][take])
        assert (match >= 0).all()
        table.update(identity=role['source_id'][take][match],
                     source=np.where(role['role'][take][match] == 'gaia_fit', 'gaia', 'specdis'),
                     mu=table['m'][:, 0]-table['y'],
                     feh=np.full(len(match), np.nan), logg=np.full(len(match), np.nan))
        historical = empirical.read_columns(ROOT/'data/v3_0/references.npz',
            ('source_id','class_label','rvs_parameter_quality','rvs_feh','rvs_logg'))
        good = historical['rvs_parameter_quality'] & (historical['source_id']>0)
        uid, counts = np.unique(historical['source_id'][good],return_counts=True)
        good &= np.isin(historical['source_id'],uid[counts==1])
        j = empirical.index_join(table['source_id'],historical['source_id'][good])
        compatible = (j>=0) & (table['class_label']>=0)
        compatible[compatible] &= (table['class_label'][compatible]
                                    == historical['class_label'][good][j[compatible]])
        table['feh'][compatible] = historical['rvs_feh'][good][j[compatible]]
        table['logg'][compatible] = historical['rvs_logg'][good][j[compatible]]
    assert (table['TYPE'] == 'PSF').all()
    return table, offset


def label_diagnostics(model, table, photo, ref, offset):
    # Normalize within each tree leaf, then average trees exactly as QRF does.
    x, z = [model.feature_matrix(np.pad(m, ((0, 0), (0, 2)))) for m in (table['m'], photo)]
    feh, logg = table['feh'], table['logg']
    columns = {
        'Mg': table['y'], 'mu_label': table['mu'],
        'feh_measured': np.isfinite(feh), 'feh_sum': np.nan_to_num(feh),
        'metalpoor': np.isfinite(feh) & (feh < -1.5),
        'logg_measured': np.isfinite(logg), 'logg_sum': np.nan_to_num(logg),
    }
    columns.update({name: table['source'] == name for name in ('gaia', 'specdis', 'spdist')})
    leaf = {name: np.zeros(len(photo)) for name in columns}
    ybin = np.searchsorted(model.edges, table['y'], side='right') - 1
    classes = np.where(table['class_label'] < 0, 3, table['class_label'])
    for tree, mapping, hist, types in zip(model.forest.estimators_, model.maps, model.tables, model.types):
        label = mapping[tree.apply(x)]
        query = mapping[tree.apply(z)]
        n = np.bincount(label, minlength=len(hist))
        assert np.all(n > 0)
        rebuilt = np.bincount(label * len(model.centers) + ybin,
                              minlength=hist.size).reshape(hist.shape)
        np.testing.assert_allclose(rebuilt / n[:, None], hist, atol=6e-8, rtol=0)
        c = np.bincount(label * 4 + classes, minlength=types.size).reshape(types.shape)
        np.testing.assert_allclose(c / n[:, None], types, atol=6e-8, rtol=0)
        for name, value in columns.items():
            leaf[name] += (np.bincount(label, weights=value, minlength=len(hist)) / n)[query]
    leaf = {'leaf_' + k: v / len(model.tables) for k, v in leaf.items()}
    mass, type_prob = model.distribution(z)
    for i, key in enumerate(('dwarf', 'giant', 'other', 'unknown')):
        leaf['leaf_' + key] = type_prob[:, i]
    x = table['m'][:, :2] - table['m'][:, 1:3]
    z = photo[:, :2] - photo[:, 1:3]
    box = np.array(SETTINGS['color_box_mag'])
    nearby = cKDTree(x / box).query_ball_point(z / box, 1, p=np.inf)
    y = table['y']
    leaf['local_n'] = np.array([len(i) for i in nearby])
    leaf['local_Mg'] = np.array([np.median(y[i]) if i else np.nan for i in nearby])
    leaf['leaf_delta_mu'] = photo[:, 0] - ref - leaf['leaf_Mg'] + offset
    leaf['local_delta_mu'] = photo[:, 0] - ref - leaf['local_Mg'] + offset
    return leaf, mass


def cdf(edges, mass, boundary):
    fraction = (boundary[:, None] - edges[:, :-1]) / np.diff(edges, axis=1)
    return np.sum(mass * np.clip(fraction, 0, 1), axis=1)


def metrics(method, edges, mass, reference, sigma):
    q = modulus_quantiles(mass, edges)
    delta = q[:, 2] - reference
    log_score, crps = histogram_scores(edges, mass, reference, sigma)
    return {method+'_'+name: value for name, value in
            dict(bias=delta, crps=crps, log=log_score,
                 c68=(q[:, 1] <= reference) & (reference <= q[:, 3]),
                 c95=(q[:, 0] <= reference) & (reference <= q[:, 4]),
                 width68=q[:, 3]-q[:, 1], width95=q[:, 4]-q[:, 0],
                 severe=np.abs(delta)>1).items()}


def cohorts():
    cluster = ROOT / 'data/cluster_depth_20261120'
    faint = ROOT / 'data/faint_signed_20261120'
    parameters = json.loads((ROOT/'configs/faint_signed_20261120.json').read_text())['cluster_candidates']
    for name, info in parameters.items():
        t = load(cluster / f'{name}_native.npz')
        p = load(faint / f'{name}_fixed_pdfs.npz')
        np.testing.assert_array_equal(t['ls_id'], p['ls_id'])
        assert (t['TYPE'] == 'PSF').all() and (p['TYPE'] == 'PSF').all()
        photo, error = t['mag_grz'], t['error_grz']
        mr = photo[:, 1]-info['rrl_mu']
        quality = load(cluster / f'{name}_diagnostics.npz')
        np.testing.assert_array_equal(t['ls_id'], quality['ls_id'])
        block = (t['ls_id'] >> 20) & ((1<<20)-1)
        yield dict(system=name, side='south', ls_id=t['ls_id'], identity=t['identity'],
                   photo=photo, error=error, r=t['r'], block=block,
                   phase=np.where((mr>=5) & (photo[:, 0]-photo[:, 1]>=.4), 'lower_MS', 'other'),
                   ref=np.full(len(mr), info['rrl_mu']),
                   sigma=np.full(len(mr), np.hypot(info['rrl_sigma_mag'], .1)),
                   edges=p['edges'], mass=p['mass'], ebv=t['EBV'], radius=t['radius'],
                   fracflux=quality['fracflux'], depth_margin=quality['depth_margin'],
                   strict=quality['clean'])
    seq = ROOT / 'data/sequence_mechanisms_20261119'
    t = load(seq/'prepare/roster.npz')
    for side in ('south','north'):
        p = load(seq/'predict'/f'QRF_{side}.npz')
        for name in np.unique(t['system'][p['ii']]):
            take = t['system'][p['ii']] == name
            j = p['ii'][take]
            assert (t['TYPE'][j] == 'PSF').all()
            ref = 10+5*np.log10(t['ref'][j])
            yield dict(system=name, side=side, ls_id=t['ls_id'][j], identity=t['qrf_identity'][j],
                       photo=t['qrf_mag'][j], error=t['qrf_err'][j], r=t['r'][j],
                       phase=np.where(t['v5'][j], t['phase'][j], 'outside_v5'),
                       ref=ref, sigma=np.full(len(j), np.std(5*np.log10(t['scale_'+name]), ddof=1)),
                       block=(t['ls_id'][j]>>20)&((1<<20)-1), edges=p['edges'][take], mass=p['mass'][take],
                       ebv=np.full(len(j),np.nan), radius=np.full(len(j),np.nan),
                       fracflux=np.full(len(j),np.nan), depth_margin=np.full(len(j),np.nan),
                       strict=np.zeros(len(j),bool))



    # M92 has prior sky exposure, but the native observed member identities
    # can be replayed independently of the fit/calibration IDs.
    name = 'M92'
    side = 'north'
    home = ROOT/'data/v3_14/audit'/name
    table = load(home/'native_join.npz')
    use = (table['member'] & table['target'] & ~table['training_or_calibration']
           & table['low_crowding'] & ~table['cluster_mask']
           & np.isfinite(table['qrf_kpc']).all(axis=1))
    j = np.flatnonzero(use)
    assert (table['TYPE'][j]=='PSF').all()
    model, offset, history = model_and_history(side,{})
    identity = table['source_id'][j]
    assert (identity>0).all() and not np.isin(identity,history).any()
    m = np.column_stack([table[k][j] for k in ('g0','r0','z0')])
    e = table['error_grz'][j]
    arms = [frozen_qrf(model,m[i],e[i],identity[i]) for i in np.array_split(np.arange(len(j)),8)]
    q = np.concatenate([p['distance'] for p in arms])*10**(offset/5)
    np.testing.assert_allclose(q,table['qrf_kpc'][j],atol=2e-5,rtol=3e-6)
    ref = np.full(len(j),10+5*np.log10(SYSTEMS[name][1]))
    yield dict(system=name,side=side,ls_id=table['ls_id'][j],identity=identity,
               photo=m,error=e,r=table['r'][j],ref=ref,sigma=np.full(len(j),.1),
               phase=table['phase_box'][j], block=(table['ls_id'][j]>>20)&((1<<20)-1),
               edges=np.concatenate([p['mu_edges'] for p in arms])+offset,
               mass=np.concatenate([p['mass'] for p in arms]),
               ebv=np.full(len(j),np.nan),radius=np.full(len(j),np.nan),
               fracflux=np.full(len(j),np.nan),depth_margin=np.full(len(j),np.nan),
               strict=np.zeros(len(j),bool))


def summary(a, use):
    ret = dict(n=int(use.sum()), blocks=len(np.unique(a['block'][use])))
    if not ret['n']:
        return ret
    for k in ('leaf_delta_mu', 'local_delta_mu', 'leaf_Mg','local_Mg','local_n','r','Mr','Mg','gr','rz',
              'err_max','ebv','radius','fracflux','depth_margin','p_near','p_near_dim','p_far_bright'):
        v = a[k][use]
        ret[k] = np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else None
    for k in ('gaia','specdis','spdist','dwarf','giant','other','unknown','feh_measured','logg_measured','metalpoor'):
        ret['leaf_'+k] = np.mean(a['leaf_'+k][use])
    for k in ('feh','logg'):
        known = a['leaf_'+k+'_measured'][use].sum()
        ret['leaf_'+k+'_mean_when_measured'] = a['leaf_'+k+'_sum'][use].sum()/known if known else None
    for method in ('qrf','simple'):
        error = a[method+'_bias'][use]
        center = np.median(error)
        for key, value in (('bias', center), ('nmad', 1.4826*np.median(np.abs(error-center)))):
            ret[method+'_'+key] = value
        for key in ('crps','log','c68','c95','severe'):
            ret[method+'_'+key] = np.mean(a[method+'_'+key][use])
        for key in ('width68','width95'):
            ret[method+'_'+key] = np.median(a[method+'_'+key][use])
    return ret


def main():
    require_compute()
    D.mkdir(parents=True, exist_ok=True)
    rows, train_rows, artifacts, inputs, models = [], [], {}, {}, {}
    for t in cohorts():
        side, system = t['side'], t['system']
        if side not in models:
            train, offset = training(side, inputs)
            model, replay_offset, _ = model_and_history(side, inputs)
            np.testing.assert_allclose(offset, replay_offset)
            simple = empirical.fit(train)
            models[side] = model, train, offset, simple
            for source in np.unique(train['source']):
                use = train['source']==source
                train_rows.append(dict(side=side,source=source,n=int(use.sum()),
                    feh_n=int(np.isfinite(train['feh'][use]).sum()), logg_n=int(np.isfinite(train['logg'][use]).sum()),
                    median_Mg=np.median(train['y'][use]), median_mu=np.median(train['mu'][use])))
            p = D/f'{side}_training.npz'
            atomic_npz(p, **train)
            artifacts[p.name] = sha(p)
        model, train, offset, simple = models[side]
        assert not np.isin(t['identity'][t['identity']>0],train['identity']).any()
        photo, ref, edges, mass = (t[k] for k in ('photo','ref','edges','mass'))
        a = {k:t[k] for k in ('ls_id','phase','r','block','ref','ebv','radius','fracflux','depth_margin','strict')}
        gr,rz=(photo[:,:2]-photo[:,1:]).T
        a.update(gr=gr,rz=rz,Mg=photo[:,0]-ref,Mr=photo[:,1]-ref,err_max=t['error'].max(1))
        leaf, _ = label_diagnostics(model, train, photo, ref, offset)
        a.update(leaf)
        w,b = SETTINGS['external_window_mag'], SETTINGS['branch_offset_mag']
        a.update(p_near=cdf(edges,mass,ref+w)-cdf(edges,mass,ref-w),
                 p_near_dim=cdf(edges,mass,ref-b),p_far_bright=1-cdf(edges,mass,ref+b))
        pdf, bins = empirical.predict_pdf(simple,photo,t['error'],t['identity'])
        simple_edges = photo[:,:1]-bins[None,::-1]+offset
        for name,e,m in (('qrf',edges,mass),('simple',simple_edges,pdf[:,::-1])):
            a.update(metrics(name,e,m,ref,t['sigma']))
        p = D/f'{system}_{side}.npz'
        atomic_npz(p, TYPE=np.full(len(ref),'PSF'), **a)
        artifacts[p.name] = sha(p)
        strata = [('all','all',np.ones(len(ref),bool)), ('quality','strict',a['strict'])]
        strata += [('phase', label, a['phase']==label) for label in np.unique(a['phase'])]
        strata += [('Mr',f'{lo:g}-{hi:g}',(a['Mr']>=lo)&(a['Mr']<hi))
                   for lo,hi in SETTINGS['phase_bands_mr']]
        strata += [('r',f'{lo:g}-{hi:g}',(a['r']>=lo)&(a['r']<hi)) for lo,hi in
                   [(0,18),(18,20),(20,21),(21,22),(22,23)]]
        for axis,label,use in strata:
            rows.append(dict(system=system,side=side,axis=axis,stratum=label,**summary(a,use)))
        print(system, side, len(ref), flush=True)
    csv_rows(D/'training_sources.csv', train_rows)
    csv_rows(D/'sequence.csv', rows)
    save_json(D/'complete.json', dict(config=sha(CONFIG), model_inputs=inputs, archives=artifacts,
              outputs={f:sha(D/f) for f in ('training_sources.csv','sequence.csv')}))


if __name__ == '__main__':
    main()
