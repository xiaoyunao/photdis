#!/usr/bin/env python3
"""Re-evaluate nine frozen dust apertures on a common reference and map support."""
import os

import numpy as np

import environment_results as env
from environment_dust import modulus_quantiles
from environment_transfer import OLD, OUT as TRANSFER
from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_6_vac_distances import read_columns

D = ROOT / 'data/science_closeout_20261120/environment'


def pdf_stats(e, w, mu, sig):
    q = modulus_quantiles(w, e)
    log, crps = env.histogram_scores(e, w, mu, sig)
    return dict(bias=q[:, 2]-mu, severe=abs(q[:, 2]-mu)>1, log=log, crps=crps,
                c68=(q[:, 1]<=mu)&(mu<=q[:, 3]), c95=(q[:, 0]<=mu)&(mu<=q[:, 4]),
                width68=q[:, 3]-q[:, 1], width95=q[:, 4]-q[:, 0])


def reduce(stats, i):
    center = np.median(stats['bias'][i])
    out = {k: float(np.mean(v[i])) for k,v in stats.items()}
    out.update(bias=center, nmad=1.4826*np.median(abs(stats['bias'][i]-center)),
               width68=np.median(stats['width68'][i]), width95=np.median(stats['width95'][i]))
    return out


def bootstrap(stats, idx, block, rng):
    groups = [idx[block[idx]==k] for k in np.unique(block[idx])]
    for pick in rng.integers(0,len(groups),size=(600,len(groups))):
        i = np.concatenate([groups[k] for k in pick])
        for k in (0,1,2):
            yield k, np.median(stats[k]['bias'][i]), np.mean(stats[k]['crps'][i]-stats[1]['crps'][i])


def main():
    require_compute()
    D.mkdir(parents=True, exist_ok=True)
    rows, flow, oracle, inputs = [], [], [], {}
    rng = np.random.default_rng(20261120)
    for root in (OLD, TRANSFER):
        names = ('dust/forecasts.npz', 'forecast/roster.npz', 'forecast/pdf.npz')
        tcols = ('TYPE','ls_id','field','longitude','latitude','gaia_sigma_mu','gaia_proxy',
                 'mag_grz','r','EBV','qrf_fit_overlap','historical_overlap','gaia_evaluation')
        d,t,p = (load(root/names[0]), read_columns(root/names[1],tcols),
                 read_columns(root/names[2],('rows','TYPE','ls_id','mu_edges','mass')))
        inputs.update({str((root/f).relative_to(ROOT)):sha(root/f) for f in names})
        j = d['rows']
        at = np.searchsorted(p['rows'],j)
        np.testing.assert_array_equal(t['ls_id'][j], d['ls_id'])
        np.testing.assert_array_equal(p['ls_id'][at], d['ls_id'])
        assert (t['TYPE'][j]=='PSF').all() and (d['TYPE']=='PSF').all()
        mu,sig = d['reference_mu'], t['gaia_sigma_mu'][j]
        mapped = d['mu_max']>d['mu_min']
        inside = mapped & (mu>=d['mu_min']) & (mu<=d['mu_max'])
        retained = mapped & (d['base_probability']>=.5)
        converged = d['iteration_state']==5
        clean = ~(t['qrf_fit_overlap'][j] | t['historical_overlap'][j])
        common = inside & retained & converged & clean
        block = env.covariance_blocks(t)[j]
        field = t['field'][j]
        idx_all = np.flatnonzero(common)
        arms = ((p['mu_edges'][at[idx_all]],p['mass'][at[idx_all]]),
                (d['base_edges'][idx_all],d['base_mass'][idx_all]),
                (d['iteration_edges'][idx_all],d['iteration_mass'][idx_all]))
        stats = [pdf_stats(e,w,mu[idx_all],sig[idx_all]) for e,w in arms]
        color = (t['mag_grz'][:,0]-t['mag_grz'][:,1])[j]
        ref_kpc = 10**((mu-10)/5)
        for name in np.unique(field):
            f = field==name
            parent = (t['field']==name)&t['gaia_evaluation']
            flow.append(dict(field=name, candidate_gaia=int(parent.sum()), registered=int(f.sum()),
                 map_valid=int((f&mapped).sum()), in_map=int((f&inside).sum()),
                 sfd_half_mass=int((f&retained).sum()), converged=int((f&converged).sum()),
                 common_before_history=int((f&inside&retained&converged).sum()), common=int((f&common).sum()),
                 **{f'state_{k}':int((f&(d['iteration_state']==k)).sum()) for k in range(6)}))
            o = np.flatnonzero(f & retained & inside & (d['oracle_state']==3) & clean)
            if len(o):
                b = modulus_quantiles(d['base_mass'][o],d['base_edges'][o])[:,2]-mu[o]
                z = d['oracle_center'][o]-mu[o]
                oracle.append(dict(field=name,n=len(o),blocks=len(np.unique(block[o])),
                              B_bias=np.median(b),D_bias=np.median(z)))
            strata = [('all','all',f)]
            for group in np.unique(t['gaia_proxy'][j]):
                strata.append(('class_proxy',group,f&(t['gaia_proxy'][j]==group)))
            for axis,x,edges in [('r',t['r'][j],(0,18,20,21,22,23)),
                    ('gr',color,(-3,.5,.8,1.4,5)),('kpc',ref_kpc,(0,2,5,10,40)),
                    ('ebv',t['EBV'][j],(0,.1,.3,.6,20))]:
                for lo,hi in zip(edges,edges[1:]):
                    strata.append((axis,f'{lo:g}-{hi:g}',f&(x>=lo)&(x<hi)))
            for axis,label,take in strata:
                i = np.flatnonzero(take[idx_all])
                stem = dict(field=name,axis=axis,stratum=label,registered=int(take.sum()),
                            n=len(i), blocks=len(np.unique(block[idx_all][i])))
                if not len(i):
                    rows.append(stem)
                    continue
                draws = np.asarray(list(bootstrap(stats,i,block[idx_all],rng))).reshape(-1,3,3)
                for k,arm in enumerate(('A_full','B_map_range','C_iteration')):
                    cl, ch = np.quantile(draws[:,k,2],[.025,.975])
                    bl, bh = np.quantile(draws[:,k,1],[.025,.975])
                    rows.append(dict(stem,arm=arm,**reduce(stats[k],i),bias_lo=bl,bias_hi=bh,
                                     crps_minus_B=float(np.mean(stats[k]['crps'][i]-stats[1]['crps'][i])),
                                     crps_delta_lo=cl,crps_delta_hi=ch))
    for name, data in [('environment_metrics',rows),('environment_flow',flow),('environment_oracle',oracle)]:
        csv_rows(D/(name+'.csv'),data)
    save_json(D/'complete.json',dict(job=os.environ.get('SLURM_JOB_ID'), inputs=inputs,
        outputs={p.name:sha(p) for p in D.glob('*.csv')}))


if __name__=='__main__':
    main()
