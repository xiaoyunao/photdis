#!/usr/bin/env python3
"""Audit Gaia uncertainty units and exact working-selection distance bins."""
import numpy as np
from photdis.stage1 import require_compute
from sv18_common import *
from paper_v2_reference import bins, BIN_AXES, reference_uncertainty_mu
from paper_v2_scan import REGIONS
from v3_10_science import WORK


def uncertainty(t,d,inputs):
    with (PRIOR/'reference/distance_metrics.csv').open() as f:
        old=list(csv.DictReader(f))
    rows=[]
    qualified=(t['gaia_stage']==3)&(t['state']==3)
    t={key:value[qualified] for key,value in t.items()}
    rc=bins(t,t['gaia_d'])
    for r in old:
        ref=r['reference']
        if ref!='gaia':
            continue
        keep=(t['gaia_stage']==3)&(t['state']==3)
        if r['frame']=='heldout':
            keep &= t['heldout']
        for j,k in enumerate(BIN_AXES):
            if r[k] and r[k]!='None':
                keep &= rc[:,j]==int(r[k])
        q,refd=t['q'][keep],t['gaia_d'][keep]
        met=metric(q,refd,t['peaks'][keep])
        for k in ('n','bias','nmad','severe','c68','c95'):
            np.testing.assert_allclose(met[k],float(r[k]),rtol=2e-6,atol=2e-7)
        sigma,half=reference_uncertainty_mu(t,'gaia',keep)
        rows.append(dict(**{k:r[k] for k in ('frame','grouping',*BIN_AXES)},
            n=int(keep.sum()),legacy_median_reference_sigma_mu=float(r['median_reference_sigma_mu']),
            median_reference_halfwidth95_mu=np.median(half),
            median_reference_sigma_mu_linear=np.median(sigma),
            median_reference_sigma_mu=np.median(sigma),
            median_reference_halfwidth95_div_1p96=np.median(half/1.96),
            median_halfwidth95_to_linear_sigma=np.median(half/sigma),centers_unchanged=True))
    csv_rows(d/'gaia_reference_uncertainty.csv',rows)
    usage=[]
    for p in (ROOT/'scripts').glob('*.py'):
        if 'median_reference_sigma_mu' in p.read_text():
            usage.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),
                   role='new explicit scale audit' if p.name.startswith('sv18_') else 'historical writer'))
    csv_rows(d/'uncertainty_field_usage.csv',usage)


def working(t,d,inputs):
    score=np.full(len(t['ls_id']),np.nan,'f4')
    rm=np.full((len(score),2),np.nan,'f4')
    wstate=np.zeros(len(score),'u1')
    assessed=np.zeros(len(score),bool)
    for side,region in (('north',3),('south',2)):
        k=np.flatnonzero((t['region']==region)&(t['observed_r']<20)&(t['quality']==7))
        for marker in sorted((WORK/side).glob('[0-9][0-9].json')):
            meta=json.loads(marker.read_text())
            if not meta['rows']:
                continue
            name=marker.stem+'.npz'
            a=read_columns(verified(marker.with_name(name),meta['outputs'][name]),
                           ('TYPE','ls_id','score','radial_mass','state'))
            assert (a['TYPE']=='PSF').all()
            cross=index_join(t['ls_id'][k],a['ls_id'])
            use=cross>=0; idx=k[use]; j=cross[use]
            assert not assessed[idx].any()
            assessed[idx]=True; score[idx]=a['score'][j]; rm[idx]=a['radial_mass'][j]; wstate[idx]=a['state'][j]
            inputs[str(marker.relative_to(ROOT))]=sha(marker)
    domain=(t['observed_r']<20)&np.isin(t['region'],(2,3))&(t['quality']==7)
    assert assessed[domain].all()
    supported=t['state']==3
    cumulative=[domain&supported,domain&supported&(score>=.8),
        domain&supported&(score>=.8)&(rm.sum(axis=1)>=.5)]
    np.testing.assert_array_equal(cumulative[-1][domain],wstate[domain]==4)
    rows=[]; step=[]
    block=sky(t)
    for ref in ('specdis','spdist'):
        for pop in (1,2,4,5,8):
            for lo,hi in ((0,2),(2,5),(5,10),(10,np.inf)):
                base=(t[ref+'_stage']==3)&t['heldout']&(t['pop']==pop)&(t[ref+'_d']>=lo)&(t[ref+'_d']<hi)
                den=base&domain&supported
                n=int(den.sum())
                passed=base&cumulative[-1]
                pred=t['q'][:,2]
                rows.append(dict(reference=ref,population=CLASSES[pop],ref_lo=lo,ref_hi=hi,
                    parent=int(base.sum()),supported=int((base&supported).sum()),applicable=n,
                    passed_score=int((base&cumulative[1]).sum()),passed_probability=int(passed.sum()),
                    selected_2_5=int((passed&(pred>=2)&(pred<5)).sum()),
                    selected_5_10=int((passed&(pred>=5)&(pred<10)).sum()),
                    hard_2_5=int((den&(pred>=2)&(pred<5)).sum()),
                    soft_2_5=np.nansum(rm[den,0]),soft_5_10=np.nansum(rm[den,1]),
                    sky_blocks=len(np.unique(block[den]))))
                for name,cut in [('support',cumulative[0]),('score',cumulative[1]),('probability',cumulative[2]),
                        ('predicted_2_5',cumulative[2]&(pred>=2)&(pred<5))]:
                    selected=(cut[den]).astype(float)
                    bb=block[den]; keys=np.unique(bb)
                    draws=[]
                    if n and len(keys)>1:
                        nums=np.array([selected[bb==k].sum() for k in keys]); ns=np.array([(bb==k).sum() for k in keys])
                        for _ in range(P['bootstrap_draws']):
                            ix=RNG.integers(len(keys),size=len(keys));draws.append(nums[ix].sum()/ns[ix].sum())
                    step.append(dict(reference=ref,population=CLASSES[pop],ref_lo=lo,ref_hi=hi,stage=name,
                        parent=n,selected=int(selected.sum()),retention=selected.mean() if n else None,
                        sky_blocks=len(keys),**interval(draws)))
    csv_rows(d/'working_selection.csv',rows); csv_rows(d/'selection_block_intervals.csv',step)
    trends=[]
    variables=(('feh',t['physical'][:,0],(-3.5,-2.5,-2.,-1.5,-1.,-.5,0,.5)),
               ('logg',t['physical'][:,2],(0,2,3.2,3.7,4.2,4.6,5.5)),
               ('gr',t['m'][:,0]-t['m'][:,1],(-.1,.3,.5,.65,.8,1.1,1.5)))
    for ref in ('specdis','spdist'):
        parent=(t[ref+'_stage']==3)&t['heldout']&domain&supported
        for variable,value,edges in variables:
            for lo,hi in zip(edges[:-1],edges[1:]):
                eligible=parent&(value>=lo)&(value<hi)
                if variable in ('feh','logg'):
                    error=t['physical'][:,1 if variable=='feh' else 3]
                    eligible&=(error>0)&(error<=.3)
                n=int(eligible.sum())
                idx=np.flatnonzero(eligible)
                b,inv=np.unique(block[idx],return_inverse=True)
                ns=np.bincount(inv,minlength=len(b))
                for name,cut in [('score',cumulative[1]),('probability',cumulative[2]),
                        ('predicted_2_5',cumulative[2]&(pred>=2)&(pred<5))]:
                    selected=cut[idx]
                    num=np.bincount(inv,weights=selected,minlength=len(b))
                    draws=[]
                    if n and len(b)>1:
                        pick=RNG.integers(len(b),size=(P['bootstrap_draws'],len(b)))
                        draws=num[pick].sum(axis=1)/ns[pick].sum(axis=1)
                    trends.append(dict(reference=ref,variable=variable,lo=lo,hi=hi,stage=name,
                       applicable=n,selected=int(selected.sum()),retention=selected.sum()/n if n else None,
                       sky_blocks=len(b),**interval(draws)))
    csv_rows(d/'selection_physical_trends.csv',trends)
    contaminants=[]
    for region in np.unique(t['region']):
        base=(t['region']==region)&(t['pop']==8)&t['heldout']
        for lo,hi in ((0,2),(2,5),(5,10),(10,np.inf)):
            k=base&supported&(pred>=lo)&(pred<hi)
            contaminants.append(dict(region=int(region),reference='qualified Redrock PSFs',lo=lo,hi=hi,
                qualified_parent=int(base.sum()),supported=int((base&supported).sum()),
                hard_layer=int(k.sum()),score_pass=int((k&cumulative[1]).sum()),
                probability_pass=int((k&cumulative[2]).sum()),
                imaging_contamination_fraction=None))
    csv_rows(d/'nonstellar_distance_tails.csv',contaminants)
    anchor=[r for r in rows if r['ref_lo']==2 and r['population'] in ('ordinary_dwarf','metal_poor_dwarf','intermediate_gravity')]
    assert [r['selected_2_5'] for r in anchor]==[15827,219,61,14584,177,53]
    atomic_npz(d/'working_reference_rows.npz',ls_id=t['ls_id'],TYPE=t['TYPE'],assessed=assessed,
               score=score,radial_mass=rm,state=wstate)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.15),layout='constrained',sharey=True)
        for ax,ref in zip(axes,('specdis','spdist')):
            for pop,color in zip((1,2,4),(BLUE,ORANGE,GREEN)):
                z=[r for r in step if r['reference']==ref and r['population']==CLASSES[pop] and r['ref_lo']==2]
                y=np.array([r['retention'] for r in z]); lower=np.array([r['ci_lo'] or r['retention'] for r in z])
                upper=np.array([r['ci_hi'] or r['retention'] for r in z])
                ax.errorbar(range(4),y,yerr=np.maximum(0,[y-lower,upper-y]),fmt='o-',color=color,
                            label=CLASSES[pop].replace('_',' ').capitalize(),ms=3)
            ax.set(ylim=(0,1.04),xticks=range(4),xticklabels=['PDF','Score','Probability','2–5 kpc'],title=ref.capitalize())
        axes[0].set_ylabel('Cumulative retained fraction');axes[0].legend(fontsize=6,frameon=False)
        figure(d,'selection_chemistry',f,step)


def main():
    require_compute()
    inputs={};t=cached(inputs)
    relevant=(t['gaia_stage']==3)|(t['specdis_stage']==3)|(t['spdist_stage']==3)|(t['pop']==8)
    t={key:value[relevant] for key,value in t.items()}
    d=out('definitions')
    uncertainty(t,d,inputs)
    working(t,d,inputs)
    complete(d,inputs,native_rows=len(t['ls_id']),distance_centers_preserved=True)


if __name__=='__main__':
    main()
