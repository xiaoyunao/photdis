#!/usr/bin/env python3
"""Rematch metal-poor and ordinary dwarfs inside each sky-block resample."""
import numpy as np
import argparse
from photdis.stage1 import require_compute
from sv18_common import *
from paper_v2_scan import R, GR, RZ, bin_axis


def cells(t, variant, distance):
    gr=t['m'][:,0]-t['m'][:,1]
    rz=t['m'][:,1]-t['m'][:,2]
    gedges=GR if variant!='double_color_width' else GR[::2]
    zedges=RZ if variant!='double_color_width' else RZ[::2]
    x=np.column_stack((t['region'],bin_axis(t['observed_r'],R),bin_axis(gr,gedges),
        bin_axis(rz,zedges),bin_axis(t['EBV'],(-np.inf,.01,.02,.03,np.inf)),
        bin_axis(t['err'].max(axis=1),(-np.inf,.045,.09,.16,np.inf)))).astype('i2')
    if variant=='r_split':
        x=np.column_stack((x,bin_axis(t['observed_r'],(-np.inf,18,19,20,21,22,23,np.inf)),
                                 bin_axis(distance,(0,2,3.5,5,7.5,10,np.inf))))
    return x


def match(x,pop,delta,weight=None):
    weight=np.ones(len(pop),'i4') if weight is None else weight
    a,labels=np.unique(x,axis=0,return_inverse=True)
    cells=[];bal=np.zeros(len(pop))
    row_order=np.argsort(labels,kind='stable')
    boundaries=np.r_[0,np.flatnonzero(np.diff(labels[row_order]))+1,len(row_order)]
    for k,(start,stop) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        local=row_order[start:stop]
        j=[local[(pop[local]==p)&(weight[local]>0)] for p in (1,2)]
        n=[int(weight[i].sum()) for i in j]
        if min(n)<P['minimum_matched_cell']:
            continue
        m=[]
        for i in j:
            value_order=np.argsort(delta[i],kind='stable')
            values=delta[i][value_order]; cum=np.cumsum(weight[i][value_order])
            size=cum[-1]
            at=np.searchsorted(cum,[(size-1)//2+1,size//2+1])
            m.append(values[at].mean())
        w=min(n)
        cells.append((k,w,m[1]-m[0],*n))
        for i in j:
            bal[i]=w*weight[i]/weight[i].sum()
    a=np.asarray(cells)
    if not len(a):
        return np.nan,[],bal
    return np.average(a[:,2],weights=a[:,1]),cells,bal


def main():
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exclude-calibration',action='store_true')
    args=parser.parse_args()
    inputs={}
    keys=('TYPE','ls_id','RA','DEC','region','observed_r','m','EBV','err','pop','state','heldout','q',
          'gaia_stage','gaia_d','specdis_stage','specdis_d','spdist_stage','spdist_d')
    t=cached(inputs,keys)
    use=t['heldout']&(t['state']==3)&np.isin(t['pop'],(1,2))
    t={k:v[use] for k,v in t.items()}
    excluded=0
    if args.exclude_calibration:
        mask=read(DEST/'provenance',('additional_exclusions.npz','TYPE','ls_id',
                  'additional_calibration_or_selection_overlap'),inputs)
        ix=index_join(t['ls_id'],mask['ls_id'])
        reject=np.zeros(len(ix),bool)
        known=ix>=0
        reject[known]=mask['additional_calibration_or_selection_overlap'][ix[known]]
        excluded=int(reject.sum())
        t={key:value[~reject] for key,value in t.items()}
    block=sky(t);d=out('populations_clean' if args.exclude_calibration else 'populations')
    result=[];drawtable=[];balance=[]
    for ref in ('specdis','spdist','gaia'):
        dist=t[ref+'_d']
        for lo,hi in ((2,5),(5,10)):
            base=(t[ref+'_stage']==3)&(dist>=lo)&(dist<hi)
            idx=np.flatnonzero(base)
            if not len(idx):
                continue
            res=5*np.log10(t['q'][idx,2]/dist[idx])
            pp=t['pop'][idx]; bb=block[idx]
            blocks=np.unique(bb)
            for variant in P['matched_sensitivity']:
                x=cells(t,variant,dist)[idx]
                point,matched,bw=match(x,pp,res)
                if not args.exclude_calibration and variant=='original' and ref in ('specdis','spdist'):
                    with (PRIOR/'figures/matched_physical_conditions.csv').open() as f:
                        old=[r for r in csv.DictReader(f) if r['reference']==ref and float(r['d_lo'])==lo]
                    sums={}
                    for r in old:
                        if int(r['common'])<5:
                            continue
                        key=tuple(r[k] for k in ('region','r_bin','gr_bin','rz_bin','ebv_bin','error_bin'))
                        sums.setdefault(key,{})[r['pop']]=r
                    pairs=list(sums.values())
                    previous=np.average([float(q['metal_poor_dwarf']['mu_median'])-float(q['ordinary_dwarf']['mu_median']) for q in pairs],
                             weights=[int(q['ordinary_dwarf']['common']) for q in pairs])
                    np.testing.assert_allclose(point,previous,atol=2e-6)
                    assert len(matched)==len(pairs)
                draws=[]
                inv=np.searchsorted(blocks,bb)
                if len(blocks)>1 and matched:
                    for draw in range(P['bootstrap_draws']):
                        mult=np.bincount(RNG.integers(len(blocks),size=len(blocks)),minlength=len(blocks))
                        val,cc,_=match(x,pp,res,mult[inv])
                        draws.append(val)
                        drawtable.append(dict(reference=ref,lo=lo,variant=variant,draw=draw,
                            shift_mu=val,common_cells=len(cc),sum_weights=sum(r[1] for r in cc)))
                eff=(bw.sum()**2/(bw@bw)) if bw.any() else 0
                weights=np.array([r[1] for r in matched])
                result.append(dict(reference=ref,lo=lo,hi=hi,variant=variant,n=len(idx),
                    ordinary=int((pp==1).sum()),metal_poor=int((pp==2).sum()),
                    common_cells=len(matched),sum_weights=weights.sum(),effective_rows=eff,
                    max_cell_weight=weights.max() if len(weights) else 0,
                    common_row_fraction=(bw>0).mean(),shift_mu=point,
                    distance_scale_ratio=10**(point/5),sky_blocks=len(blocks),**interval(draws)))
                if variant=='original':
                    for name,value in [('r',t['observed_r'][idx]),('gr',t['m'][idx,0]-t['m'][idx,1]),
                           ('rz',t['m'][idx,1]-t['m'][idx,2]),('EBV',t['EBV'][idx]),('error',t['err'][idx].max(axis=1))]:
                        edges=np.linspace(*{'r':(14,23),'gr':(-.1,1.5),'rz':(-.3,1.2),'EBV':(0,.05),'error':(0,.2)}[name],31)
                        for pop in (1,2):
                            h=np.histogram(value[pp==pop],edges)[0]
                            w=np.histogram(value[pp==pop],edges,weights=bw[pp==pop])[0]
                            balance.extend(dict(reference=ref,lo=lo,variable=name,population=CLASSES[pop],
                                bin_lo=a,bin_hi=b,before_n=n,matched_weight=c)
                                for a,b,n,c in zip(edges[:-1],edges[1:],h,w))
    csv_rows(d/'matched_metallicity.csv',result)
    csv_rows(d/'sky_rematch_draws.csv',drawtable)
    csv_rows(d/'observable_balance.csv',balance)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.15),layout='constrained',sharey=True)
        for ax,lo in zip(axes,(2,5)):
            for j,ref in enumerate(('specdis','spdist','gaia')):
                a=[r for r in result if r['reference']==ref and r['lo']==lo]
                for i,r in enumerate(a):
                    if not np.isfinite(r['shift_mu']):
                        continue
                    y=r['shift_mu'];l=r['ci_lo'];u=r['ci_hi']
                    ax.errorbar(i+(j-1)*.15,y,yerr=None if l is None else [[max(0,y-l)],[max(0,u-y)]],
                       fmt=('o','s','^')[j],ms=4,color=(BLUE,ORANGE,GREEN)[j],
                       label=REFERENCE_NAMES[ref] if i==0 else None)
            ax.axhline(0,color='.6',lw=.7,ls=':')
            ax.set(xticks=range(3),xticklabels=['Original','Wider colors','Finer r / d'],
                   title=f'{lo}–{5 if lo==2 else 10} kpc')
        axes[0].set_ylabel(r'Metal-poor minus ordinary $\Delta\mu$ (mag)')
        axes[0].legend(fontsize=7,frameon=False)
        figure(d,'matched_metallicity',f,result)
    complete(d,inputs,exclude_calibration=args.exclude_calibration,excluded_sources=excluded)


if __name__=='__main__':
    main()
