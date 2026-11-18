#!/usr/bin/env python3
"""Same-star known-system tests of QRF, label-grid and a fixed old native sequence."""
from scipy.spatial import cKDTree
from photdis.baseline_v1 import qmc_noise
from photdis.stage1 import require_compute
from halo_common import *
from sv18_common import cached,index_join,metric,interval
from sv18_benchmark import fit,predict
from v3_9_catalog import from_north
from v3_13_clusters import SYSTEMS


def relation(template,key):
    m=template[key];use=(template['phase']==0)&(m[:,1]>=3)&(m[:,1]<=8)
    if key=='north':use&=template['north_support']
    gr=m[use,0]-m[use,1];mg=m[use,0]
    order=np.argsort(gr);gr=gr[order];mg=mg[order]
    _,j=np.unique(np.round(gr,6),return_index=True)
    return gr[j],mg[j]


def old_sequence(m,e,ids,curve):
    gr,mg=curve
    mus=[]
    for draw in range(128):
        noise=qmc_noise(ids,draw)[:,:3]
        pert=m+noise*e
        color=pert[:,0]-pert[:,1]
        lum=np.interp(color,gr,mg,left=np.nan,right=np.nan)
        # Fixed 0.10-mag intrinsic scatter, using an independent Sobol dimension.
        mu=pert[:,0]-lum+qmc_noise(ids,draw)[:,3]*.10
        mus.append(mu)
    mus=np.array(mus).T
    count=np.isfinite(mus).sum(axis=1)
    with np.errstate(invalid='ignore'):
        quant=np.nanquantile(mus,[.025,.16,.5,.84,.975],axis=1).T
    q=10**((quant-10)/5)
    q[count<64]=np.nan
    return q,count


def main():
    require_compute();d=folder('sequence');inputs={};pieces=[]
    for name in ('M5','M13','M30','M92'):
        t=read(ROOT/f'data/v3_14/audit/{name}',('native_join.npz','TYPE','ls_id','source_id','r','g0','r0','z0',
          'error_grz','qrf_kpc','member','target','training_or_calibration','low_crowding','cluster_mask','phase_box'),inputs)
        ref=7.480 if name=='M5' else SYSTEMS[name][1]
        mu=10+5*np.log10(ref)
        m=np.column_stack((t['g0'],t['r0'],t['z0']))
        clean=t['target']&t['member']&~t['training_or_calibration']&t['low_crowding']
        # Match the previously reported low-crowding sample; mask retained per row.
        domain=clean&(m[:,1]-mu>=3)&(m[:,1]-mu<=8)
        north=name in ('M13','M92')
        i=np.flatnonzero(domain)
        pieces.append(dict(TYPE=t['TYPE'][i],ls_id=t['ls_id'][i],source_id=t['source_id'][i],
          system=np.full(len(i),name),native=np.full(len(i),north,'u1'),m=m[i],err=t['error_grz'][i],
          r=t['r'][i],q=t['qrf_kpc'][i],ref=np.full(len(i),ref),phase=t['phase_box'][i],mask=t['cluster_mask'][i]))
    p=ROOT/'data/v3_18/members';z=read(p,('posteriors.npz','TYPE','ls_id','science','quantiles_kpc'),inputs)
    record=json.loads((p/'complete.json').read_text())
    with verified(p/'roster.csv',record['outputs']['roster.csv']).open() as f:roster={int(r['ls_id']):r for r in csv.DictReader(f)}
    cache=cached(inputs,('TYPE','ls_id','source_id','m','err','region'))
    jj=index_join(z['ls_id'],cache['ls_id'])
    for group in ('thin','cocoon'):
        rows=np.array([i for i in range(len(jj)) if jj[i]>=0 and z['science'][i] and roster[int(z['ls_id'][i])]['group']==group],int)
        rr=[roster[int(z['ls_id'][i])] for i in rows];j=jj[rows]
        ref=np.array([float(r['desi_kpc']) for r in rr]);r=np.array([float(r['r']) for r in rr])
        mr=cache['m'][j,1]-10-5*np.log10(ref);use=(mr>=3)&(mr<=8)&(ref>0)
        rows=rows[use];j=j[use];ref=ref[use];r=r[use]
        pieces.append(dict(TYPE=z['TYPE'][rows],ls_id=z['ls_id'][rows],source_id=cache['source_id'][j],
            system=np.full(len(j),'GD1_'+group),native=(cache['region'][j]==3).astype('u1'),m=cache['m'][j],err=cache['err'][j],
            r=r,q=z['quantiles_kpc'][rows],ref=ref,phase=np.full(len(j),'MS/TO'),mask=np.zeros(len(j),bool)))
    t={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};assert (t['TYPE']=='PSF').all()
    identity=np.where(t['source_id']>0,t['source_id'],-t['ls_id'])
    template=read(OUT/'native',('template.npz','south','north','phase','north_support'),inputs)
    simple=np.full_like(t['q'],np.nan);seq=simple.copy();support=np.zeros(len(seq),int);training={}
    for code,side in enumerate(('south','north')):
        train=read(ROOT/'data/science_validation_20261118/benchmark',(side+'_training_sample.npz','TYPE','ls_id','source_id','m','y','class_label'),inputs)
        training[side]=train;model=fit(train)
        offset=0.
        if code:modeln,hh=from_north();offset=modeln['mu_offset'];inputs['north_model']=hh
        ii=np.flatnonzero(t['native']==code)
        for start in range(0,len(ii),256):
            j=ii[start:start+256]
            simple[j],_,_=predict(model,t['m'][j],t['err'][j],identity[j],offset,draws=128)
            seq[j],support[j]=old_sequence(t['m'][j],t['err'][j],identity[j],relation(template,side))
    methods={'QRF':t['q'],'Label grid':simple,'Old sequence':seq}
    common=np.logical_and.reduce([np.isfinite(q).all(axis=1)&(q>0).all(axis=1) for q in methods.values()])
    rows=[];cohorts=[]
    for name in np.unique(t['system']):
        for low,high in ((14,18),(18,20),(20,21),(21,22),(22,23),(14,23)):
            parent=(t['system']==name)&(t['r']>=low)&(t['r']<high)
            for quality,keep in (('unmasked_low_crowding',~t['mask']),('all_low_crowding',np.ones(len(common),bool))):
                use=parent&common&keep
                cohorts.append(dict(system=name,r_lo=low,r_hi=high,quality=quality,parent=int((parent&keep).sum()),common=int(use.sum()),
                    qrf_supported=int((parent&keep&np.isfinite(t['q']).all(axis=1)).sum()),
                    sequence_supported=int((parent&keep&np.isfinite(seq).all(axis=1)).sum()),masked_common=int((use&t['mask']).sum())))
                if not use.any():continue
                for method,q in methods.items():
                    stat=metric(q[use],t['ref'][use]);res=q[use,2]/t['ref'][use]-1
                    ci=interval([np.median(res[RNG.integers(len(res),size=len(res))]) for _ in range(256)])
                    rows.append(dict(system=name,r_lo=low,r_hi=high,quality=quality,method=method,**stat,**ci,
                        bias_pass=bool(len(res)>=30 and abs(stat['bias'])<=.1),
                        role='test with prior development ancestry',reference_scope='common system scale, not independent per-star truth'))
    phases=[]
    for name in np.unique(t['system']):
        for phase in np.unique(t['phase'][t['system']==name]):
            for quality,keep in (('unmasked_low_crowding',~t['mask']),('all_low_crowding',np.ones(len(common),bool))):
                use=common&(t['system']==name)&(t['phase']==phase)&keep
                if not use.any():continue
                for method,q in methods.items():
                    phases.append(dict(system=name,phase=phase,quality=quality,method=method,**metric(q[use],t['ref'][use])))
    csv_rows(d/'sequence_phase_scale.csv',phases)
    atomic_npz(d/'same_source_predictions.npz',**t,simple=simple,sequence=seq,sequence_draw_support=support,common=common)
    csv_rows(d/'sequence_scale.csv',rows);csv_rows(d/'sequence_support.csv',cohorts)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,3,figsize=(7.15,5.7),layout='constrained')
        shown=[]
        # Independent counts are given in the sidecar; all three CMDs use identical members.
        for col,name in enumerate(('M5','M13','GD1_thin')):
            ii=np.flatnonzero(common&~t['mask']&(t['system']==name))
            color=t['m'][ii,0]-t['m'][ii,1]
            mu=10+5*np.log10(t['ref'][ii]);qmu=10+5*np.log10(t['q'][ii,2])
            for row,values in enumerate((t['m'][ii,1]-mu,t['m'][ii,1]-qmu)):
                ax=axes[row,col];ax.scatter(color,values,s=5,lw=0,alpha=.6,color=BLUE,rasterized=True)
                native=int(np.bincount(t['native'][ii],minlength=2).argmax()) if len(ii) else int(name=='M13')
                mag=template[('south','north')[native]];take=(template['phase']==0)|(template['phase']==2)
                if native:take&=template['north_support']
                ax.plot(mag[take,0]-mag[take,1],mag[take,1],color=ORANGE,lw=.8)
                ax.set(xlim=(-.1,1),ylim=(8.5,1.5),xlabel=r'$(g-r)_0$',ylabel=r'$M_r$ (mag)',title=name.replace('_',' ')+' · '+('reference' if row==0 else 'QRF'))
                shown.append(dict(system=name,coordinate=('reference','QRF')[row],n=len(ii),outside=int(((color<-.1)|(color>1)|(values<1.5)|(values>8.5)).sum())))
        figure(d,'known_sequence_same_source_cmd',f,shown)
        f,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
        for code,side in enumerate(('south','north')):
            tr=training[side];col=tr['m'][:,0]-tr['m'][:,1]
            from matplotlib.colors import LogNorm
            h,xe,ye=np.histogram2d(col,tr['y'],(np.linspace(-.1,1.1,80),np.linspace(1,12,100)))
            z=axes[code].pcolormesh(xe,ye,np.ma.masked_equal(h.T,0),cmap='viridis',norm=LogNorm(1,max(2,h.max())),rasterized=True)
            mag=template[side];take=template['phase']==0
            if code:take&=template['north_support']
            axes[code].plot(mag[take,0]-mag[take,1],mag[take,0],color=ORANGE,lw=1)
            axes[code].set(xlabel=r'$(g-r)_0$',ylabel=r'$M_g$ training label (mag)',ylim=(12,1),title=side.capitalize())
            f.colorbar(z,ax=axes[code],fraction=.035,label='Training stars / bin',extend='max')
        figure(d,'training_luminosity_old_sequence',f,[dict(native=s,n=len(t['ls_id']),inside_axes=int(np.histogram2d(t['m'][:,0]-t['m'][:,1],t['y'],(np.linspace(-.1,1.1,80),np.linspace(1,12,100)))[0].sum())) for s,t in training.items()])
        f,axes=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
        for ax,names in zip(axes,(('M5','M13','M30','M92'),('GD1_thin','GD1_cocoon'))):
            for method,color in zip(methods,(BLUE,GREEN,ORANGE)):
                a=[r for r in rows if r['system'] in names and r['quality']=='unmasked_low_crowding' and r['r_lo']!=14 and r['r_hi']<=23]
                for name in names:
                    z=[r for r in a if r['method']==method and r['system']==name]
                    if z:ax.plot([(r['r_lo']+r['r_hi'])/2 for r in z],[r['bias'] for r in z],
                        marker='o' if name==names[0] else 's',ms=3,lw=.8,color=color,label=method if name==names[0] else None)
            ax.axhspan(-.1,.1,color='.85',alpha=.4);ax.axhline(0,color='.5',ls=':',lw=.6)
            ax.set(xlabel='Observed r (mag)',ylabel='Median relative-distance residual',title='Clusters' if len(names)>2 else 'GD-1')
        axes[0].legend(frameon=False,fontsize=7)
        figure(d,'known_sequence_scale_comparison',f,rows)
    complete(d,inputs,relation='fixed MIST12Gyr Z=.0001 phase0, reference Mr3..8; monotone color interpolation, no zero-point fitting',
        uncertainty='128 shared-band Sobol draws plus fixed 0.10mag intrinsic sequence scatter; 256 member bootstrap CI conditional on each common reference scale',
        reference_systematics='external common-distance errors are not independent member noise and are not reduced by sqrt(N); reported CIs condition on reference centers',
        history='M13/M92 sky regions occurred in northern model construction; all systems previously inspected; no fresh blind claim')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--cmd-only',action='store_true')
    args=parser.parse_args()
    if not args.cmd_only:main()
    else:
        require_compute();d=folder('sequence');marker=d/'complete.json';r=json.loads(marker.read_text())
        archive=folder('execution')/'sequence_before_native_plot_revision.json'
        if not archive.exists():archive.write_text(marker.read_text())
        with np.load(verified(d/'same_source_predictions.npz',r['outputs']['same_source_predictions.npz'])) as a:
            t={k:a[k] for k in a.files}
        template=read(OUT/'native',('template.npz','south','north','phase','north_support'),{})
        with plt.rc_context(PAPER):
            f,axes=plt.subplots(2,3,figsize=(7.15,5.7),layout='constrained');shown=[]
            for col,name in enumerate(('M5','M13','GD1_thin')):
                ii=np.flatnonzero(t['common']&~t['mask']&(t['system']==name))
                color=t['m'][ii,0]-t['m'][ii,1];mu=10+5*np.log10(t['ref'][ii]);qmu=10+5*np.log10(t['q'][ii,2])
                for row,values in enumerate((t['m'][ii,1]-mu,t['m'][ii,1]-qmu)):
                    ax=axes[row,col];ax.scatter(color,values,s=5,lw=0,alpha=.6,color=BLUE,rasterized=True)
                    codes=np.unique(t['native'][ii])
                    for native in codes:
                        mag=template[('south','north')[native]];take=np.isin(template['phase'],(0,2))
                        if native:take&=template['north_support']
                        ax.plot(mag[take,0]-mag[take,1],mag[take,1],color=ORANGE,lw=.8,ls='-' if len(codes)==1 or native==0 else '--')
                    ax.set(xlim=(-.1,1),ylim=(8.5,1.5),xlabel=r'$(g-r)_0$',ylabel=r'$M_r$ (mag)',title=name.replace('_',' ')+' · '+('reference' if row==0 else 'QRF'))
                    shown.append(dict(system=name,coordinate=('reference','QRF')[row],n=len(ii),
                        native_south=int((t['native'][ii]==0).sum()),native_north=int((t['native'][ii]==1).sum()),
                        outside=int(((color<-.1)|(color>1)|(values<1.5)|(values>8.5)).sum())))
            figure(d,'known_sequence_same_source_cmd',f,shown)
        r['figure_revision']=dict(code=sha(Path(__file__)),parent_complete=sha(archive),
            scope='plot-only exact same saved members/predictions, native template selected from actual source metadata')
        for name in r['outputs']:r['outputs'][name]=sha(d/name)
        save_json(marker,r)

