#!/usr/bin/env python3
"""Source-bound legacy replay and independent integration accuracy checks."""
from seq7_common import *
from seq7_predict import load
from seq7_math import evaluate_mix,uniform_curve,curve_crps,row_interp
from sky6_width import mixture_probability,uniform_convolution
from stream_sequence import count_median
from photdis.stage1 import require_compute
from scipy.special import logsumexp


def legacy():
    d=folder('validation');t=load(OUT/'prepare/roster.npz');old=list(csv.DictReader((OLD/'width/center_width.csv').open()));cal=json.loads((STREAM/'sequence/calibration.json').read_text());delta=cal['delta_Mg'];dd=np.array(cal['zero_point_draws']);rows=[]
    for s in C['systems']:
        a=load(OUT/'predict'/f'{s}_A_4.npz');z=load(OUT/'predict'/('QRF_north.npz' if s=='M13' else 'QRF_south.npz'));shift=5*np.log10(t['scale_'+s]);rs=np.std(shift,ddof=1)
        for phase in ('all','lower_MS','turnoff','other'):
            ii=np.flatnonzero((t['system']==s)&t['v5']&((t['phase']==phase) if phase!='all' else True));ref=10+5*np.log10(t['ref'][ii]);mult=t['v5_mult'][:,ii];j=np.searchsorted(a['ii'],ii);mm=a['means'][j];ww=a['logw'][j];sg=a['sigma'][j]
            for method,key in [('QRF','q'),('Fixed sequence','legacy_new'),('Calibrated sequence','legacy_calibrated')]:
                qs=10+5*np.log10(t[key][ii]);res=qs[:,2]-ref;center=np.median(res);centers=np.array([count_median(res-ss+(delta-dd[k] if key=='legacy_calibrated' else 0),mult[k]) for k,ss in enumerate(shift)])
                if method=='QRF':
                    at=np.searchsorted(z['ii'],ii);u,lp=uniform_convolution(z['edges'][at],z['mass'][at],ref,rs);uc,lc=uniform_convolution(z['edges'][at],z['mass'][at],ref+center,rs)
                else:
                    LP=[];LC=[];U=[];UC=[]
                    for k,ss in enumerate(shift):
                        adjust=dd[k] if key=='legacy_calibrated' else 0;uu,ll=mixture_probability(ref+ss+adjust,mm,ww,sg);cc,cl=mixture_probability(ref+ss+adjust+center,mm,ww,sg);LP.append(ll);LC.append(cl);U.append(uu);UC.append(cc)
                    lp=logsumexp(LP,axis=0)-np.log(256);lc=logsumexp(LC,axis=0)-np.log(256);u=np.mean(U,axis=0);uc=np.mean(UC,axis=0)
                values=dict(N=len(ii),median_mu_residual=center,reference_convolved_logscore=lp.mean(),centered_reference_convolved_logscore=lc.mean(),reference_convolved_C68=((u>=.16)&(u<=.84)).mean(),centered_reference_convolved_C68=((uc>=.16)&(uc<=.84)).mean(),shared_center_lo=np.quantile(centers,.025),shared_center_hi=np.quantile(centers,.975))
                original=next(r for r in old if r['system']==s and r['phase']==phase and r['method']==method);diff=max(abs(float(original[k])-float(v)) for k,v in values.items());assert diff<2e-8,(s,phase,method,diff)
                rows.append(dict(system=s,phase=phase,method=method,max_replay_error=diff,**values))
        print('Legacy scores replayed',s,flush=True)
    csv_rows(d/'legacy_scores_replay.csv',rows)


def numerical_one(task):
    s,m=task;t=load(OUT/'prepare/roster.npz');grid=load(OUT/'curves/grid.npz')['mu'];q=json.loads((OUT/'predict/precision.json').read_text())['chosen'];xf=np.linspace(grid[0],grid[-1],2*len(grid)-1);rows=[]
    rs=np.std(5*np.log10(t['scale_'+s]),ddof=1);z=load(OUT/'curves'/f'{s}_{m}.npz');ids=np.unique(np.linspace(0,len(z['ii'])-1,3).astype(int))
    p=load(OUT/'predict'/('QRF_north.npz' if s=='M13' else 'QRF_south.npz')) if m=='QRF' else load(OUT/'predict'/f'{s}_{m}_{q}.npz')
    for j in ids:
        i=z['ii'][j];at=np.flatnonzero(p['ii']==i)[0]
        if m=='QRF':cf,lp=uniform_curve(xf,p['edges'][at],p['mass'][at],rs)
        else:
            mm=p['means'][at];lw=p['logw'][at];sig=np.hypot(p['sigma'][at],rs);cf=np.empty(len(xf));lp=np.empty(len(xf))
            for a in range(0,len(xf),512):cf[a:a+512],lp[a:a+512]=evaluate_mix(xf[a:a+512],mm,lw,sig)
        cr=curve_crps(xf,cf);ref=10+5*np.log10(t['ref'][i]);queries=np.linspace(ref-1.5,ref+1.5,61);fine_score=row_interp(xf,queries[None],lp[None],cubic=True);coarse_score=row_interp(grid,queries[None],z['logpdf'][j:j+1],cubic=True);error=float(abs(fine_score-coarse_score).max());crerror=float(abs(cr[::2]-z['crps'][j]).max());cdferror=float(abs(cf[::2]-z['cdf'][j]).max())
        assert error<.001 and crerror<1e-4 and cdferror<1e-8,(s,m,i,error,crerror,cdferror)
        rows.append(dict(system=s,method=m,ls_id=int(t['ls_id'][i]),max_logscore_interpolation_error=error,max_CRPS_grid_error=crerror,max_CDF_error=cdferror))
    print('Half-grid checked',s,m,flush=True);return rows


def numerical():
    from concurrent.futures import ProcessPoolExecutor
    d=folder('validation')
    with ProcessPoolExecutor(max_workers=C['workers']) as pool:result=list(pool.map(numerical_one,[(s,m) for s in C['systems'] for m in METHODS]))
    rows=[row for rr in result for row in rr]
    csv_rows(d/'grid_precision.csv',rows)
    for s in C['systems']:
        for n in (4,8,16,32):
            a=load(OUT/'prepare'/f'{s}_C_{n}.npz');b=load(OUT/'prepare'/f'{s}_D_{n}.npz');np.testing.assert_array_equal(a['nodes'],b['nodes']);assert abs(a['prior']-b['prior']).max()>0
    from v3_10_figures import regression,STYLE
    reg=d/'style_regression';reg.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):result=regression(reg)
    save_json(d/'style_regression.json',dict(original_v38_pixel_identical=result,style_lock=check_templates()))
    complete(d,dict(config=CH,prepare=sha(OUT/'prepare/complete.json'),predict=sha(OUT/'predict/complete.json'),curves=sha(OUT/'curves/complete.json')),legacy_replayed=True,half_grid_pass=True,C_D_identical_nodes=True,full_parent_status_rows=933*6)


if __name__=='__main__':
    require_compute();legacy();numerical()
