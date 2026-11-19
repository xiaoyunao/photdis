#!/usr/bin/env python3
"""Fixed-rule complete-PDF examples and native geometry diagnostics."""
from seq7_common import *
from seq7_predict import load
from seq7_math import evaluate_mix
from photdis.stage1 import require_compute
from sky6_width import uniform_convolution
from scipy.special import ndtr
from scipy.signal import find_peaks


def main():
    require_compute();d=folder('examples');t=load(OUT/'prepare/roster.npz');q=json.loads((OUT/'predict/precision.json').read_text())['chosen'];x=load(OUT/'curves/grid.npz')['mu'];inputs={'evaluate':sha(OUT/'evaluate/complete.json')};selected=[];scores=[];geometry=[];arrays={}
    for s in C['systems']:
        rows=np.flatnonzero(t['system']==s);models={m:load(OUT/'curves'/f'{s}_{m}.npz') for m in METHODS};ref=10+5*np.log10(t['ref'][rows][0]);rs=np.std(5*np.log10(t['scale_'+s]),ddof=1)
        if s=='M13':
            z=load(OUT/'predict/QRF_north.npz');ii=np.intersect1d(rows[(t['phase'][rows]=='lower_MS')&t['v5'][rows]],z['ii']);j=np.searchsorted(z['ii'],ii);center=np.median(z['q'][j,2]-ref)
            _,raw=uniform_convolution(z['edges'][j],z['mass'][j],np.full(len(j),ref),rs);_,cen=uniform_convolution(z['edges'][j],z['mass'][j],np.full(len(j),ref+center),rs);delta=cen-raw;order=np.lexsort((t['ls_id'][ii],delta))
            for k in order[:3]:selected.append(dict(system=s,index=int(ii[k]),kind='QRF_decline',rule='three most negative exact centered-minus-raw marginal scores on frozen v5 lower-MS'))
            k=np.lexsort((t['ls_id'][ii],abs(delta-np.median(delta))))[0];selected.append(dict(system=s,index=int(ii[k]),kind='QRF_ordinary',rule='closest to median exact centered-minus-raw marginal score on frozen v5 lower-MS'))
            for k,i in enumerate(ii):scores.append(dict(ls_id=int(t['ls_id'][i]),raw_logscore=float(raw[k]),centered_logscore=float(cen[k]),delta=float(delta[k]),group_center=float(center)))
            save_json(d/'QRF_centering_explanation.json',dict(N=len(ii),center=float(center),raw_mean=float(raw.mean()),centered_mean=float(cen.mean()),mean_delta=float(delta.mean()),fraction_improved=float(np.mean(delta>0)),three_worst_mean_contribution=float(delta[order[:3]].sum()/len(ii)),median_delta=float(np.median(delta))))
        a=models['A'];supported=a['ii'][a['support']]
        for phase in (('lower_MS','turnoff') if s=='M5' else ('turnoff',)):
            ii=np.intersect1d(rows[(t['phase'][rows]==phase)&t['v5'][rows]],supported);j=np.searchsorted(a['ii'],ii);res=a['q'][j,2]-ref;k=np.lexsort((t['ls_id'][ii],abs(res-np.median(res))))[0]
            selected.append(dict(system=s,index=int(ii[k]),kind='sequence_'+phase,rule='nearest phase median A modulus residual on frozen v5 A-supported members'))
        e=models['E'];common=np.intersect1d(models['D']['ii'][models['D']['support']],e['ii'][e['support']]);jj=np.searchsorted(e['ii'],common);k=np.lexsort((t['ls_id'][common],-e['post'][jj]))[0]
        selected.append(dict(system=s,index=int(common[k]),kind='branch',rule='largest absolute D-to-E post-MS posterior-mass change on common support, native-ID tie break'))
        # Reference luminosity chooses diagnostic intersections only in the
        # pre-existing v6 geometry table. Prediction-color residuals here use
        # photometry-only posterior means and retain every admitted member.
        for m in METHODS[:-1]:
            z=load(OUT/'predict'/f'{s}_{m}_{q}.npz');node=load(OUT/'prepare'/f'{s}_{m}_{q}.npz')['nodes'];w=np.exp(z['logw']);color=np.column_stack((node[:,0]-node[:,1],node[:,1]-node[:,2]));pred=w@color;obs=np.column_stack((t['m'][rows,0]-t['m'][rows,1],t['m'][rows,1]-t['m'][rows,2]));res=obs-pred
            for phase in ('all','lower_MS','turnoff','other'):
                mask=z['support']&((t['phase'][rows]==phase) if phase!='all' else True)
                if not mask.any():continue
                geometry.append(dict(system=s,method=m,phase=phase,N=int(mask.sum()),gr_residual=float(np.median(res[mask,0])),gr_NMAD=float(1.4826*np.median(abs(res[mask,0]-np.median(res[mask,0])))),rz_residual=float(np.median(res[mask,1])),rz_NMAD=float(1.4826*np.median(abs(res[mask,1]-np.median(res[mask,1])))),post_mass_mean=float(z['post'][mask].mean())))
            arrays[s+'_'+m+'_post']=z['post'];arrays[s+'_'+m+'_support']=z['support']
        arrays[s+'_gr']=t['m'][rows,0]-t['m'][rows,1];arrays[s+'_rz']=t['m'][rows,1]-t['m'][rows,2];arrays[s+'_Mg']=t['m'][rows,0]-ref
        arrays[s+'_phase']=t['phase'][rows];arrays[s+'_ls_id']=t['ls_id'][rows]
    cases=[];curves={}
    for k,row in enumerate(selected):
        i=row['index'];s=row['system'];ref=10+5*np.log10(t['ref'][i]);rs=np.std(5*np.log10(t['scale_'+s]),ddof=1);case=dict(**row,case=k,ls_id=int(t['ls_id'][i]),source_id=int(t['source_id'][i]),phase=t['phase'][i],ref_mu=float(ref),ref_sigma=float(rs),TYPE=t['TYPE'][i],native=int(t['native'][i]));methods={}
        for m in METHODS:
            z=load(OUT/'curves'/f'{s}_{m}.npz');jj=np.flatnonzero(z['ii']==i)
            if not len(jj):continue
            j=int(jj[0]);jqs=z['q'][j];data=dict(q=jqs.tolist(),supported=bool(z['support'][j]),post=float(z['post'][j]),raw_logscore=float(np.interp(ref,x,z['logpdf'][j])),CDF_reference=float(np.interp(ref,x,z['cdf'][j])))
            if m=='QRF':
                a=load(OUT/'predict'/('QRF_north.npz' if s=='M13' else 'QRF_south.npz'));at=int(np.flatnonzero(a['ii']==i)[0]);edges=a['edges'][at];mass=a['mass'][at];jbin=np.searchsorted(edges,x,side='right')-1;valid=(jbin>=0)&(jbin<len(mass));density=np.zeros(len(x));density[valid]=(mass/np.diff(edges))[jbin[valid]]
                ii=np.flatnonzero((t['system']==s)&t['v5']&(t['phase']==t['phase'][i]));ii=np.intersect1d(ii,a['ii']);center=np.median(a['q'][np.searchsorted(a['ii'],ii),2]-ref);curves[f'{k}_QRF_center']=np.array([center]);curves[f'{k}_QRF_centered_density']=np.interp(x+center,x,density,left=0,right=0);curves[f'{k}_QRF_centered_CDF']=np.interp(x+center,x,z['raw'][j],left=0,right=1)
                data.update(edges=edges.tolist(),mass=mass.tolist(),diagnostic_center=float(center))
            else:
                a=load(OUT/'predict'/f'{s}_{m}_{q}.npz');at=int(np.flatnonzero(a['ii']==i)[0]);_,lp=evaluate_mix(x,a['means'][at],a['logw'][at],a['sigma'][at]);density=np.exp(lp)
                data.update(means=a['means'][at].tolist(),logweights=a['logw'][at].tolist(),sigma=float(a['sigma'][at]))
            modes=find_peaks(density)[0];modes=modes[np.argsort(density[modes])[::-1]][:5];data.update(top_modes_mu=x[modes].tolist(),top_modes_density=density[modes].tolist());methods[m]=data;curves[f'{k}_{m}_density']=density;curves[f'{k}_{m}_CDF']=z['raw'][j]
        case['methods']=methods;cases.append(case)
    save_json(d/'cases.json',cases);csv_rows(d/'selected.csv',[{k:v for k,v in r.items() if k!='methods'} for r in cases]);csv_rows(d/'QRF_M13_lowerMS_per_star.csv',scores);csv_rows(d/'prediction_color_residuals.csv',geometry);npz(d/'example_curves.npz',mu=x,**curves);npz(d/'geometry.npz',**arrays)
    # Public curves include every component parameter of these few examples.
    complete(d,inputs,selection='registered rules, no visual cherry-picking',full_example_distributions=True);print('Examples',len(cases),flush=True)


if __name__=='__main__':main()
