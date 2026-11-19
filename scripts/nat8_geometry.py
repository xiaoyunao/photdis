#!/usr/bin/env python3
"""Distance-independent original-segment geometry, covariance and branches."""
from nat8_common import *
from photdis.stage1 import require_compute


def main():
    require_compute();d=folder('geometry');inputs={'prepare':sha(OUT/'prepare/complete.json')};t=load(OUT/'prepare/roster.npz');curves=json.loads((OLD/'prepare/curves.json').read_text());rows=[];minimum=[];redraw={};candidates=[]
    for s in C['systems']:
        a=load(OUT/'prepare'/f'{s}_E.npz');q=old('prepare',f'{s}_E_32.npz',inputs);ii=a['ii'];model=curves[s+'_E'];mag=np.array(model['mag']);colors=np.column_stack((mag[:,0]-mag[:,1],mag[:,1]-mag[:,2]));left=q['left'];right=q['right'];v=colors[right]-colors[left];start=colors[left];m=t['m'][ii];obs=np.column_stack((m[:,0]-m[:,1],m[:,1]-m[:,2]));e=np.maximum(t['err'][ii],.001)
        S=np.zeros((len(ii),2,2));S[:,0,0]=e[:,0]**2+e[:,1]**2;S[:,1,1]=e[:,1]**2+e[:,2]**2;S[:,0,1]=S[:,1,0]=-e[:,1]**2;W=np.linalg.inv(S);diff=obs[:,None]-start[None];vv=np.einsum('sj,ijk,sk->is',v,W,v);dv=np.einsum('sj,ijk,isk->is',v,W,diff);u=np.clip(dv/vv,0,1);res=diff-u[:,:,None]*v;chi=np.einsum('isj,ijk,isk->is',res,W,res);tan=np.einsum('sj,ijk,isk->is',v,W,res)/np.sqrt(vv);cross=v[None,:,0]*res[:,:,1]-v[None,:,1]*res[:,:,0];normal=cross/(np.sqrt(np.linalg.det(S))[:,None]*np.sqrt(vv));np.testing.assert_allclose(tan*tan+normal*normal,chi,atol=1e-9,rtol=1e-9)
        lengths=np.linalg.norm(v,axis=1);perp_mag=cross/lengths[None];tan_mag=(res*v).sum(axis=2)/lengths[None];sig_perp=np.sqrt(np.einsum('sj,ijk,sk->is',np.column_stack((-v[:,1],v[:,0]))/lengths[:,None],S,np.column_stack((-v[:,1],v[:,0]))/lengths[:,None]));best=np.argmin(chi,axis=1);at=np.arange(len(ii));phase=np.array(model['phase']);branch=phase[right]==2;weights=np.exp(a['logw']).reshape(len(ii),len(left),32).sum(axis=2)
        arrays=dict(ii=ii,TYPE=t['TYPE'][ii],ls_id=t['ls_id'][ii],chi=chi,u=u,tangent_sigma=tan,normal_sigma=normal,perp_mag=perp_mag,tangent_mag=tan_mag,sigma_perp=sig_perp,best=best,left=left,right=right,branch=branch,segment_posterior=weights)
        npz(d/f'{s}_segments.npz',**arrays)
        for k,i in enumerate(ii):
            j=best[k];minimum.append(dict(TYPE=t['TYPE'][i],ls_id=int(t['ls_id'][i]),system=s,native=int(t['native'][i]),gr=float(obs[k,0]),rz=float(obs[k,1]),E_supported=bool(a['support'][k]),segment_chi2=float(chi[k,j]),original_node_chi2=float(a['chi'][k]),normal_sigma=float(normal[k,j]),tangent_sigma=float(tan[k,j]),perp_mag=float(perp_mag[k,j]),sigma_perp=float(sig_perp[k,j]),at_endpoint=bool(u[k,j] in (0,1)),best_left=int(left[j]),best_right=int(right[j]),branch_post=bool(branch[j])))
            # Count distinct local minima, not adjacent quadrature/table points.
            good=[]
            for j in range(len(left)):
                prev=chi[k,j-1] if j and right[j-1]==left[j] else np.inf;following=chi[k,j+1] if j+1<len(left) and right[j]==left[j+1] else np.inf
                if chi[k,j]<=prev and chi[k,j]<=following and chi[k,j]<=chi[k].min()+C['geometry']['competitive_delta_chi2']:good.append(j)
            for j in good:candidates.append(dict(TYPE=t['TYPE'][i],ls_id=int(t['ls_id'][i]),system=s,left=int(left[j]),right=int(right[j]),u=float(u[k,j]),chi2=float(chi[k,j]),post=bool(branch[j]),segment_posterior=float(weights[k,j]),competing_minima=len(good)))
        for lo,hi in zip(C['gr_edges'][:-1],C['gr_edges'][1:]):
            for support,label in [(np.ones(len(ii),bool),'all'),(a['support'],'supported'),(~a['support'],'unsupported')]:
                take=support&(obs[:,0]>=lo)&(obs[:,0]<hi);val=normal[at,best][take]
                if len(val):rows.append(dict(system=s,gr_lo=lo,gr_hi=hi,sample=label,N=int(take.sum()),median_perp_sigma=float(np.median(val)),NMAD_perp_sigma=float(1.4826*np.median(abs(val-np.median(val)))),fraction_abs_gt3=float(np.mean(abs(val)>3)),endpoint_fraction=float(np.mean((u[at,best][take]==0)|(u[at,best][take]==1)))))
        redraw.update({s+'_colors':obs,s+'_model_colors':colors,s+'_left':left,s+'_right':right,s+'_support':a['support'],s+'_normal':normal[at,best],s+'_tangent':tan[at,best]});print('Geometry',s,len(ii),flush=True)
    csv_rows(d/'per_member.csv',minimum);csv_rows(d/'competitive_segments.csv',candidates);csv_rows(d/'color_bins.csv',rows);npz(d/'redraw.npz',**redraw);complete(d,inputs,scope='distance-independent weighted nearest original physical segments, frozen E support remains unchanged',tangent='zero at interior nearest projection by construction; endpoints and physical components reported, not evidence of no shape error')


if __name__=='__main__':main()
