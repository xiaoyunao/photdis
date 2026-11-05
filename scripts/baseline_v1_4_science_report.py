#!/usr/bin/env python3
"""Render scientific figures on allocated compute from frozen diagnostic tables."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize,LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator,NullFormatter,FuncFormatter
from scipy.stats import binned_statistic_2d
from photdis.stage1 import require_compute
from photdis.diagnostics_v1_4 import CLASS_NAMES,residual_stats,distance_kpc,cell_area
from baseline_v1_4_experiments import save_json
from baseline_v1_4_assemble import load
from baseline_v1_4_extract import sha
BASE=ROOT/'data/baseline_v1_4/diagnostics'
FIG=ROOT/'figures/baseline_v1_4_science'
COLORS={-1:'#777777',0:'#2475a8',1:'#ce4b37',2:'#a07812'}
DISPLAY={-1:'Unknown',0:'Dwarfs',1:'Evolved',2:'Other / transition'}
KINDS=['Gaia','specdis','spdist']
LABELS={'Gaia':'Gaia high-S/N anchors','specdis':'SpecDis point estimates','spdist':'SPDist point estimates'}
PLOTS={};BINS={};EXTRA={}
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'figure.facecolor':'white','savefig.facecolor':'white'})


def save(fig,name,caption):
    path=FIG/(name+'.png');fig.savefig(path,dpi=150);plt.close(fig)
    PLOTS[name]=dict(path=str(path.relative_to(ROOT)),caption=caption,sha256=sha(path))
    print('figure',name,flush=True)


def summary_text(s):
    if not s['n']:return 'N=0: unvalidated'
    return (f"N={s['n']:,}; bias={s['bias_mag']:+.2f}; NMAD={s['nmad_mag']:.2f} mag\n"
            f"|error|>1: {s['tail_1']:.1%}; C68/C95={s['coverage68']:.1%}/{s['coverage95']:.1%}")


def bins1(x,y,edges,min_n=10):
    out=[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        m=np.isfinite(x)&np.isfinite(y)&(x>=lo)&((x<hi) if j<len(edges)-2 else (x<=hi))
        v=y[m];r=dict(lo=float(lo),hi=float(hi),n=int(m.sum()))
        if len(v)>=min_n:r.update(q16=float(np.quantile(v,.16)),median=float(np.median(v)),q84=float(np.quantile(v,.84)),tail=float(np.mean(abs(v)>1)))
        out.append(r)
    return out


def trend(ax,x,y,edges,color='#222222',label=None,key=None):
    b=bins1(x,y,edges);valid=[v for v in b if 'median' in v]
    if valid:
        center=np.array([(v['lo']+v['hi'])/2 for v in valid]);med=[v['median'] for v in valid]
        ax.plot(center,med,'o-',color=color,label=label,ms=3,lw=1.4)
        ax.fill_between(center,[v['q16'] for v in valid],[v['q84'] for v in valid],color=color,alpha=.15)
    if key:BINS[key]=b
    return b


def full_limits(v,pad=.04):
    v=np.asarray(v);v=v[np.isfinite(v)]
    if not len(v):return (0,1)
    lo,hi=v.min(),v.max();delta=max(hi-lo,.1)*pad
    return lo-delta,hi+delta


def distance_figures(name,t):
    for cls in [0,1,2,-1]:
        fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
        for col,kind in enumerate(KINDS):
            use=(t['class_label']==cls)&np.isfinite(t['ref_'+kind]);q=t['mu'][use];ref=t['ref_'+kind][use]
            s=residual_stats(q,ref);ax,ar=axes[:,col]
            if s['n']:
                x=distance_kpc(ref);y=distance_kpc(q[:,2]);res=q[:,2]-ref
                lo,hi=full_limits(np.log10(np.r_[x,y]));lim=(10**lo,10**hi)
                h=ax.hexbin(x,y,xscale='log',yscale='log',gridsize=48,mincnt=1,bins='log',cmap='viridis')
                fig.colorbar(h,ax=ax,label='Stars / hexagon',shrink=.7)
                ax.plot(lim,lim,'k--',lw=1);ax.set(xlim=lim,ylim=lim);ax.set_aspect('equal',adjustable='box')
                ar.scatter(x,res,s=3,alpha=.12,color=COLORS[cls],rasterized=True)
                ar.set_xscale('log');ar.set_ylim(full_limits(np.r_[res,0]))
                # bins in log distance, shown on a log axis; no extreme clipping.
                logx=np.log10(x);edges=np.linspace(*full_limits(logx,.001),13)
                bs=bins1(logx,res,edges);BINS[f'{name}_{cls}_{kind}_distance']=bs
                v=[r for r in bs if 'median' in r]
                if v:
                    center=10**np.array([(r['lo']+r['hi'])/2 for r in v])
                    ar.plot(center,[r['median'] for r in v],color='k',lw=1.5)
                    ar.fill_between(center,[r['q16'] for r in v],[r['q84'] for r in v],color='k',alpha=.15)
                # Fixed identity-ordered subset shows reference uncertainty, not
                # residual scatter or uncertainty of the median trend.
                pick=np.argsort(t['identity'][use])[:min(16,s['n'])]
                err=t['error_'+kind][use][pick]
                xe=np.vstack([x[pick]-distance_kpc(ref[pick]-err),distance_kpc(ref[pick]+err)-x[pick]])
                ax.errorbar(x[pick],y[pick],xerr=xe,fmt='none',ecolor='#c44935',alpha=.5,lw=.6)
            else:
                ax.text(.5,.5,'No eligible references',ha='center',transform=ax.transAxes)
                ar.text(.5,.5,'Unvalidated',ha='center',transform=ar.transAxes)
            if s['n']:
                for axis in [ax.xaxis,ax.yaxis,ar.xaxis]:
                    low,high=axis.get_view_interval()
                    subs=(1,) if np.log10(high/low)>2 else (1,2,5)
                    axis.set_major_locator(LogLocator(base=10,subs=subs,numticks=7))
                    axis.set_major_formatter(FuncFormatter(lambda value,pos: f'{value:g}'))
                    axis.set_minor_formatter(NullFormatter())
            ax.set(title=LABELS[kind],xlabel='Reference distance (kpc)',ylabel='Predicted median distance (kpc)')
            ar.axhline(0,c='k',ls='--',lw=.8);ar.axhline(-1,c='gray',ls=':',lw=.6);ar.axhline(1,c='gray',ls=':',lw=.6)
            ar.set(xlabel='Reference distance (kpc)',ylabel=r'$\Delta\mu=\mu_{phot,50}-\mu_{ref}$ (mag)')
            ar.text(.02,.98,summary_text(s),transform=ar.transAxes,va='top',fontsize=9,bbox=dict(fc='white',alpha=.9,ec='none'))
        fig.suptitle(f'{name} — {DISPLAY[cls]} — existing new-sky test\nFull tails retained; black trend = median and 16–84% star spread (not confidence limits)',fontsize=14)
        save(fig,f'distance_{name}_{CLASS_NAMES[cls]}','Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.')


def cmd_figure(name,t):
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    use=np.isfinite(t['ref_chosen']);gr=t['g0']-t['r0'];mr=t['g0']-t['ref_chosen'];mp=t['g0']-t['mu'][:,2]
    xl=full_limits(gr[use]);yl=full_limits(np.r_[mr[use],mp[use]])
    for col,cls in enumerate([0,1,2,-1]):
        q=use&(t['class_label']==cls)
        artists=[]
        cmap=LinearSegmentedColormap.from_list('visible_class',plt.get_cmap('Blues' if cls==0 else 'Reds' if cls==1 else 'Greys')(np.linspace(.25,1,256)))
        for row,y in enumerate([mr,mp]):
            ax=axes[row,col]
            if q.any():artists.append(ax.hexbin(gr[q],y[q],gridsize=45,mincnt=1,cmap=cmap))
            ax.set(xlim=xl,ylim=yl[::-1],xlabel=r'$(g-r)_0$',ylabel='Legacy $M_g$ (mag)',title=f'{DISPLAY[cls]}: N={int(q.sum()):,} — '+('reference' if row==0 else 'prediction'))
        if artists:
            norm=LogNorm(1,max(2,max(h.get_array().max() for h in artists)))
            for ax,h in zip(axes[:,col],artists):
                h.set_norm(norm);fig.colorbar(h,ax=ax,label='Stars / hexagon',shrink=.6)
        ids=np.flatnonzero(q);ids=ids[np.argsort(t['identity'][ids])][:4]
        for i in ids:
            axes[0,col].annotate('',xy=(gr[i],mp[i]),xytext=(gr[i],mr[i]),arrowprops=dict(arrowstyle='->',color=COLORS[cls],lw=1))
            axes[0,col].scatter([gr[i]],[mp[i]],s=20,facecolors='none',edgecolors=COLORS[cls])
    fig.suptitle(name+' — same-star reference and predicted color–absolute-magnitude diagrams',fontsize=14)
    save(fig,'cmd_'+name,'Same Legacy g definition and same objects in each column; chosen Gaia-then-SpecDis reference. Four lowest-ID objects per class have within-panel arrows from reference to prediction in the top row; open endpoints mark predicted positions. Color scales match vertically within each class. A plausible predicted sequence does not establish correct distances.')


def grid_values(x,y,delta,xe,ye):
    good=np.isfinite(x)&np.isfinite(y)&np.isfinite(delta)
    count=np.histogram2d(x[good],y[good],bins=[xe,ye])[0]
    if good.any():
        med=binned_statistic_2d(x[good],y[good],delta[good],'median',bins=[xe,ye]).statistic
        tail=binned_statistic_2d(x[good],y[good],(abs(delta[good])>1).astype(float),'mean',bins=[xe,ye]).statistic
    else:med=tail=np.full_like(count,np.nan)
    med=np.where(count>=10,med,np.nan);tail=np.where(count>=10,tail,np.nan)
    return count,med,tail,int(good.sum()-count.sum())


def color_figures(t):
    gr=t['g0']-t['r0'];delta=t['mu'][:,2]-t['ref_chosen']
    planes=[('gr_rz',gr,t['r0']-t['z0'],(-1,3),(-1,3),'$(g-r)_0$','$(r-z)_0$',np.ones(len(gr),bool)),
            ('gr_zW1',gr,t['z0']-t['w10'],(-1,3),(-2,6),'$(g-r)_0$','$(z-W1)_0$',t['B']),
            ('ri_iz',t['r0']-t['i0'],t['i0']-t['z0'],(-1,2),(-1,2),'$(r-i)_0$','$(i-z)_0$',t['i_usable'])]
    for name,x,y,xlim,ylim,xlab,ylab,eligible in planes:
        xe=np.linspace(*xlim,25);ye=np.linspace(*ylim,25)
        fig,axes=plt.subplots(3,3,figsize=(14,12),layout='constrained')
        for row,cls in enumerate([None,0,1]):
            q=eligible&np.isfinite(delta)
            if cls is not None:q &= t['class_label']==cls
            count,med,tail,outside=grid_values(x[q],y[q],delta[q],xe,ye)
            BINS[name+'_'+str(cls)]=dict(xedges=xe,yedges=ye,count=count,median=med,tail=tail,outside=outside,total=int(q.sum()))
            for col,(v,title) in enumerate([(count,'Reference stars'),(med,'Median residual (mag)'),(tail,'Severe-error fraction')]):
                ax=axes[row,col];ax.set_facecolor('#dedede')
                norm=LogNorm(1,max(2,count.max())) if col==0 else Normalize(-7,7) if col==1 else Normalize(0,1)
                value=np.ma.masked_less(v,1) if col==0 else np.ma.masked_invalid(v)
                h=ax.pcolormesh(xe,ye,value.T,norm=norm,cmap='viridis' if col==0 else 'RdBu_r' if col==1 else 'magma',shading='flat')
                fig.colorbar(h,ax=ax,label=title,shrink=.8,extend='both' if col==1 else 'neither')
                ax.set(xlabel=xlab,ylabel=ylab,title=f'{"All classes" if cls is None else DISPLAY[cls]}: N={int(q.sum()):,}; outside={outside}')
                if row==0:
                    for cl in [0,1]:
                        mask=q&(t['class_label']==cl)
                        c=np.histogram2d(x[mask],y[mask],bins=[xe,ye])[0]
                        if c.max()>3:
                            levels=np.unique(np.quantile(c[c>0],[.6,.85]))
                            if len(levels):ax.contour((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,c.T,levels=levels,colors=[COLORS[cl]],linewidths=.8)
        fig.suptitle('matched_griz — '+name+' — star counts and class-resolved errors\nBias/tail cells require N≥10; blue/red contours: dwarf/evolved count levels',fontsize=14)
        save(fig,'colors_'+name,'Fixed color limits; excluded counts disclosed per row and saved. All residuals contribute inside color range, including catastrophic tails. Empty/low-count performance cells are blank. WISE plane requires usable W1; i plane uses matched i-usable stars. Color overlap alone cannot prove absence of useful information.')


def sky_panel(ax,values,key,label,fig):
    records=[r for r in values if r['n']>0 and (key=='n' or r['n']>=20)]
    if key=='n':norm=LogNorm(1,max([2]+[r['n'] for r in records]));cmap=plt.get_cmap('viridis')
    elif key=='bias_mag':norm=Normalize(-7,7);cmap=plt.get_cmap('RdBu_r')
    elif key=='nmad_mag':norm=Normalize(0,3);cmap=plt.get_cmap('viridis')
    else:norm=Normalize(0,1);cmap=plt.get_cmap('magma')
    for r in records:
        ra=(r['cell']//18)*10;dec=(r['cell']%18)*10-90
        ax.add_patch(Rectangle((ra,dec),10,10,fc=cmap(norm(r[key])),ec='white',lw=.2))
    ax.set(xlim=(360,0),ylim=(-90,90),xlabel='RA (deg)',ylabel='Dec (deg)',title=label+f' ({len(records)} cells)')
    ax.set_facecolor('#eeeeee');ax.grid(alpha=.15)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.65,extend='both' if key=='bias_mag' else 'neither')


def sky_figures(stats):
    s=stats['models']['expanded_A']['sky']
    settings=[('class',[('chosen',k,k) for k in ['all','dwarf','evolved']]),
              ('source',[(k,'all',LABELS[k]) for k in KINDS]),
              ('brightness',[('chosen',k,k) for k in ['r_lt18','r18_20','r_ge20']])]
    for name,rows in settings:
        fig,axes=plt.subplots(3,4,figsize=(19,11),layout='constrained')
        for i,(source,layer,label) in enumerate(rows):
            for j,(key,title) in enumerate([('n','Raw reference N'),('bias_mag','Median residual (mag)'),('nmad_mag','NMAD (mag)'),('tail_1','Severe-error fraction')]):
                sky_panel(axes[i,j],s[source][layer],key,label+' — '+title,fig)
        fig.suptitle('expanded_A — original 10° sky cells; gray = unvalidated/sparse\nRaw counts are not area-normalized density; no interpolation or sky correction',fontsize=14)
        save(fig,'sky_'+name,'Existing new-sky test only; performance needs20 references per cell. Reference counts shown without a minimum; raw count is not sky density. Stratification exposes composition differences; residual patterns alone do not establish a spatial calibration error.')
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    foot=stats['footprint']
    for ax,role in zip(axes.flat,range(4)):
        values=[dict(cell=r['cell'],n=r['split_counts'][str(role)]) for r in foot]
        sky_panel(ax,values,'n',['Train','Calibration','Test','Quarantined / historical / conflicts'][role],fig)
    for ax,kind in zip(axes.flat[4:],['clean','historical']):
        vals=[dict(cell=r['cell'],n=r[kind]) for r in foot]
        sky_panel(ax,vals,'n',kind+' raw reference N',fig)
    fig.suptitle('Frozen reference footprint — all assembled candidates before per-model scientific eligibility',fontsize=14)
    save(fig,'footprint','Train/cal/test geometry and quarantined identities; cells may contain quarantined historical identities alongside a nonhistorical role. Counts concern assembled references, not the full optical sky. No unobserved cell is filled.')
    # Explicit solid-angle-normalized density, separate from count maps.
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    vals=s['chosen']['all'];dens=[r['n']/r['area_sqdeg'] for r in vals if r['n']]
    norm=LogNorm(min(dens),max(dens))
    for r in vals:
        if r['n']:
            ax.add_patch(Rectangle(((r['cell']//18)*10,(r['cell']%18)*10-90),10,10,fc=plt.cm.viridis(norm(r['n']/r['area_sqdeg'])),ec='white',lw=.2))
    ax.set(xlim=(360,0),ylim=(-90,90),xlabel='RA (deg)',ylabel='Dec (deg)',title='expanded_A chosen test references per square degree (10° cell solid angle)');ax.set_facecolor('#eeeeee')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,label='References / deg²')
    save(fig,'sky_density','N divided by exact spherical cell area; this is reference selection density, not stellar population density.')


def physical_figures(t):
    delta=t['mu'][:,2]-t['ref_chosen'];good=t['rvs_parameter_quality']&np.isfinite(delta)
    xe=np.linspace(3000,10000,29);ye=np.linspace(-1,6,29)
    fig,axes=plt.subplots(3,3,figsize=(14,12),layout='constrained')
    for row,cls in enumerate([0,1,2]):
        q=good&(t['class_label']==cls)
        count,med,tail,out=grid_values(t['rvs_teff'][q],t['rvs_logg'][q],delta[q],xe,ye)
        BINS['kiel_'+str(cls)]=dict(xedges=xe,yedges=ye,count=count,median=med,tail=tail,outside=out)
        for col,(v,title,norm,cmap) in enumerate([(count,'Reference N',LogNorm(1,max(2,count.max())),'viridis'),(med,'Median residual (mag)',Normalize(-7,7),'RdBu_r'),(tail,'Severe-error fraction',Normalize(0,1),'magma')]):
            ax=axes[row,col];ax.set_facecolor('#dedede');v=np.ma.masked_less(v,1) if col==0 else np.ma.masked_invalid(v)
            h=ax.pcolormesh(xe,ye,v.T,norm=norm,cmap=cmap);fig.colorbar(h,ax=ax,shrink=.75,label=title)
            ax.set(xlim=(10000,3000),ylim=(6,-1),xlabel='RVS Teff (K)',ylabel='RVS log g',title=f'{DISPLAY[cls]} N={int(q.sum()):,}, outside={out}')
    fig.suptitle('matched_griz — reliable spectral parameters: Kiel diagrams\nCounts and class-resolved errors; performance cells require N≥10',fontsize=14)
    save(fig,'kiel','Published RVS parameters with original parameter-quality qualification. Fixed displayed Teff/logg limits and excluded counts. Other/transition is not assigned a definite evolutionary phase.')
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    fe=np.array([-4,-2,-1.5,-1,-.5,0,.5,1.])
    for col,cls in enumerate([0,1,2]):
        q=good&(t['class_label']==cls)&np.isfinite(t['rvs_feh'])&(t['rvs_feh_err']>0)
        ax=axes[0,col];ax.scatter(t['rvs_feh'][q],delta[q],s=3,c=COLORS[cls],alpha=.15,rasterized=True)
        b=trend(ax,t['rvs_feh'][q],delta[q],fe,color=COLORS[cls],key='FeH_'+str(cls))
        ax.axhline(0,c='k',ls='--',lw=.8);ax.set(xlabel='RVS [Fe/H]',ylabel='Residual (mag)',title=f'{DISPLAY[cls]} N={int(q.sum()):,}')
        ax.set_ylim(full_limits(np.r_[delta[q],0]));ax.set_xlim(full_limits(t['rvs_feh'][q]))
        low=q&(t['rvs_feh']<-1.5);EXTRA['metal_poor_'+str(cls)]=residual_stats(t['mu'][low],t['ref_chosen'][low])
        ar=axes[1,col];x=np.arange(len(b));ar.bar(x,[r.get('tail',np.nan) for r in b],color=COLORS[cls]);ar.set_ylim(0,1)
        ar.set_xticks(x,[f"{r['lo']:g}–{r['hi']:g}\nN={r['n']}" for r in b],rotation=35,ha='right');ar.set(ylabel='Severe-error fraction',xlabel='[Fe/H] bin (N<10 blank)')
    fig.suptitle('matched_griz — metallicity dependence; [Fe/H]<−1.5 is a metal-poor subset, not a halo label',fontsize=14)
    save(fig,'metallicity','Continuous spectral trends use reliable RVS parameters, finite metallicity and positive formal metallicity error. Full scatter extents retained; fixed-bin medians/16–84% spread show occupied ranges, not causal metallicity corrections.')
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    wise=np.where(t['C'],2,np.where(t['B'],1,0))
    variables=[('r magnitude',t['r'],np.array([14,16,18,20,22,25.])),('Max grz magnitude error',t['optical_error'],np.array([0,.01,.03,.1,.3,1.])),('WISE state',wise,np.array([-.5,.5,1.5,2.5]))]
    for col,(name,x,edges) in enumerate(variables):
        for cls in [-1,0,1,2]:
            q=np.isfinite(delta)&(t['class_label']==cls)
            b=trend(axes[0,col],x[q],delta[q],edges,COLORS[cls],DISPLAY[cls],key=name+'_'+str(cls))
            v=[r for r in b if 'tail' in r];center=[(r['lo']+r['hi'])/2 for r in v]
            axes[1,col].plot(center,[r['tail'] for r in v],'o-',color=COLORS[cls],label=DISPLAY[cls]+': '+','.join(str(r['n']) for r in b),ms=3)
        for ax in axes[:,col]:ax.set_xlabel(name)
        axes[0,col].axhline(0,c='k',ls='--',lw=.7);axes[0,col].set_ylabel('Median residual; 16–84% spread (mag)')
        axes[1,col].set(ylabel='Severe-error fraction',ylim=(0,1));axes[1,col].legend(title='Class: N in each bin',fontsize=7)
        if col==2:
            for ax in axes[:,col]:ax.set_xticks([0,1,2],['No usable W1','W1 only','W1+W2'])
    axes[0,0].legend();fig.suptitle('matched_griz — brightness, reported optical errors and WISE state',fontsize=14)
    save(fig,'observing_conditions','Fixed bins with N≥10; class labels retained separately. Bottom legends give actual reference N in every bin. Trends mix selection and measurement effects; sparse20–25 bins do not establish faint-star accuracy.')


def probability_figures(t):
    good=np.isfinite(t['ref_chosen']);re=np.array([14,18,20,22,25.]);classes=[0,1,2,-1]
    stats=[[residual_stats(t['mu'][good&(t['class_label']==c)&(t['r']>=lo)&(t['r']<hi)],t['ref_chosen'][good&(t['class_label']==c)&(t['r']>=lo)&(t['r']<hi)]) for lo,hi in zip(re[:-1],re[1:])] for c in classes]
    EXTRA['class_brightness_probability']=stats
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,key,label in zip(axes.flat,['coverage68','coverage95','width68','width95'],['68% reference coverage','95% reference coverage','Median 68% width (mag)','Median 95% width (mag)']):
        v=np.array([[s.get(key,np.nan) if s['n']>=10 else np.nan for s in row] for row in stats]);h=ax.imshow(v,aspect='auto',vmin=0,vmax=1 if 'coverage' in key else max(1,np.nanmax(v)),cmap='viridis');fig.colorbar(h,ax=ax)
        for i,row in enumerate(stats):
            for j,s in enumerate(row):
                ax.text(j,i,(f"{s.get(key,np.nan):.2f}\n" if s['n']>=10 else 'Sparse\n')+f"N={s['n']}",ha='center',va='center',fontsize=9,color='black',bbox=dict(fc='white',alpha=.7,ec='none'))
        ax.set_xticks(range(4),['14–18','18–20','20–22','22–25']);ax.set_yticks(range(4),[DISPLAY[c] for c in classes]);ax.set(xlabel='r magnitude',title=label)
    fig.suptitle('matched_griz — interval behavior by true class and brightness\nAgreement with chosen reference point; no VAC reference-error convolution',fontsize=14)
    save(fig,'intervals','All occupied cells show N; N<10 estimates blank. Width is in distance-modulus magnitudes. Large evolved-star misses remain visible despite the overall dwarf-dominated coverage.')
    fig,axes=plt.subplots(2,3,figsize=(16,8.5),layout='constrained')
    for ax,c in zip(axes.flat[:4],classes):
        q=(t['class_label']==c)&np.isfinite(t['full_pit']);v=t['full_pit'][q]
        for lo,hi,col in [(14,18,'#2166ac'),(18,20,'#b2182b'),(20,25,'#6b8e23')]:
            w=q&(t['r']>=lo)&(t['r']<hi);h=np.histogram(t['full_pit'][w],np.linspace(0,1,11))[0]
            if w.any():ax.stairs(h/w.sum(),np.linspace(0,1,11),label=f'r {lo}–{hi}: N={w.sum()}',color=col)
        ax.axhline(.1,c='gray',ls='--');ax.set(xlabel='Full observed-parallax PIT',ylabel='Fraction per bin',title=f'{DISPLAY[c]} N={q.sum():,}; negative={(q&(t["p_corr"]<0)).sum()}');ax.legend()
        EXTRA['refined_pit_'+str(c)]=dict(n=int(q.sum()),negative=int((q&(t['p_corr']<0)).sum()),hist=np.histogram(v,np.linspace(0,1,11))[0],mean_log_density=float(t['full_log_density'][q].mean()) if q.any() else None)
    for ax,flag in zip(axes.flat[4:],['supported','ambiguous']):
        labels=[];values=[]
        for c in classes:
            for state in [False,True]:
                q=good&(t['class_label']==c)&(t[flag]==state);s=residual_stats(t['mu'][q],t['ref_chosen'][q]);EXTRA[flag+'_'+str(c)+'_'+str(state)]=s
                labels.append(f'{DISPLAY[c]}\n{int(state)}; N={s["n"]}');values.append(s.get('tail_1',np.nan))
        ax.bar(range(8),values,color=[COLORS[c] for c in classes for _ in range(2)]);ax.set_xticks(range(8),labels,rotation=60,ha='right');ax.set(ylim=(0,1),ylabel='Severe-error fraction',title=flag+' flag: false=0 / true=1')
    fig.suptitle('matched_griz — refined signed-Gaia diagnostics and predictive flags',fontsize=14)
    save(fig,'gaia_flags','Gaia PIT and log density read only from gaia_refined/matched_griz/evaluation.npz. Negative/weak parallaxes retained in observation space. Flat PIT is not proof of unbiased target-population calibration; support/ambiguity flags are assessed against actual errors.')
    p=load(BASE/'pdf_examples.npz');n=len(p['identity'])
    for page,start in enumerate(range(0,n,5),1):
        fig,axes=plt.subplots(5,1,figsize=(13,15),layout='constrained')
        for k,ax in enumerate(axes):
            row=start+k
            if row>=n:ax.axis('off');continue
            for name,color in [('matched_grz','#777777'),('matched_griz','#2475a8')]:
                edges=p[name+'_mu_edges'][row];mass=p[name+'_pdf'][row]
                ax.stairs(mass/np.diff(edges),edges,color=color,label=name.replace('matched_',''),lw=1.3)
                ax.axvline(p[name+'_quantiles'][row,2],c=color,ls=':',lw=.9)
            ax.axvline(p['ref'][row],c='#ce4b37',label='Chosen reference');ax.axvspan(p['ref'][row]-p['ref_error'][row],p['ref'][row]+p['ref_error'][row],color='#ce4b37',alpha=.13,label='Reference ±sigma_mu')
            ax.set(xlabel='Distance modulus (mag)',ylabel='PDF / mag',title=f'{p["category"][row]} | ID {p["identity"][row]} | {DISPLAY[int(p["class_label"][row])]} | r={p["r"][row]:.2f} | ref={"Gaia" if p["reference_source"][row]==1 else "SpecDis"}')
            ax.legend(ncol=4)
        fig.suptitle('Fixed diagnostic examples — complete PDFs from frozen models; identical Sobol128\nDotted lines: predicted medians. Examples explain mechanisms, not population frequencies.',fontsize=14)
        save(fig,'pdf_examples_'+str(page),'Two deterministic, nonoverlapping identity-hash examples from each registered category. Full modulus support shown; densities integrate to1. Frozen grz/griz quantiles independently reproduced within1e-6. Reference uncertainty is shown; an ambiguity flag does not itself prove a distinct astrophysical branch.')


def paired_figures(stats):
    names=['expanded_A','expanded_B','expanded_C','matched_griz']
    fig,axes=plt.subplots(4,3,figsize=(16,14),layout='constrained')
    for row,name in enumerate(names):
        old=name.replace('expanded','frozen') if name.startswith('expanded') else 'matched_grz'
        a=load(BASE/(old+'.npz'));b=load(BASE/(name+'.npz'));ref=a['ref_chosen'];d=abs(b['mu'][:,2]-ref)-abs(a['mu'][:,2]-ref)
        variables=[(a['g0']-a['r0'],np.linspace(-1,3,17),'$(g-r)_0$',np.ones(len(d),bool)),
                   (a['rvs_logg'],np.linspace(-1,6,15),'RVS log g',a['rvs_parameter_quality']),
                   (a['rvs_feh'],np.array([-4,-2,-1.5,-1,-.5,0,.5,1]),'RVS [Fe/H]',a['rvs_parameter_quality']&(a['rvs_feh_err']>0))]
        for col,(x,edges,label,valid) in enumerate(variables):
            ax=axes[row,col]
            for c in [0,1,2,-1]:
                q=valid&(a['class_label']==c)&np.isfinite(d)
                trend(ax,x[q],d[q],edges,COLORS[c],DISPLAY[c],key=f'pair_{name}_{label}_{c}')
            ax.axhline(0,c='k',ls='--',lw=.8);ax.set(xlabel=label,ylabel='Δ absolute error (mag)',title=name+' minus '+old)
            # Shared range per model and all classes retains catastrophic changes.
            ax.set_yscale('symlog',linthresh=.03);ax.set_ylim(full_limits(np.r_[d[np.isfinite(d)],0]));ax.legend()
    fig.suptitle('Same-star gains and harms — negative values improve; positive values worsen\nSymmetric-log y axis (linear within ±0.03mag); median and16–84% star spread; N≥10',fontsize=14)
    save(fig,'paired_physics','Exact same reference and identity in each pair. Symmetric-log y scale resolves small changes while retaining full extrema. Bands show dispersion of star-level changes, not bootstrap uncertainty. Validated bin counts and summaries saved separately.')
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    labels=[];v=[];lo=[];hi=[];rescued=[];harmed=[];colors=[]
    for name in names:
        for label in ['all','dwarf','evolved','other_transition','unknown']:
            s=stats['pairs'][name]['chosen'][label];labels.append(name+' / '+label+f' (N={s["n"]}, cells={s.get("cells",0)})')
            v.append(100*s.get('tail_difference',np.nan));ci=s.get('tail_ci95',[np.nan,np.nan]);lo.append(100*ci[0]);hi.append(100*ci[1]);rescued.append(s.get('rescued',0));harmed.append(s.get('harmed',0));colors.append('#ce4b37' if label=='evolved' else '#2475a8')
    y=np.arange(len(labels));axes[0].hlines(y,lo,hi,color=colors);axes[0].scatter(v,y,c=colors,s=20);axes[0].axvline(0,c='k',ls='--');axes[0].set_yticks(y,labels,fontsize=8);axes[0].invert_yaxis();axes[0].set(xlabel='New − old severe-error rate (percentage points)',title='Paired whole-sky-cell bootstrap, 95% interval')
    axes[1].barh(y-.18,rescued,height=.34,label='Wrong → right',color='#2475a8');axes[1].barh(y+.18,harmed,height=.34,label='Right → wrong',color='#ce4b37');axes[1].set_yticks(y,labels,fontsize=8);axes[1].invert_yaxis();axes[1].set(xlabel='Stars crossing |residual|=1mag boundary',title='Internal changes, not just net count');axes[1].legend()
    save(fig,'paired_uncertainty','2000 paired bootstrap resamples of entire occupied10° test cells; star-weighted rate recomputed in every draw, percentile95% interval. Negative rate difference improves. Small cell counts and reference selection limit generalization; this is exploratory comparison, not an adoption test.')


def coverage_figures():
    from baseline_v1_4_coverage import application_coverage
    fig, coverage = application_coverage()
    EXTRA['application_coverage'] = coverage
    save(fig, 'application_coverage',
         'Actual expanded A class and distance training identities, and separate cal/test eligibility. Distance evidence includes qualified point estimates and signed Gaia parallaxes. Empty-bin fractions use the original weighted DECam parent.')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    stats=json.loads((BASE/'statistics.json').read_text())
    for ax,band in zip(axes,['W1','W2']):
        j=stats['parent']['joint'][band];v=np.zeros((2,4));n=np.zeros((2,4),int)
        for r in j['cells']:
            row=int(r['low_snr']);col=2*int(r['mask_failed'])+int(r['fracflux_failed']);v[row,col]=r['fraction'];n[row,col]=r['n']
        im=ax.imshow(v,vmin=0,vmax=1,cmap='viridis',aspect='auto');fig.colorbar(im,ax=ax,label='Weighted fraction of entire old parent')
        for i in range(2):
            for k in range(4):ax.text(k,i,f'{v[i,k]:.2%}\nN={n[i,k]}',ha='center',va='center',bbox=dict(fc='white',alpha=.8,ec='none'))
        ax.set_xticks(range(4),['Both pass','FRACFLUX fail','WISEMASK fail','Both fail']);ax.set_yticks([0,1],['S/N ≥5','S/N <5']);ax.set_title(f'{band}: all low-S/N={j["low_snr_total_fraction"]:.1%}\nLow-S/N within quality failures={j["low_snr_given_quality"]:.1%}')
    fig.suptitle('Joint WISE loss conditions — low-S/N remains visible inside quality failures',fontsize=14)
    save(fig,'wise_joint','Independent axes: WISEMASK!=0, FRACFLUX>=1 (or invalid), measured S/N<5. Finite positive IVAR, observations and transmission required. Negative flux belongs to low S/N. Fractions use original weights and entire-parent denominator; unmeasured fraction retained separately in statistics.')


def main():
    require_compute();FIG.mkdir(parents=True,exist_ok=True)
    completion=json.loads((BASE/'complete.json').read_text());assert sha(BASE/'statistics.json')==completion['statistics_sha256']
    stats=json.loads((BASE/'statistics.json').read_text())
    for name in ['expanded_A','matched_griz']:
        t=load(BASE/(name+'.npz'));distance_figures(name,t);cmd_figure(name,t)
    t=load(BASE/'matched_griz.npz');color_figures(t);sky_figures(stats);physical_figures(t);probability_figures(t);paired_figures(stats);coverage_figures()
    save_json(BASE/'plot_data.json',dict(bins=BINS,extra=EXTRA,figures=PLOTS,statistics_sha256=sha(BASE/'statistics.json'),code_sha256=sha(__file__)))
    print('RENDER COMPLETE',len(PLOTS),'figures',flush=True)

if __name__=='__main__':main()
