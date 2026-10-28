#!/usr/bin/env python3
"""Audit the existing 10-degree pilot against Gaia BP/RP and PS1 standards."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".deps")]
import fitsio
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.spatial import cKDTree
from photdis.audit import (GROUP_NAMES,GROUP_COLORS,checked_gaia_color,rvs_metallicity,
                           ps1_decam,failure_groups,stats)
from photdis.gaia import join_gaia,derive_gaia_labels
from photdis.photometry import derive_photometry
from photdis.stage1 import (CORE_COLUMNS,PHOT_COLUMNS,legacy_ids,structured,write_json,
                           require_compute,unit_vectors)

PS1_ROOT=Path("/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3")
OUT=Path("data/pilot_audit_v1")
FIG=Path("figures/pilot_audit")


def canonical_reference(t):
    rows=np.flatnonzero(t["anchor_astrometry"])
    error=np.nan_to_num(t["sigma_g"]+t["sigma_r"]+t["sigma_z"],nan=1e4)
    score=(~t["grz_high_quality"]).astype(float)*1e6+(~t["grz_base"]).astype(float)*1e5+error
    order=np.lexsort((t["ls_id"][rows],score[rows],t["source_id"][rows]))
    rows=rows[order]
    return rows[np.r_[True,np.diff(t["source_id"][rows])!=0]]


def raw_reference(config,ref):
    chunks=[];provenance=[]
    wanted=np.sort(ref["ls_id"])
    for name in config["sweep_files"]:
        path=Path(config["sweep_root"])/name
        with fitsio.FITS(str(path)) as f:
            ids=legacy_ids(f[1].read(columns=["RELEASE","BRICKID","OBJID"]))
            keep=np.isin(ids,wanted)
            rows=np.flatnonzero(keep)
            chunks.append(f[1].read(columns=CORE_COLUMNS+PHOT_COLUMNS,rows=rows))
        provenance.append({"path":str(path),"rows_read":len(rows),"bytes":path.stat().st_size})
        print(json.dumps({"reference_source":name,"rows":len(rows)}),flush=True)
    raw=np.concatenate(chunks)
    ids=legacy_ids(raw); order=np.argsort(ids)
    where=np.searchsorted(ids[order],ref["ls_id"])
    assert len(raw)==len(ref) and np.array_equal(ids[order[where]],ref["ls_id"])
    return raw[order[where]],provenance


def ps1_reference(gaia):
    pixels=hp.query_disc(32,hp.ang2vec(5,-5,lonlat=True),np.deg2rad(7.3),inclusive=True,nest=False)
    columns=["OBJ_ID","RA_OK","DEC_OK","EPOCHMEAN_OK","MEDIAN","NMAG_OK","STDEV"]
    chunks=[];sources=[]
    for pixel in pixels:
        path=PS1_ROOT/f"ps1-{pixel:05d}.fits"
        if not path.exists():
            raise FileNotFoundError(path)
        table=fitsio.read(str(path),ext=1,columns=columns)
        mask=((table["RA_OK"]>=359.9)|(table["RA_OK"]<=10.1))&(table["DEC_OK"]>=-10.1)&(table["DEC_OK"]<=.1)
        mask &= np.isfinite(table["EPOCHMEAN_OK"])&(table["EPOCHMEAN_OK"]>50000)&(table["EPOCHMEAN_OK"]<60000)
        chunks.append(table[mask]);sources.append({"path":str(path),"rows_in_margin":int(mask.sum())})
    cat=np.concatenate(chunks)
    _,unique=np.unique(cat["OBJ_ID"],return_index=True)
    cat=cat[unique]
    epoch=2000+(cat["EPOCHMEAN_OK"]-51544.5)/365.25
    ref_epoch=gaia["ref_epoch"]
    def position(target_epoch):
        dt=target_epoch-ref_epoch
        return ((gaia["ra"]+gaia["pmra"]*dt/(3.6e6*np.cos(np.deg2rad(gaia["dec"]))))%360,
                gaia["dec"]+gaia["pmdec"]*dt/3.6e6)
    ra,dec=position(float(np.median(epoch)))
    tree=cKDTree(unit_vectors(cat["RA_OK"],cat["DEC_OK"]))
    distance,candidate=tree.query(unit_vectors(ra,dec),k=3,distance_upper_bound=2*np.sin(np.deg2rad(5/3600)/2))
    safe=np.minimum(candidate,len(cat)-1)
    seps=np.full(candidate.shape,np.inf)
    for j in range(3):
        pra,pdec=position(epoch[safe[:,j]])
        xyz=unit_vectors(pra,pdec)
        refxyz=unit_vectors(cat["RA_OK"][safe[:,j]],cat["DEC_OK"][safe[:,j]])
        chord=np.linalg.norm(xyz-refxyz,axis=1)
        angle=np.rad2deg(2*np.arcsin(np.minimum(chord/2,1)))*3600
        seps[:,j]=np.where(np.isfinite(distance[:,j]),angle,np.inf)
    nearest=np.argmin(seps,axis=1)
    index=safe[np.arange(len(ra)),nearest]
    sep=seps[np.arange(len(ra)),nearest]
    matched=sep<1
    ambiguous=((seps<1).sum(axis=1)>1)
    values,counts=np.unique(cat["OBJ_ID"][index[matched]],return_counts=True)
    ambiguous |= matched&np.isin(cat["OBJ_ID"][index],values[counts>1])
    out={"ps1_match":matched,"ps1_ambiguous":ambiguous,"ps1_sep_arcsec":np.where(matched,sep,np.nan),
         "ps1_obj_id":np.where(matched,cat["OBJ_ID"][index],0),
         "ps1_epoch":np.where(matched,epoch[index],np.nan)}
    median=cat["MEDIAN"][index].astype(float)
    nobs=cat["NMAG_OK"][index]; scatter=cat["STDEV"][index]
    gi=median[:,0]-median[:,2]
    color_good=matched&~ambiguous&(gi>.4)&(gi<2.7)
    for j in (0,2):
        color_good &= (nobs[:,j]>=3)&np.isfinite(scatter[:,j])&(scatter[:,j]>0)&(scatter[:,j]<.03)&np.isfinite(median[:,j])&(median[:,j]>15)&(median[:,j]<20)
    out["ps1_gi"]=np.where(matched,gi,np.nan)
    for b,j in (("g",0),("r",1),("i",2),("z",3),("y",4)):
        out[f"ps1_{b}"]=np.where(matched,median[:,j],np.nan)
        out[f"ps1_{b}_nobs"]=np.where(matched,nobs[:,j],0).astype("i2")
        out[f"ps1_{b}_stdev"]=np.where(matched,scatter[:,j],np.nan)
        if b in "grz":
            good=color_good&(nobs[:,j]>=3)&np.isfinite(scatter[:,j])&(scatter[:,j]>0)&(scatter[:,j]<.03)&np.isfinite(median[:,j])&(median[:,j]>15)&(median[:,j]<20)
            out[f"ps1_{b}_quality"]=good
            out[f"ps1_{b}_decam"]=np.where(good,ps1_decam(median,b),np.nan)
    metadata={"root":str(PS1_ROOT),"nside":32,"ordering":"RING","catalog_rows_in_margin":len(cat),"files":sources,
              "epoch_jyear":stats(epoch),"match_radius_arcsec":1,"search_radius_arcsec":5,
              "quality":"NMAG_OK>=3, 0<STDEV<0.03, 15<median<20 in band and g/i; 0.4<g-i<2.7; unique 1 arcsec propagated match"}
    return out,metadata


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/name,dpi=170,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def plots(ref,raw,phot,gaia,col,fail,ps1,target,summary):
    group=fail["group"];g=gaia["phot_g_mean_mag"]
    color=col["bp_rp_checked"]
    mg=g-10+5*np.log10(ref["p_corr"])
    fig,axes=plt.subplots(1,3,figsize=(15,5.3))
    for j,name in enumerate(GROUP_NAMES):
        m=group==j
        axes[0].scatter(color[m],mg[m],s=7,alpha=.6,color=GROUP_COLORS[j],label=f"{name}: {m.sum():,}")
        axes[1].hist(g[m],bins=np.arange(6,22,.5),histtype="step",color=GROUP_COLORS[j])
    axes[0].set(xlabel="Validated Gaia BP-RP",ylabel="$M_G$",ylim=(17,-6),xlim=(-.5,4))
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Gaia G [mag]",ylabel="Reference sources",title="Brightness by exclusive failure group")
    matrix=np.array(summary["satur_fracmasked_matrix"])
    im=axes[2].imshow(matrix,cmap="Blues")
    axes[2].set_xticks([0,1],["FRACMASKED pass","FRACMASKED fail"])
    axes[2].set_yticks([0,1],["No SATUR","SATUR"])
    for y in range(2):
        for x in range(2):
            axes[2].text(x,y,f"{matrix[y,x]:,}",ha="center",va="center",fontsize=15,
                         color="white" if matrix[y,x]>matrix.max()*.55 else "black")
    axes[2].set_title("All 6,942 reference sources")
    fig.suptitle("Joint optical-quality failures in the same 10-degree field")
    fig.tight_layout();save(fig,"01_joint_failures.png")

    fig,axes=plt.subplots(1,3,figsize=(14,5.5),sharex=True,sharey=True)
    strict=ref["anchor"]
    for ax,x,mask,title in [(axes[0],gaia["bp_rp"],np.ones(len(ref),bool),"Original catalog color"),
                             (axes[1],color,col["bp_rp_valid"],"BP/RP flux and NOBS verified"),
                             (axes[2],color,col["bp_rp_valid"]&strict,"Fixed strict anchors")]:
        finite=mask&np.isfinite(x)
        ax.scatter(x[finite],mg[finite],s=5,alpha=.55,color="#326b9a")
        ax.set(xlabel="Gaia BP-RP",xlim=(-.5,4),ylim=(17,-6),title=f"{title}\nN={finite.sum():,}");ax.grid(alpha=.15)
    axes[0].set_ylabel("$M_G$ [mag]")
    fig.suptitle("Gaia color audit: observed magnitudes, corrected parallax")
    fig.tight_layout();save(fig,"02_gaia_color_check.png")

    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    for j,b in enumerate("grz"):
        q=ps1[f"ps1_{b}_quality"]&phot[f"valid_{b}"]
        delta=ps1[f"delta_{b}"]
        for k in range(len(GROUP_NAMES)):
            mask=q&(group==k)
            for ax,x in [(axes[0,j],ps1[f"ps1_{b}_decam"]),(axes[1,j],ps1["ps1_gi"])]:
                ax.scatter(x[mask],delta[mask],s=5,alpha=.45,color=GROUP_COLORS[k])
        for ax in axes[:,j]:
            ax.axhline(0,color=".3",lw=1);ax.axhline(.05,color=".6",ls=":");ax.axhline(-.05,color=".6",ls=":")
            ax.set_ylim(-.5,.5);ax.grid(alpha=.15)
        axes[0,j].set(xlabel=f"PS1-predicted DECam {b}",title=f"{b}: N={q.sum():,}")
        axes[1,j].set_xlabel("PS1 g-i")
    for ax in axes[:,0]:ax.set_ylabel("Legacy - transformed PS1 [mag]")
    handles=[plt.Line2D([],[],marker="o",linestyle="",color=c,label=n) for n,c in zip(GROUP_NAMES,GROUP_COLORS)]
    fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=8)
    fig.suptitle("Measured grz comparison, with published PS1-to-DECam color terms")
    fig.tight_layout(rect=(0,.08,1,.96));save(fig,"03_ps1_residuals.png")

    pop=[("Target points",target["grz_base"]),("Fixed strict anchors",target["anchor"]),
         ("Quality DESI stars",target["grz_base"]&target["desi_quality"]),
         ("DESI QSO / GALAXY",target["grz_base"]&target["desi_known_contaminant"]&~target["desi_match_ambiguous"])]
    definitions=[("gr0","rz0","grz_base",(-1,3),(-1,3.5),"grz"),
                 ("rz0","zw10","grz_w1",(-1,3.5),(-2,6),"grzW1"),
                 ("zw10","w1w20","grz_w1_w2",(-2,6),(-2,2),"grzW1W2")]
    labels={"gr0":"$(g-r)_0$","rz0":"$(r-z)_0$","zw10":"$(z-W1)_0$","w1w20":"$(W1-W2)_0$"}
    coverage={}
    for plane,(x,y,tier,xrange,yrange,tname) in enumerate(definitions):
        fig,axes=plt.subplots(1,4,figsize=(15,4.5),sharex=True,sharey=True)
        bins=(np.linspace(*xrange,101),np.linspace(*yrange,101))
        hists=[];meta=[]
        for title,mask in pop:
            mask=mask&target[tier]&np.isfinite(target[x])&np.isfinite(target[y])
            h,xe,ye=np.histogram2d(target[x][mask],target[y][mask],bins=bins)
            n=int(mask.sum())
            area=(xe[1]-xe[0])*(ye[1]-ye[0])
            hists.append(h/max(n,1)/area)
            meta.append({"population":title,"n":n,"in_view":int(h.sum()),"outside_view":int(n-h.sum())})
        vmax=max([float(h.max()) for h in hists]+[.01]);norm=LogNorm(vmin=1e-4,vmax=vmax)
        for ax,d,info in zip(axes,hists,meta):
            mesh=ax.pcolormesh(xe,ye,np.ma.masked_equal(d.T,0),norm=norm,cmap="Blues",rasterized=True)
            ax.set(xlabel=labels[x],title=f"{info['population']}\n{info['in_view']:,} in view / {info['n']:,}")
        axes[0].set_ylabel(labels[y])
        fig.colorbar(mesh,ax=axes.tolist(),label="Probability density [mag$^{-2}$]",fraction=.025,pad=.02)
        fig.suptitle(f"{tname}: measured input-color coverage (AB system)")
        fig.subplots_adjust(left=.055,right=.89,wspace=.12,bottom=.16,top=.78)
        save(fig,f"0{plane+4}_colors_{tname}.png")
        coverage[tname]=meta
    summary["color_planes"]=coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args();require_compute();started=time.time()
    if args.plot_only:
        ref=fitsio.read(str(OUT/"reference.fits"),ext=1)
        target=fitsio.read("data/pilot_10deg_v1/field_diagnostics.fits",ext=1)
        summary=json.loads((OUT/"summary.json").read_text())
        raw={name.upper():ref[name] for name in ref.dtype.names if name.startswith(("frac","mask","flux","nobs","mw_"))}
        phot={name:ref[name] for name in ref.dtype.names}
        gaia={name[5:]:ref[name] for name in ref.dtype.names if name.startswith("gaia_")}
        col={k:ref[k] for k in ("bp_rp_checked","bp_rp_valid","bp_rp_precise")}
        fail={"group":ref["audit_group"]};ps1={name:ref[name] for name in ref.dtype.names}
        plots(ref,raw,phot,gaia,col,fail,ps1,target,summary)
        write_json(OUT/"summary.json",summary);write_json("manifests/pilot_audit.json",summary);return
    if (OUT/"summary.json").exists():raise FileExistsError(OUT/"summary.json")
    OUT.mkdir(parents=True,exist_ok=True)
    task=json.loads(Path("configs/pilot_10deg.json").read_text());config=json.loads(Path(task["base_config"]).read_text())
    config.update(sweep_files=task["sweep_files"])
    target=fitsio.read(str(Path(task["output_dir"])/"field_diagnostics.fits"),ext=1)
    ref=target[canonical_reference(target)].copy()
    assert len(ref)==6942 and int(target["anchor"].sum())==1802
    raw,source_info=raw_reference(config,ref)
    phot=derive_photometry(raw,**config["photometry"])
    assert np.array_equal(phot["grz_high_quality"],ref["grz_high_quality"])
    gaia=join_gaia(ref["source_id"],config["gaia_root"])
    labels=derive_gaia_labels(gaia);assert labels["anchor_astrometry"].all()
    np.testing.assert_allclose(labels["p_corr"],ref["p_corr"],rtol=1e-6,atol=1e-7)
    col=checked_gaia_color(gaia);fail=failure_groups(raw,phot)
    ps1,ps1_info=ps1_reference(gaia)
    for b in "grz":ps1[f"delta_{b}"]=phot[b]-ps1[f"ps1_{b}_decam"]
    print(json.dumps({"phase":"references matched","reference":len(ref),"ps1":int(ps1["ps1_match"].sum())}),flush=True)
    group=fail["group"]
    matrix=[[int(((fail["satur"]==a)&(fail["fracmasked_fail"]==b)).sum()) for b in (0,1)] for a in (0,1)]
    zero=gaia["bp_rp"]==0
    summary={"field_bounds_deg":[0,-10,10,0],"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"compute_host":socket.gethostname(),"job_id":os.environ["SLURM_JOB_ID"],
             "reference_count":len(ref),"strict_anchor_count":int(target["anchor"].sum()),"raw_sources":source_info,
             "canonical_rule":"Gaia source ID; prefer strict optical pass, then measured grz, then smallest summed grz error, then Legacy ID",
             "satur_fracmasked_matrix":matrix,"groups":[],"per_band":{},
             "gaia_color":{"catalog_zero":int(zero.sum()),"zero_missing_bp_rp":int((zero&~col["bp_rp_valid"]).sum()),
                           "zero_valid_bp_rp":int((zero&col["bp_rp_valid"]).sum()),"valid_color":int(col["bp_rp_valid"].sum()),
                           "precise_color":int(col["bp_rp_precise"].sum()),"catalog_difference_gt_0p001":int(col["bp_rp_catalog_disagrees"].sum()),
                           "zero_bp_nobs":stats(gaia["phot_bp_n_obs"][zero]),"zero_rp_nobs":stats(gaia["phot_rp_n_obs"][zero])},
             "ps1":ps1_info,"layer_counts":{},"photometry_residuals":{},"strata":{}}
    mg=gaia["phot_g_mean_mag"]-10+5*np.log10(ref["p_corr"])
    for j,name in enumerate(GROUP_NAMES):
        mask=group==j
        summary["groups"].append({"index":j,"name":name,"n":int(mask.sum()),"valid_gaia_color":int((mask&col["bp_rp_valid"]).sum()),
                                  "G":stats(gaia["phot_g_mean_mag"][mask]),"M_G":stats(mg[mask]),
                                  "desi_giants":int((mask&ref["desi_quality"]&(ref["desi_logg"]<3.2)).sum()),
                                  "apogee_giants":int((mask&ref["apogee_quality"]&(ref["apogee_logg"]<3.2)).sum())})
    for label,values,edges in [("G",gaia["phot_g_mean_mag"],[6,10,12,14,16,18,20,22]),
                               ("BP_RP",col["bp_rp_checked"],[-.5,.5,1,1.5,2,3,4.5]),
                               ("M_G",mg,[-6,0,2,4,6,8,10,14,18])]:
        summary["strata"][label]=[{"low":lo,"high":hi,"group_counts":[int(((values>=lo)&(values<hi)&(group==j)).sum()) for j in range(7)]} for lo,hi in zip(edges[:-1],edges[1:])]
    for b,bit,allbit in (("g",2,5),("r",3,6),("z",4,7)):
        mask=raw["MASKBITS"]
        summary["per_band"][b]={"satur":int(((mask&(1<<bit))!=0).sum()),"allmask":int(((mask&(1<<allbit))!=0).sum()),
                                "fracmasked_fail":int((~np.isfinite(raw[f"FRACMASKED_{b.upper()}"])|(raw[f"FRACMASKED_{b.upper()}"]>=.4)).sum()),
                                "invalid_measurement":int((~phot[f"valid_{b}"]).sum())}
        q=ps1[f"ps1_{b}_quality"]&phot[f"valid_{b}"]
        summary["photometry_residuals"][b]={name:stats(ps1[f"delta_{b}"][q&(group==j)]) for j,name in enumerate(GROUP_NAMES)}
        summary["photometry_residuals"][b]["by_reference_mag"]=[{"lo":lo,"hi":lo+1,"n":int((q&(ps1[f"ps1_{b}_decam"]>=lo)&(ps1[f"ps1_{b}_decam"]<lo+1)).sum()),
            "groups":{name:stats(ps1[f"delta_{b}"][q&(group==j)&(ps1[f"ps1_{b}_decam"]>=lo)&(ps1[f"ps1_{b}_decam"]<lo+1)]) for j,name in enumerate(GROUP_NAMES)}} for lo in range(15,20)]
    common=np.logical_and.reduce([ps1[f"ps1_{b}_quality"]&phot[f"valid_{b}"] for b in "grz"])
    candidates=np.isin(group,[1,2,3])
    summary["recovery_agreement"]={str(tol):{"candidate_count":int((common&candidates&np.logical_and.reduce([np.abs(ps1[f"delta_{b}"])<tol for b in "grz"])).sum()),
        "strict_count":int((common&(group==0)&np.logical_and.reduce([np.abs(ps1[f"delta_{b}"])<tol for b in "grz"])).sum())} for tol in (.03,.05,.1)}
    summary["recovery_agreement"]["common_reference_candidates"]=int((common&candidates).sum())
    summary["recovery_agreement"]["common_reference_strict"]=int((common&(group==0)).sum())
    for tier,flag,precise in (("grz","grz_base","grz_high_quality"),("grzW1","grz_w1","grz_w1_high_quality"),("grzW1W2","grz_w1_w2","full_high_quality")):
        summary["layer_counts"][tier]={"target":int(target[flag].sum()),"strict_anchors_usable_ir":int((target["anchor"]&target[flag]).sum()),
                                       "strict_anchors_precise_ir":int((target["anchor"]&target[precise]).sum())}
    calibrated,domain=rvs_metallicity(target["desi_feh"],target["desi_teff"],target["desi_logg"])
    use=target["grz_base"]&target["desi_quality"]
    summary["desi_metallicity"]={"quality_in_grz":int(use.sum()),"calibrated_in_grz":int((use&domain).sum()),
        "quality_in_anchors":int((target["anchor"]&target["desi_quality"]).sum()),"calibrated_in_anchors":int((target["anchor"]&target["desi_quality"]&domain).sum()),
        "correction":stats((calibrated-target["desi_feh"])[use&domain]),"bounds":"4200<=Teff<6600;1.8<=logg<5.5; author feh_correct RVS"}
    metallicity=structured({"ls_id":target["ls_id"][use],"source_id":target["source_id"][use],"feh_raw":target["desi_feh"][use],
        "feh_calibrated":calibrated[use],"calibration_in_range":domain[use],"teff":target["desi_teff"][use],"logg":target["desi_logg"][use]})
    fitsio.write(str(OUT/"desi_metallicity.fits"),metallicity,extname="RVS_CALIBRATION",clobber=False)
    columns={name:ref[name] for name in ref.dtype.names}
    columns.update({name.lower():raw[name] for name in raw.dtype.names});columns.update(phot)
    columns.update({"gaia_"+k:v for k,v in gaia.items()});columns.update(col);columns.update(ps1);columns["audit_group"]=group
    fitsio.write(str(OUT/"reference.fits"),structured(columns),extname="AUDIT",clobber=False)
    assert sum(x["n"] for x in summary["groups"])==6942
    summary["validation"]={"group_partition":True,"strict_anchor_ids_unchanged":int(ref["anchor"].sum())==1802,
                           "strict_optical_predicate":True,"gaia_parallax_labels_unchanged":True,"source_files":4}
    assert all(summary["validation"][x] for x in ("group_partition","strict_anchor_ids_unchanged","strict_optical_predicate","gaia_parallax_labels_unchanged"))
    plots(ref,raw,phot,gaia,col,fail,ps1,target,summary)
    summary["elapsed_seconds"]=time.time()-started
    write_json(OUT/"summary.json",summary);write_json("manifests/pilot_audit.json",summary)
    print(json.dumps({k:summary[k] for k in ("groups","gaia_color","layer_counts","desi_metallicity","recovery_agreement","elapsed_seconds")},indent=2),flush=True)

if __name__=="__main__":main()
