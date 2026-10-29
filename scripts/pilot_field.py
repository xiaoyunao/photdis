#!/usr/bin/env python3
"""Inspect one configured 10-degree field and produce its selection tables and plots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from photdis.gaia import join_gaia, derive_gaia_labels
from photdis.audit import checked_gaia_color
from photdis.photometry import derive_photometry
from photdis.stage1 import (CORE_COLUMNS, PHOT_COLUMNS, galactic, legacy_ids,
                           load_spectroscopy, match_spectroscopy, spec_columns,
                           structured, require_compute, write_json)


def field_mask(ra, dec, bounds):
    r0, d0, r1, d1 = bounds
    return (ra >= r0) & (ra < r1) & (dec >= d0) & (dec < d1)


def unique_rows(mask, ids, score=None):
    rows = np.flatnonzero(mask)
    if not len(rows):
        return mask.copy()
    if score is None:
        score = np.zeros(len(ids))
    order = np.lexsort((rows, score[rows], ids[rows]))
    rows = rows[order]
    first = np.r_[True, ids[rows[1:]] != ids[rows[:-1]]]
    result = np.zeros(len(ids), bool)
    result[rows[first]] = True
    return result


def finite_stats(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max())}


def cut_entry(label, count, previous):
    return {"condition": label, "remaining": int(count),
            "removed": int(previous-count) if previous is not None else 0}


def make_plots(raw, phot, gaia, labels, spec, masks, flows, summary, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 120})
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    english = [
        ["10 x 10 degree field", "PSF", "|b| > 30 deg", "0 <= E(B-V) < 0.03",
         "grz NOBS >= 1", "Positive grz flux", "Positive grz inverse variance",
         "Positive MW transmission", "grz S/N >= 5", "FRACMASKED < 0.4",
         "FRACIN > 0.7", "FRACFLUX < 0.5", "Saturation / allmask / bailout cut", "grz errors < 0.03 mag"],
        ["High-quality grz", "Matched Gaia DR3", "Finite parallax, positive error",
         "5p / 6p astrometric solution", "0 < RUWE < 1.4", "Visibility periods >= 9",
         "Not duplicated_source", "Valid parallax zero point", "Corrected parallax S/N >= 20",
         "Gaia QSO / galaxy flag cut", "Spectral nonstar classification cut", "One row per Gaia source"]]
    for ax, flow, names, title in zip(axes, flows, english, ["Optical selection", "Gaia anchor selection"]):
        numbers = [x["remaining"] for x in flow]
        y = np.arange(len(numbers))
        ax.barh(y, numbers, color="#326b9a")
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1, max(numbers)*5)
        ax.set_xlabel("Remaining rows")
        ax.set_title(title)
        for yy, value in zip(y, numbers):
            ax.text(max(value, 1)*1.08, yy, f"{value:,}", va="center", fontsize=9)
    r0,d0,r1,d1=summary["field"]["bounds_deg"]
    fig.suptitle(f"Pilot field: RA {r0}–{r1} deg, Dec {d0}–{d1} deg")
    fig.tight_layout()
    path = output / "01_selection_counts.png"
    fig.savefig(path, dpi=170); plt.close(fig); files.append(str(path))

    bp_rp = checked_gaia_color(gaia)["bp_rp_checked"]
    mg = np.full(len(raw), np.nan)
    ok = labels["p_corr"] > 0
    mg[ok] = gaia["phot_g_mean_mag"][ok]-10+5*np.log10(labels["p_corr"][ok])
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    cmd_counts = {}
    for ax, key, title, color in zip(axes,
            ["gaia_reference", "gaia_with_grz", "anchor"],
            ["Gaia astrometric selection", "+ measured grz, S/N >= 5", "Final Legacy + Gaia anchors"],
            ["#637989", "#326b9a", "#b85c24"]):
        mask = masks[key] & np.isfinite(bp_rp) & np.isfinite(mg)
        ax.scatter(bp_rp[mask], mg[mask], s=5, alpha=.55, color=color, rasterized=True)
        in_view = mask & (bp_rp >= -.5) & (bp_rp <= 4) & (mg >= -6) & (mg <= 17)
        cmd_counts[key] = {"finite": int(mask.sum()), "in_view": int(in_view.sum())}
        ax.set_title(f"{title}\nN = {int(mask.sum()):,}")
        ax.set(xlim=(-.5, 4), ylim=(17, -6), xlabel="Gaia BP - RP [mag]")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Gaia absolute magnitude $M_G$ [mag]")
    fig.suptitle("Gaia color–absolute magnitude diagram: pilot field")
    fig.text(.06, .025, "$M_G = G - 10 + 5\\log_{10}(\\varpi_{corr}/\\mathrm{mas})$. G, BP and RP are observed magnitudes; E(B-V) < 0.03.", fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.94))
    path = output / "02_gaia_cmd.png"
    fig.savefig(path, dpi=170); plt.close(fig); files.append(str(path))
    summary["cmd_counts"] = cmd_counts

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9))
    for survey, color in [("desi", "#326b9a"), ("apogee", "#b85c24")]:
        matched = spec[f"{survey}_match"] & phot["grz_base"]
        quality = spec[f"{survey}_quality"] & phot["grz_base"]
        for method, style in [(1, "-"), (2, "--")]:
            values = spec[f"{survey}_separation_arcsec"][matched & (spec[f"{survey}_match_method"] == method)]
            if len(values):
                axes[0,0].hist(values, bins=np.geomspace(1e-5,10,81), histtype="step", color=color,
                    linestyle=style, label=f"{survey.upper()} {'ID' if method==1 else 'position'} N={len(values):,}")
        axes[0,1].scatter(spec[f"{survey}_teff"][quality], spec[f"{survey}_logg"][quality], s=12,
                          alpha=.65, color=color, label=f"{survey.upper()} N={int(quality.sum()):,}")
        values = spec[f"{survey}_feh"][quality]
        if len(values):
            axes[1,0].hist(values, bins=np.linspace(-4,1,41), histtype="step", color=color,
                           label=f"{survey.upper()} N={len(values):,}")
    axes[0,0].set(xlabel="Coordinate separation [arcsec]", ylabel="Matched rows", title="grz sample: matching methods", xlim=(1e-5,10))
    axes[0,0].set_xscale("log")
    axes[0,0].set_yscale("symlog", linthresh=1)
    axes[0,1].set(xlabel="$T_{eff}$ [K]", ylabel="log g [dex]", xlim=(10000,2500), ylim=(5.6,-.6), title="Quality spectroscopy in grz sample")
    axes[1,0].set(xlabel="[Fe/H] [dex]", ylabel="Rows", title="Quality metallicities", xlim=(-4,1))
    reference = masks["gaia_reference"] & np.isfinite(bp_rp) & np.isfinite(mg)
    axes[1,1].scatter(bp_rp[reference], mg[reference], s=4, color=".8", label="Gaia reference")
    for survey, color in [("desi", "#326b9a"), ("apogee", "#b85c24")]:
        mask = reference & spec[f"{survey}_quality"]
        axes[1,1].scatter(bp_rp[mask], mg[mask], s=18, color=color,
                          label=f"{survey.upper()} N={int(mask.sum()):,}")
    axes[1,1].set(xlabel="Gaia BP - RP [mag]", ylabel="$M_G$ [mag]", xlim=(-.5,4), ylim=(17,-6), title="Spectroscopy on the Gaia diagram")
    for ax in axes.flat:
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.suptitle("Spectroscopic matches in the 10 x 10 degree pilot field")
    fig.tight_layout(rect=(0,0,1,.96))
    path = output / "03_spectroscopy.png"
    fig.savefig(path, dpi=170); plt.close(fig); files.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ext = masks["extended"]
    snr = labels["snr_raw"][ext]
    edges = np.unique(np.r_[-np.geomspace(1,1e4,60)[::-1],np.linspace(-1,1,21),np.geomspace(1,1e4,60)])
    axes[0].hist(snr, bins=edges, histtype="step", color="#326b9a")
    axes[0].set_xscale("symlog", linthresh=1); axes[0].set_yscale("log")
    axes[0].axvline(0,color=".6"); axes[0].axvline(20,color=".6",linestyle=":")
    axes[0].set(xlabel="Raw signed parallax S/N", ylabel="Rows per bin", title=f"Extended: {int(ext.sum()):,}; negative parallax: {int((ext & (gaia['parallax']<0)).sum()):,}")
    for mask, name, color in [(phot["grz_base"],"Measured grz","#637989"),(masks["anchor"],"Anchors","#b85c24")]:
        values=phot["g0"][mask]; counts, bins=np.histogram(values[np.isfinite(values)], bins=np.linspace(10,27,69))
        n=int(np.isfinite(values).sum())
        axes[1].stairs(counts/max(n,1)/np.diff(bins),bins,label=f"{name}: N={n:,}",color=color)
    axes[1].set(xlabel="Legacy extinction-corrected $g_0$ [mag]", ylabel="Probability density [mag$^{-1}$]", title="Optical brightness coverage")
    axes[1].legend(); axes[0].grid(alpha=.15); axes[1].grid(alpha=.15)
    fig.tight_layout()
    path=output/"04_parallax_brightness.png"
    fig.savefig(path,dpi=170);plt.close(fig);files.append(str(path))
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/pilot_10deg.json")
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args()
    require_compute()
    task=json.loads(Path(args.config).read_text())
    cfg=json.loads(Path(task["base_config"]).read_text())
    out=Path(task["output_dir"])
    if args.plot_only:
        summary=json.loads((out/"summary.json").read_text())
        table=fitsio.read(str(out/"field_diagnostics.fits"),ext=1)
        columns={name:table[name] for name in table.dtype.names}
        gaia={name[5:]:value for name,value in columns.items() if name.startswith("gaia_")}
        if "phot_bp_n_obs" not in gaia:
            fresh=join_gaia(columns["source_id"],cfg["gaia_root"])
            for name in ("phot_bp_n_obs","phot_rp_n_obs","phot_bp_mean_flux",
                         "phot_rp_mean_flux","phot_bp_mean_flux_error","phot_rp_mean_flux_error"):
                gaia[name]=fresh[name]
        summary["figures"]=make_plots(table,columns,gaia,columns,columns,columns,
            [summary["optical_cutflow"],summary["anchor_cutflow"]],summary,Path(task["figure_dir"]))
        summary["plot_script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(out/"summary.json",summary)
        write_json(task["manifest_path"],summary)
        print(json.dumps({"figures":summary["figures"],"mode":"plot-only"}))
        return
    if (out/"summary.json").exists():
        raise FileExistsError(out/"summary.json")
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    bounds=task["bounds_deg"]
    names=task["sweep_files"]
    assert len(names)==4 and bounds[2]-bounds[0]==10 and bounds[3]-bounds[1]==10
    sweep_root=Path(task.get("sweep_root",cfg["sweep_root"]))
    chunks=[]; sources=[]
    for name in names:
        path=sweep_root/name
        table=fitsio.read(str(path),ext=1,columns=CORE_COLUMNS+PHOT_COLUMNS)
        table=table[field_mask(table["RA"],table["DEC"],bounds)]
        chunks.append(table)
        sources.append({"path":str(path),"bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns,"field_rows":len(table)})
        print(json.dumps({"file":name,"field_rows":len(table)}),flush=True)
    raw=np.concatenate(chunks);del chunks
    optical=[]
    def initial(label,selected):
        nonlocal raw
        before=len(raw);raw=raw[selected]
        optical.append(cut_entry(label,len(raw),before))
    optical.append(cut_entry(
        f"RA∈[{bounds[0]},{bounds[2]})°，Dec∈[{bounds[1]},{bounds[3]})°",
        len(raw), None))
    initial("TYPE=PSF",np.char.strip(raw["TYPE"].astype("U"))=="PSF")
    _,latitude=galactic(raw["RA"],raw["DEC"])
    initial("|b|>30°",np.abs(latitude)>cfg["sky"]["abs_b_min"])
    initial("E(B−V)有限且0≤E(B−V)<0.03",np.isfinite(raw["EBV"])&(raw["EBV"]>=0)&(raw["EBV"]<cfg["sky"]["ebv_max"]))
    phot=derive_photometry(raw,**cfg["photometry"])
    keep=np.ones(len(raw),bool)
    def phot_step(label,condition):
        nonlocal keep
        before=int(keep.sum());keep &= condition
        optical.append(cut_entry(label,keep.sum(),before))
    phot_step("g、r、z各波段NOBS≥1",np.logical_and.reduce([raw[f"NOBS_{b}"]>=1 for b in "GRZ"]))
    phot_step("g、r、z流量有限且>0",np.logical_and.reduce([np.isfinite(raw[f"FLUX_{b}"])&(raw[f"FLUX_{b}"]>0) for b in "GRZ"]))
    phot_step("g、r、z逆方差有限且>0",np.logical_and.reduce([np.isfinite(raw[f"FLUX_IVAR_{b}"])&(raw[f"FLUX_IVAR_{b}"]>0) for b in "GRZ"]))
    phot_step("g、r、z消光透射率有限且>0",np.logical_and.reduce([np.isfinite(raw[f"MW_TRANSMISSION_{b}"])&(raw[f"MW_TRANSMISSION_{b}"]>0) for b in "GRZ"]))
    phot_step("g、r、z各波段S/N≥5",np.logical_and.reduce([phot[f"snr_{b}"]>=5 for b in "grz"]))
    assert np.array_equal(keep,phot["grz_base"])
    phot_step("g、r、z各波段FRACMASKED<0.4",np.logical_and.reduce([np.isfinite(raw[f"FRACMASKED_{b}"])&(raw[f"FRACMASKED_{b}"]<.4) for b in "GRZ"]))
    phot_step("g、r、z各波段FRACIN>0.7",np.logical_and.reduce([np.isfinite(raw[f"FRACIN_{b}"])&(raw[f"FRACIN_{b}"]>.7) for b in "GRZ"]))
    phot_step("g、r、z各波段FRACFLUX<0.5",np.logical_and.reduce([np.isfinite(raw[f"FRACFLUX_{b}"])&(raw[f"FRACFLUX_{b}"]<.5) for b in "GRZ"]))
    phot_step("MASKBITS的2、3、4、5、6、7、10位为0",phot["grz_mask_clean"])
    phot_step("g、r、z各波段星等误差<0.03 mag",phot["grz_precise"])
    assert np.array_equal(keep,phot["grz_high_quality"])
    catalogs,_=load_spectroscopy(task["spectroscopy_run_dir"])
    sid=raw["REF_ID"].astype("i8").copy()
    sid[(np.char.strip(raw["REF_CAT"].astype("U"))!="G3")|(sid<=0)]=0
    spec={}
    for survey,catalog in catalogs.items():
        matched,method,ambiguous=match_spectroscopy(sid,raw["RA"],raw["DEC"],bounds,catalog,1.0)
        spec.update(spec_columns(catalog,matched,method,ambiguous,survey))
        separation=np.full(len(raw),np.nan)
        ok=matched>=0
        cat=catalog["table"]
        ra1,dec1=np.deg2rad(raw["RA"][ok]),np.deg2rad(raw["DEC"][ok])
        ra2,dec2=np.deg2rad(cat["ra"][matched[ok]]),np.deg2rad(cat["dec"][matched[ok]])
        separation[ok]=np.rad2deg(2*np.arcsin(np.sqrt(np.clip(np.sin((dec1-dec2)/2)**2+np.cos(dec1)*np.cos(dec2)*np.sin((ra1-ra2)/2)**2,0,1))))*3600
        spec[f"{survey}_separation_arcsec"]=separation.astype("f4")
    di=np.where(spec["desi_match"]&~spec["desi_match_ambiguous"],spec["desi_source_id"],0)
    ai=np.where(spec["apogee_match"]&~spec["apogee_match_ambiguous"],spec["apogee_source_id"],0)
    conflict=(di>0)&(ai>0)&(di!=ai)
    recover=(sid==0)&~conflict&(np.maximum(di,ai)>0)
    sid[recover]=np.maximum(di,ai)[recover]
    gaia=join_gaia(sid,cfg["gaia_root"])
    assert not gaia["gaia_file_missing"].any()
    labels=derive_gaia_labels(gaia)
    known=((spec["desi_known_contaminant"]&~spec["desi_match_ambiguous"])|(spec["apogee_known_contaminant"]&~spec["apogee_match_ambiguous"]))
    anchor_flow=[cut_entry("光学高质量样本",keep.sum(),None)]
    anchor=keep.copy()
    def anchor_step(label,condition):
        nonlocal anchor
        before=int(anchor.sum());anchor &= condition
        anchor_flow.append(cut_entry(label,anchor.sum(),before))
    for label,condition in [
        ("匹配Gaia DR3",gaia["gaia_match"]),
        ("视差有限，误差有限且>0",labels["finite_parallax"]),
        ("五或六参数解（31或95）",labels["astrometric_solution_valid"]),
        ("0<RUWE<1.4",np.isfinite(gaia["ruwe"])&(gaia["ruwe"]>0)&(gaia["ruwe"]<1.4)),
        ("visibility_periods_used≥9",gaia["visibility_periods_used"]>=9),
        ("duplicated_source=False",~gaia["duplicated_source"]),
        ("零点修正适用域内且修正值有效",labels["zpt_valid"]),
        ("修正视差S/N≥20",labels["snr_corr"]>=20),
        ("Gaia QSO和galaxy候选标记均为False",~labels["contaminant_candidate"]),
        ("光谱分类未标为QSO或GALAXY",~known)]:
        anchor_step(label,condition)
    assert np.array_equal(anchor,phot["grz_high_quality"]&labels["anchor_astrometry"]&~labels["contaminant_candidate"]&~known)
    pre_unique=anchor.copy()
    score=np.nan_to_num(phot["sigma_g"]+phot["sigma_r"]+phot["sigma_z"],nan=1e6)
    anchor=unique_rows(anchor,sid,score)
    anchor_flow.append(cut_entry("每个Gaia source_id保留测光误差最小的一行",anchor.sum(),pre_unique.sum()))
    extended=phot["grz_base"]&labels["finite_parallax"]
    extended_quality=extended&labels["astro_quality"]&labels["zpt_valid"]&phot["grz_clean"]&~labels["contaminant_candidate"]&~known
    masks={"gaia_reference":unique_rows(labels["anchor_astrometry"],sid),
           "gaia_with_grz":unique_rows(labels["anchor_astrometry"]&phot["grz_base"],sid),
           "anchor":anchor,"extended":extended,"extended_quality":extended_quality}
    spectral={}
    for survey in catalogs:
        info={}
        for name,base in [("domain_psf",np.ones(len(raw),bool)),("grz_base",phot["grz_base"]),("anchor",anchor)]:
            matched=base&spec[f"{survey}_match"]
            quality=base&spec[f"{survey}_quality"]
            cls={str(k):int(v) for k,v in zip(*np.unique(spec[f"{survey}_spectype"][matched].astype("U"),return_counts=True))}
            info[name]={"matched_rows":int(matched.sum()),"unique_spectra":len(np.unique(spec[f"{survey}_object_id"][matched])),
                        "id_match":int((matched&(spec[f"{survey}_match_method"]==1)).sum()),
                        "coordinate_match":int((matched&(spec[f"{survey}_match_method"]==2)).sum()),
                        "ambiguous":int((matched&spec[f"{survey}_match_ambiguous"]).sum()),"classes":cls,
                        "quality_rows":int(quality.sum()),"quality_unique_spectra":len(np.unique(spec[f"{survey}_object_id"][quality])),
                        "quality_logg_lt3p2":int((quality&(spec[f"{survey}_logg"]<3.2)).sum()),
                        "quality_feh_lt_m1p5":int((quality&(spec[f"{survey}_feh"]<-1.5)).sum()),
                        "teff":finite_stats(spec[f"{survey}_teff"][quality]),"logg":finite_stats(spec[f"{survey}_logg"][quality]),
                        "feh":finite_stats(spec[f"{survey}_feh"][quality]),
                        "separation_arcsec":finite_stats(spec[f"{survey}_separation_arcsec"][matched])}
        subset=phot["grz_base"]&spec[f"{survey}_match"]
        flow=[cut_entry("grz母样本内光谱匹配",subset.sum(),None)]
        checks=[("坐标关联无歧义",~spec[f"{survey}_match_ambiguous"])]
        reason=spec[f"{survey}_quality_reason"]
        if survey=="desi":
            checks += [("PRIMARY=True",(reason&(1<<5))==0),
                       ("RR_SPECTYPE=STAR",(reason&1)==0),
                       ("RVS_WARN=0且SUCCESS=True",(reason&(1<<1))==0),
                       ("SN_R≥10",(reason&(1<<2))==0),
                       ("0≤VSINI<30 km/s",(reason&(1<<3))==0),
                       ("Teff、logg、[Fe/H]有效且误差>0",(reason&(1<<4))==0)]
        else:
            checks += [("ASPCAPFLAG的23、24、31、32、33、36、40、41位和STARFLAG的0、3、18、22位为0",(reason&(1<<1))==0),
                       ("SNR≥50",(reason&(1<<2))==0),
                       ("FE_H_FLAG=0",(reason&(1<<6))==0),
                       ("Teff、logg、[Fe/H]有效且误差>0",(reason&(1<<4))==0)]
        checks += [("光谱坐标有效",(reason&(1<<7))==0)]
        for label,passed in checks:
            before=int(subset.sum());subset &= passed
            flow.append(cut_entry(label,subset.sum(),before))
        assert np.array_equal(subset,phot["grz_base"]&spec[f"{survey}_quality"])
        info["quality_cutflow_grz"]=flow
        spectral[survey]=info
    ls=legacy_ids(raw)
    summary={"field":task,"source_files":sources,"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             "compute_host":socket.gethostname(),"optical_cutflow":optical,"anchor_cutflow":anchor_flow,
             "domain_psf_rows":len(raw),"grz_base_rows":int(phot["grz_base"].sum()),"gaia_matches_domain":int(gaia["gaia_match"].sum()),
             "anchor_rows":int(anchor.sum()),"anchor_duplicate_rows_removed":int(pre_unique.sum()-anchor.sum()),
             "gaia_identity_recovered":int(recover.sum()),"spectral_identity_conflicts":int(conflict.sum()),
             "extended_rows":int(extended.sum()),"extended_unique_gaia":len(np.unique(sid[extended])),
             "negative_parallax":int((extended&(gaia["parallax"]<0)).sum()),"zero_parallax":int((extended&(gaia["parallax"]==0)).sum()),
             "weak_corrected_snr_lt5":int((extended&labels["zpt_valid"]&(labels["snr_corr"]<5)).sum()),
             "extended_quality_rows":int(extended_quality.sum()),"negative_in_extended_quality":int((extended_quality&(gaia["parallax"]<0)).sum()),
             "grz_w1_rows":int(phot["grz_w1"].sum()),"grz_w1_w2_rows":int(phot["grz_w1_w2"].sum()),
             "gaia_reference_photometry":{
                 "rows":int(masks["gaia_reference"].sum()),
                 "grz_base":int((masks["gaia_reference"]&phot["grz_base"]).sum()),
                 "grz_high_quality":int((masks["gaia_reference"]&phot["grz_high_quality"]).sum()),
                 "fracmasked_fail":int((masks["gaia_reference"]&~np.logical_and.reduce([np.isfinite(raw[f"FRACMASKED_{b}"])&(raw[f"FRACMASKED_{b}"]<.4) for b in "GRZ"])).sum()),
                 "fracin_fail":int((masks["gaia_reference"]&~np.logical_and.reduce([np.isfinite(raw[f"FRACIN_{b}"])&(raw[f"FRACIN_{b}"]>.7) for b in "GRZ"])).sum()),
                 "fracflux_fail":int((masks["gaia_reference"]&~np.logical_and.reduce([np.isfinite(raw[f"FRACFLUX_{b}"])&(raw[f"FRACFLUX_{b}"]<.5) for b in "GRZ"])).sum()),
                 "maskbits_fail":int((masks["gaia_reference"]&~phot["grz_mask_clean"]).sum()),
                 "sigma_fail":int((masks["gaia_reference"]&~np.logical_and.reduce([phot[f"sigma_{b}"]<.03 for b in "grz"])).sum()),
                 "maskbit_counts":{str(bit):int((masks["gaia_reference"]&((raw["MASKBITS"]&(1<<bit))!=0)).sum()) for bit in [1,2,3,4,5,6,7,10,11,12,13]}},
             "spectroscopy":spectral,"desi_apogee_overlap_grz":int((phot["grz_base"]&spec["desi_match"]&spec["apogee_match"]).sum()),
             "validation":{"unique_legacy_ids":len(np.unique(ls))==len(ls),"exact_grz_base_predicates":True,
                           "exact_anchor_predicates":True,"unique_anchor_gaia_ids":len(np.unique(sid[anchor]))==int(anchor.sum())}}
    assert all(summary["validation"].values())
    selected=phot["grz_base"]|gaia["gaia_match"]|spec["desi_match"]|spec["apogee_match"]
    cols={"ls_id":ls,"source_id":sid,"ra":raw["RA"],"dec":raw["DEC"],"ebv":raw["EBV"],
          "mw_transmission_g":raw["MW_TRANSMISSION_G"],
          **phot,**labels,**masks,**spec,**{"gaia_"+k:v for k,v in gaia.items()}}
    fitsio.write(str(out/"field_diagnostics.fits"),structured({k:v[selected] for k,v in cols.items()}),extname="PILOT",clobber=False)
    summary["figures"]=make_plots(raw,phot,gaia,labels,spec,masks,[optical,anchor_flow],summary,Path(task["figure_dir"]))
    summary["elapsed_seconds"]=time.time()-start
    summary["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out/"summary.json",summary)
    Path(task["manifest_path"]).parent.mkdir(parents=True,exist_ok=True)
    write_json(task["manifest_path"],summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
