#!/usr/bin/env python3
"""Check NGC 2419 Legacy northern photometry against transformed PS1 stars."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps"), str(ROOT / "scripts")]
import fitsio
import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

from cluster_rgb_phase import CONFIG, classify
from photdis.stage1 import require_compute, write_json

PS1_ROOT = Path("/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3")
OUT = Path("manifests/cluster_rgb_ngc2419_ps1_check_v1.json")
COLS = ["OBJ_ID", "RA_OK", "DEC_OK", "MEDIAN", "NMAG_OK", "STDEV"]
COEFFS = {
    "g": [.00464, .08672, -.00668, -.00255],
    "r": [.00110, -.06875, .02480, -.00855],
    "z": [.03664, -.11084, .04477, -.01223],
}


def unit_vectors(ra,dec):
    ra,dec = np.deg2rad(ra),np.deg2rad(dec)
    return np.column_stack((np.cos(dec)*np.cos(ra),
                            np.cos(dec)*np.sin(ra),np.sin(dec)))


def summarize(values):
    x=np.asarray(values,float)
    x=x[np.isfinite(x)]
    if not len(x):
        return {"n":0}
    return {"n":len(x),"median":float(np.median(x)),
            "nmad":float(1.4826*np.median(np.abs(x-np.median(x)))),
            "p05":float(np.percentile(x,5)),"p95":float(np.percentile(x,95))}


def transformed_ps1(median):
    gi=median[:,0]-median[:,2]
    return {name:median[:,j]+np.polynomial.polynomial.polyval(gi,coef)
            for name,j,coef in (("g",0,COEFFS["g"]),("r",1,COEFFS["r"]),
                                ("z",3,COEFFS["z"]))}


def read_ps1():
    center=(114.535,38.882)
    pixels=hp.query_disc(32,hp.ang2vec(*center,lonlat=True),
                         np.deg2rad(1.2),inclusive=True,nest=False)
    parts=[]
    for pixel in pixels:
        path=PS1_ROOT/f"ps1-{pixel:05d}.fits"
        if not path.exists():
            continue
        rows=fitsio.read(str(path),ext=1,columns=COLS)
        inside=(np.abs(rows["RA_OK"]-center[0])<.9)&(np.abs(rows["DEC_OK"]-center[1])<.9)
        if inside.any():
            parts.append(rows[inside])
    if not parts:
        raise ValueError("No northern PS1 comparison stars")
    cat=np.concatenate(parts)
    _,first=np.unique(cat["OBJ_ID"],return_index=True)
    return cat[first],len(pixels)


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    table=fitsio.read("data/cluster_rgb_ngc2419_v1/members.fits")
    cfg=json.loads(CONFIG.read_text())["systems"]["ngc2419"]
    phase=classify(table["gaia_g"]-19.71,table["bp_rp"],cfg)
    target=(table["pmember"]>.9)&(phase=="upper_rgb")&table["grz_clean"]
    star=table[target]
    cat,npixels=read_ps1()
    tree=cKDTree(unit_vectors(cat["RA_OK"],cat["DEC_OK"]))
    nearest,idx=tree.query(unit_vectors(star["ra"],star["dec"]),k=2)
    sep=2*np.arcsin(np.minimum(nearest/2,1))*206264.806
    matched=(sep[:,0]<1)&(sep[:,1]>=1)
    _,counts=np.unique(idx[matched,0],return_counts=True)
    if np.any(counts>1):
        duplicated=np.unique(idx[matched,0],return_counts=True)[0][counts>1]
        matched &= ~np.isin(idx[:,0],duplicated)
    phot=cat["MEDIAN"][idx[:,0]].astype(float)
    stdev=cat["STDEV"][idx[:,0]].astype(float)
    nmag=cat["NMAG_OK"][idx[:,0]]
    gi=phot[:,0]-phot[:,2]
    quality=matched&(gi>.4)&(gi<2.7)
    for j in (0,1,2,3):
        quality &= (np.isfinite(phot[:,j])&(phot[:,j]>15)&(phot[:,j]<20)
                    &(nmag[:,j]>=3)&np.isfinite(stdev[:,j])
                    &(stdev[:,j]>0)&(stdev[:,j]<.03))
    pred=transformed_ps1(phot)
    delta={band:np.asarray(star[band],float)-pred[band] for band in "grz"}
    result={"system":"NGC 2419","natural_system":"BASS g/r and MzLS z",
            "ps1_transform":"Published DR11 PS1 g-i polynomial, calibrated for 0.4<g-i<2.7",
            "ps1_calibration_not_independent_zero_point":True,
            "queried_pixels":int(npixels),"catalog_sources":int(len(cat)),
            "upper_clean_targets":int(len(star)),
            "unique_match_within_1arcsec":int(matched.sum()),
            "quality_match":int(quality.sum()),
            "separation_arcsec":summarize(sep[quality,0]),
            "legacy_minus_ps1_predicted":{b:summarize(delta[b][quality]) for b in "grz"},
            "color_residuals":{
                "gr":summarize((delta["g"]-delta["r"])[quality]),
                "rz":summarize((delta["r"]-delta["z"])[quality])},
            "radius_quality":{},"compute_host":socket.gethostname()}
    if quality.any():
        radius=star["radius_arcmin"]
        med=float(np.median(radius[quality]))
        result["radius_quality"]={"split_arcmin":med,
            "inner_gr":summarize((delta["g"]-delta["r"])[quality&(radius<=med)]),
            "outer_gr":summarize((delta["g"]-delta["r"])[quality&(radius>med)])}
    write_json(OUT,result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
