#!/usr/bin/env python3
"""Validate public archives and extract inputs for GD-1 without changing tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ARCHIVES = {
    3: ('https://zenodo.org/records/19889980','29a2e81ba5423f56e5869b5661e9cb5a'),
    4: ('https://zenodo.org/records/23599156','c14f81e498de5208c40b0ab8b166d2b2'),
}
GAIA = ('https://doi.org/10.5281/zenodo.15428120','0b6038ef1adea3d4a796fda96a9a27fc')


def hash_file(path,algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()


def downloads(root,out):
    out.mkdir(parents=True,exist_ok=True)
    for v,(url,digest) in ARCHIVES.items():
        path = root/f'jarvis-v{v}.zip'
        if hash_file(path,'md5')!=digest:
            raise ValueError(f'Archive checksum mismatch for {url}')
        suffix = '' if v==3 else '_all'
        with ZipFile(path) as z:
            names = ('Table1','Table3','Table7'+suffix,'Table8'+suffix,
                     'Figure3_empirical_isochrone','Figure3_dotter_isochrone','Figure2_BHBs','Figure2_track_jarvis')
            for name in names:
                output = out/f'v{v}_{name}.fits'
                output.write_bytes(z.read(f'Jarvis_GD1_DESI_DR2_Zenodo/{name}.fits'))
    for table in ('Table1','Table3','Figure3_empirical_isochrone','Figure3_dotter_isochrone','Figure2_BHBs','Figure2_track_jarvis'):
        assert (out/f'v3_{table}.fits').read_bytes()==(out/f'v4_{table}.fits').read_bytes()
    gaia = root/'gaia.mrt'
    if gaia.is_file():
        if hash_file(gaia,'md5')!=GAIA[1]:
            raise ValueError('Gaia comparison archive checksum mismatch')
        dest = out/'tavangar_candidates.csv'
        rows = 0
        with gaia.open('rb') as src,dest.open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('source_id','ra','dec','phi1','phi2','p_membership','external_r0','external_color'))
            for l in src:
                if len(l.rstrip(b'\r\n'))<582 or not l[:19].strip().isdigit():
                    continue
                probability = float(l[559:582])
                if not .90<=probability<=1:
                    continue
                phi1 = float(l[266:290])
                if not -40.3<=phi1< -9.7:
                    continue
                g,r = float(l[201:222]),float(l[223:244])
                writer.writerow((int(l[:19]),float(l[20:38]),float(l[39:61]),
                                 phi1,float(l[291:315]),probability,r,g-r))
                rows += 1
        if rows==0:
            raise ValueError('Empty Gaia roster after explicit probability and longitude cuts')
        (out/'tavangar_source.json').write_text(json.dumps(
            dict(doi=GAIA[0],archive_md5=GAIA[1],candidate_sha256=hash_file(dest,'sha256'),rows=rows),
            indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('downloads',type=Path)
    parser.add_argument('out',type=Path)
    opts = parser.parse_args()
    downloads(opts.downloads,opts.out)
