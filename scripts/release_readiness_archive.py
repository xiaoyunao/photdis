#!/usr/bin/env python3
"""Fetch exact public Gaia IDs with TAP and keep per-batch completion evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from astropy.table import Table
import numpy as np
import requests

PRODUCTS = {
    'bailer_jones': ('https://dc.g-vo.org/tap', 'gedr3dist.main', .001, {
        'BJ_geometric': ('r_lo_geo','r_med_geo','r_hi_geo'),
        'BJ_photogeometric': ('r_lo_photogeo','r_med_photogeo','r_hi_photogeo')}),
    'gsp_phot': ('https://gea.esac.esa.int/tap-server/tap', 'gaiadr3.astrophysical_parameters', .001, {
        'GSP_Phot': ('distance_gspphot_lower','distance_gspphot','distance_gspphot_upper')}),
    'starhorse': ('https://gaia.aip.de/tap', 'gaiaedr3_contrib.starhorse', 1., {
        'StarHorse': ('dist16','dist50','dist84')}),
}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def query_ids(ids, url, name, columns):
    upload = BytesIO()
    Table({'source_id':ids}).write(upload,format='votable',tabledata_format='binary2')
    select = ', '.join(f'a.{c}' for c in columns)
    query = f'SELECT u.source_id, {select} FROM TAP_UPLOAD.ids u JOIN {name} a ON u.source_id = a.source_id'
    payload = dict(REQUEST='doQuery',LANG='ADQL',FORMAT='votable',UPLOAD='ids,param:uids',
                   MAXREC=len(ids)+1,QUERY=query)
    attach = {'uids':('ids.xml',upload.getvalue(),'application/x-votable+xml')}
    for retry in range(3):
        try:
            reply = requests.post(url+'/sync',data=payload,files=attach,timeout=(25,180))
            reply.raise_for_status()
            root = ET.fromstring(reply.content)
            status = [x.attrib.get('value') for x in root.iter()
                      if x.tag.split('}')[-1]=='INFO' and x.attrib.get('name')=='QUERY_STATUS']
            if not status or any(k!='OK' for k in status):
                info = [x.text for x in root.iter() if x.attrib.get('name')=='QUERY_STATUS']
                raise ValueError(f'Incomplete TAP result: {status}, {info}')
            t = Table.read(BytesIO(reply.content),format='votable')
            return t, query, hashlib.sha256(reply.content).hexdigest()
        except (requests.RequestException, ET.ParseError, ValueError):
            if retry==2:
                raise
            time.sleep(2**(retry+1))


def run(product, ids_path, dest, batch_size, workers):
    ids = np.load(ids_path,allow_pickle=False)
    if ids.dtype!=np.int64 or not len(ids) or np.any(np.diff(ids)<=0):
        raise ValueError('Archive upload requires sorted distinct int64 identities')
    url, table, scale, fields = PRODUCTS[product]
    dest = dest/product
    dest.mkdir(exist_ok=True,parents=True)
    inputs = dict(ids_sha256=sha(ids_path),url=url,table=table,scale_to_kpc=scale,
                  fields={k:list(v) for k,v in fields.items()},batch_size=batch_size)
    columns = [c for cols in fields.values() for c in cols]

    def batch(args):
        i, interval = args
        out = dest/f'{i:03d}.npz'
        marker = out.with_suffix('.json')
        if marker.exists():
            record = json.loads(marker.read_text())
            if record['inputs']!=inputs or sha(out)!=record['sha256']:
                raise ValueError(f'Cached batch changed: {marker}')
            return record
        t, query, raw_sha = query_ids(interval,url,table,columns)
        returned = np.asarray(t['source_id'],dtype='i8')
        if len(np.unique(returned))!=len(t) or not np.isin(returned,interval).all():
            raise ValueError(f'Archive returned an invalid identity set in batch {i}')
        data = {k:np.column_stack([np.ma.asarray(t[c],dtype='f8').filled(np.nan)
                                   for c in cols])*scale for k,cols in fields.items()}
        temp = out.with_suffix('.tmp.npz')
        np.savez_compressed(temp,source_id=returned,**data)
        temp.replace(out)
        record = dict(input_rows=len(interval),matched=len(returned),first=int(interval[0]),last=int(interval[-1]),
                      payload_sha256=raw_sha,sha256=sha(out),inputs=inputs,query=query)
        marker.write_text(json.dumps(record,indent=2)+'\n')
        print(product,i,'matched',len(returned),'of',len(interval),flush=True)
        return record

    intervals = [ids[start:start+batch_size] for start in range(0,len(ids),batch_size)]
    with ThreadPoolExecutor(max_workers=workers) as p:
        batches = list(p.map(batch,enumerate(intervals)))
    parts = []
    for i in range(len(batches)):
        with np.load(dest/f'{i:03d}.npz',allow_pickle=False) as t:
            parts.append({k:t[k] for k in t.files})
    joined = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out = dest/'matched.npz'
    np.savez_compressed(out,**joined)
    marker = dict(inputs=inputs,rows=len(ids),matched=len(joined['source_id']),
                  outputs={out.name:sha(out)},batches={f'{i:03d}':sha(dest/f'{i:03d}.json') for i in range(len(batches))})
    (dest/'complete.json').write_text(json.dumps(marker,indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('product',choices=tuple(PRODUCTS))
    parser.add_argument('ids',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--batch-size',type=int,default=25000)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    run(args.product,args.ids,args.destination,args.batch_size,args.workers)
