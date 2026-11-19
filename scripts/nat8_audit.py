#!/usr/bin/env python3
"""Exactly three existing tail identities; at most two no-prediction inventories."""
from nat8_common import *
from photdis.stage1 import require_compute,legacy_ids,tile_bounds
from photdis.holdouts import GC_ARCHIVE_ALIASES
from v3_1_external_history import HISTORY,identity_position
from v3_9_catalog import settings,read_native,scan_chunk
from photdis.reference_v1_4 import unique_join
from v3_12_reference import separation
from scipy.spatial import cKDTree
import fitsio,zipfile,io,hashlib


def members(name,archive):
    target='catalogues/'+GC_ARCHIVE_ALIASES.get(name,name)+'.txt'
    if target not in archive.namelist():
        matches=[n for n in archive.namelist() if n.startswith('catalogues/'+name+'_') and n.endswith('.txt')];assert len(matches)==1,(name,matches);target=matches[0]
    with archive.open(target) as f:
        text=f.read().decode()
    header=[l for l in text.splitlines() if l.startswith('#')];data=[]
    for line in text.splitlines():
        if line.strip() and not line.startswith('#'):data.append(line.split())
    return header,data


def raw_tail(t,indices,output):
    cfg=settings()['north'];records={};wanted=['TYPE','RELEASE','BRICKID','OBJID','RA','DEC','REF_CAT','REF_ID','REF_EPOCH','PMRA','PMRA_IVAR','PMDEC','PMDEC_IVAR','PARALLAX','PARALLAX_IVAR','MASKBITS','FITBITS','GAIA_PHOT_G_MEAN_MAG','GAIA_PHOT_BP_MEAN_MAG','GAIA_PHOT_RP_MEAN_MAG','GAIA_ASTROMETRIC_EXCESS_NOISE','GAIA_ASTROMETRIC_PARAMS_SOLVED','GAIA_DUPLICATED_SOURCE','GAIA_RADIAL_VELOCITY','GAIA_RADIAL_VELOCITY_ERROR']
    wanted += [prefix+'_'+b for prefix in ('FLUX','FLUX_IVAR','MW_TRANSMISSION','FRACFLUX','FRACMASKED','FRACIN','ANYMASK','NOBS') for b in 'GRZ']
    for path in sorted(Path(cfg['sweep_root']).glob('sweep-*.fits')):
        r0,d0,r1,d1=tile_bounds(path);take=[i for i in indices if r0<=t['RA'][i]<=r1 and d0<=t['DEC'][i]<=d1]
        if not take:continue
        with fitsio.FITS(str(path)) as f:
            names=f[1].get_colnames();cols=[k for k in wanted if k in names];a=f[1].read(columns=cols)
        ids=legacy_ids(a)
        for i in take:
            j=np.flatnonzero(ids==t['ls_id'][i]);assert len(j)==1,(path,i,j);j=int(j[0]);assert str(a['TYPE'][j].decode() if a['TYPE'].dtype.kind=='S' else a['TYPE'][j]).strip()=='PSF'
            values={}
            for k in cols:
                value=a[k][j];value=value.item() if isinstance(value,np.generic) else value;values[k]=value.decode().strip() if isinstance(value,bytes) else value
            records[int(t['ls_id'][i])]=dict(path=str(path),available=cols,missing=[k for k in wanted if k not in names],values=values)
    assert len(records)==3;save_json(output,records);return records


def inventory_candidate(name,public,center,hist_ids,history_summary):
    d=folder('candidate_inventory/'+name);cfg=dict(settings()['south']);cfg['maximum_grz_error_mag']=settings()['maximum_grz_error_mag'];cfg['domains']={'cluster':dict(r_max=23.,abs_b_min=30.,ebv_max=.08)};ids=np.array([int(r[0]) for r in public if float(r[-1])>.9],'i8');photos=[];counts=[]
    for path in sorted(Path(cfg['sweep_root']).glob('sweep-*.fits')):
        r0,d0,r1,d1=tile_bounds(path)
        if not(r0<center[0]+.5 and r1>center[0]-.5 and d0<center[1]+.5 and d1>center[1]-.5):continue
        raw=read_native(path);near=separation(raw['RA'],raw['DEC'],center)<.5;raw=raw[near]
        if not len(raw):continue
        member=np.isin(raw['REF_ID'],ids)&(np.char.strip(raw['REF_CAT'].astype('U'))=='G3');type_psf=np.char.strip(raw['TYPE'].astype('U'))=='PSF';counts.append(dict(sweep=path.name,raw=len(raw),raw_member=int(member.sum()),member_nonPSF=int((member&~type_psf).sum()),member_missing_TYPE=int((member&(np.char.strip(raw['TYPE'].astype('U'))=='')).sum())))
        for photo,code in scan_chunk(raw,(r0,d0,r1,d1),cfg,0):
            if not photo:continue
            use=photo['row_state']==7
            if not use.any():continue
            p={k:photo[k][use] for k in ('TYPE','ls_id','source_id','RA','DEC','r','g0','r0','z0','mag_grz','error_grz','identity','EBV')};j=unique_join(p['ls_id'],legacy_ids(raw));assert(j>=0).all();flags=raw[j];p['max_fracflux']=np.maximum.reduce([flags['FRACFLUX_'+b] for b in 'GRZ']);p['mask']=(flags['MASKBITS'].astype('i8')&(1<<13))!=0;photos.append(p)
    historical=history_summary[name];counts_row=dict(system=name,published_members=int(len(ids)),history_member_ID_exposures=int(sum(hist_ids[name])),prior_system='availability_only_empty' if name=='NGC1851' else 'proposed_no_scored_system_found',forecast_opened=False)
    if photos:
        a={k:np.concatenate([p[k] for p in photos]) for k in photos[0]};assert len(np.unique(a['ls_id']))==len(a['ls_id']) and (a['TYPE']=='PSF').all();field_N=len(a['ls_id']);ra,de=np.deg2rad([a['RA'],a['DEC']]);vec=np.column_stack((np.cos(ra)*np.cos(de),np.sin(ra)*np.cos(de),np.sin(de)));nearest=2*np.arcsin(np.clip(cKDTree(vec).query(vec,k=2)[0][:,1]/2,0,1))*180/np.pi*3600
        member=np.isin(a['source_id'],ids);nearest=nearest[member];a={k:v[member] for k,v in a.items()};radius=separation(a['RA'],a['DEC'],center)
        low=(a['max_fracflux']<.1)&(nearest>=2);target=(radius>=.08)&(radius<.35);seen=np.isin(a['source_id'],np.array(sorted(historical),dtype='i8'));keep=target&low&~a['mask']&~seen;gr=a['g0']-a['r0'];npz(d/'photometry_inventory.npz',**a,low_crowding=low,target=target,seen=seen,inventory_eligible=keep,nearest_arcsec=nearest)
        counts_row.update(qualified_field_PSF=field_N,qualified_native_PSF=len(gr),annulus_member=int(target.sum()),unmasked_low_crowding=int((target&low&~a['mask']).sum()),source_history_excluded=int((target&low&~a['mask']&seen).sum()),inventory_eligible=int(keep.sum()),gr_min=float(gr[keep].min()) if keep.any() else None,gr_max=float(gr[keep].max()) if keep.any() else None,r_min=float(a['r'][keep].min()) if keep.any() else None,r_max=float(a['r'][keep].max()) if keep.any() else None)
        csv_rows(d/'identities.csv',[dict(TYPE=a['TYPE'][i],ls_id=int(a['ls_id'][i]),source_id=int(a['source_id'][i]),eligible=bool(keep[i]),seen=bool(seen[i]),native='DECam') for i in range(len(gr))])
    else:counts_row.update(qualified_native_PSF=0,inventory_eligible=0)
    csv_rows(d/'admission.csv',counts);save_json(d/'inventory.json',counts_row);print('Candidate inventory',counts_row,flush=True);return counts_row


def main():
    require_compute();d=folder('audit');inputs={'prepare':sha(OUT/'prepare/complete.json')};t=load(OUT/'prepare/roster.npz');selected=list(csv.DictReader((OLD/'examples/selected.csv').open()));tail=[r for r in selected if r['kind']=='QRF_decline'];assert len(tail)==3;indices=[int(np.flatnonzero(t['ls_id']==int(r['ls_id']))[0]) for r in tail];raw=raw_tail(t,indices,d/'three_raw_imaging.json');archive_path=ROOT/'data/reference/vasiliev_baumgardt_2021_members.zip';assert hashlib.md5(archive_path.read_bytes()).hexdigest()=='3a07093be9a1fcbfb0e3c545ca175c99';inputs['membership_zip']=sha(archive_path)
    centers=np.genfromtxt(ROOT/'data/reference/vasiliev_baumgardt_2021_clusters.txt',names=True,dtype=None,encoding='utf8');center_lookup={str(r['Name']):r for r in centers};candidates={};cluster_ids={};history_summary={};history_rows=[];member_detail={}
    with zipfile.ZipFile(archive_path) as archive:
        header,rows=members('NGC_6205',archive);lookup={int(r[0]):r for r in rows};save_json(d/'membership_columns.json',dict(header=header,example_column_count=len(rows[0]),source='Vasiliev/Baumgardt2021v2'))
        for i in indices:
            row=lookup[int(t['source_id'][i])];member_detail[str(int(t['ls_id'][i]))]=dict(TYPE=t['TYPE'][i],source_id=int(t['source_id'][i]),published_probability=float(row[-1]),raw_membership_columns=row,position_offset_arcsec=float(separation(np.array([t['RA'][i]]),np.array([t['DEC'][i]]),(float(row[1]),float(row[2])))[0]*3600))
        for name,key in [('NGC1851','NGC_1851'),('NGC5053','NGC_5053')]:
            header,rows=members(key,archive);candidates[name]=(rows,center_lookup[key]);cluster_ids[name]=np.array([int(r[0]) for r in rows if float(r[-1])>.9],'i8');history_summary[name]=set()
    save_json(d/'three_membership.json',member_detail);seen={k:np.zeros(len(v),bool) for k,v in cluster_ids.items()}
    paths=list(HISTORY)+[('data/v3_1/common_scale_native/photometry.npz','v3.1 opened native common-scale cohort'),('data/v3_2/common_scale_inventory/photometry.npz','v3.2 opened common scales')]
    for rel,lineage in paths:
        p=ROOT/rel
        if not p.exists():history_rows.append(dict(path=rel,lineage=lineage,missing=True));continue
        try:oldids,_=identity_position(p)
        except KeyError:
            z=load(p);oldids=z.get('source_id',z.get('identity',np.array([],dtype='i8')))
        positive=np.unique(oldids[oldids>0]);r=dict(path=rel,lineage=lineage,N=len(oldids),identity_sha256=hashlib.sha256(oldids.tobytes()).hexdigest())
        for name,ids in cluster_ids.items():
            match=np.isin(ids,positive);seen[name]|=match;history_summary[name].update(int(v) for v in ids[match]);r[name]=int(match.sum())
        history_rows.append(r)
    # Current north/south training and calibration identities augment old lineages.
    from v3_13_clusters import model_and_history
    for side in ('south','north'):
        _,_,ids=model_and_history(side,inputs);r=dict(path='frozen_current_'+side,lineage='direct fit/calibration IDs',N=len(ids))
        for name,memberids in cluster_ids.items():
            match=np.isin(memberids,ids);seen[name]|=match;history_summary[name].update(int(v) for v in memberids[match]);r[name]=int(match.sum())
        history_rows.append(r)
    csv_rows(d/'candidate_history.csv',history_rows);inventories=[]
    for name,(rows,center) in candidates.items():inventories.append(inventory_candidate(name,rows,(float(center['ra']),float(center['dec'])),seen,history_summary))
    csv_rows(d/'candidate_inventory.csv',inventories);complete(d,inputs,fixed_tail_sources=3,candidate_count=2,no_new_candidate_distance_predictions=True,membership_probability_is_astrometric_not_independent_distance_truth=True);print('Audits complete',flush=True)


if __name__=='__main__':main()
