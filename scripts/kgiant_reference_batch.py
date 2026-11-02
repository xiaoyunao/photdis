#!/usr/bin/env python3
"""Member-first, fixed-parent external inventory. Run only on allocated CPU."""
from pathlib import Path
import sys,json,zipfile,io,socket
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import fitsio
from photdis.holdouts import GC_ARCHIVE_ALIASES
from photdis.stage1 import require_compute,CORE_COLUMNS,PHOT_COLUMNS,tile_bounds,galactic
from photdis.photometry import derive_photometry
from photdis.audit import rvs_metallicity
from kgiant_distance_inventory import sha,write_json
from giant_ngc2419_reference import MWS

OUT=Path('data/kgiant_reference_batch_v1')
REF=Path('manifests/kgiant_bv21_reference_v1.json')
BANDS=('g0','r0','z0','w10');ERRORS=('sigma_g','sigma_r','sigma_z','sigma_w1')

def main():
    require_compute();OUT.mkdir(exist_ok=True)
    if (OUT/'inventory.json').exists():raise FileExistsError(OUT/'inventory.json')
    ref=json.load(open(REF));systems=ref['membership_mapping'];distance={r['name']:r for r in ref['rows']}
    archive=Path('data/reference/vasiliev_baumgardt_2021_members.zip')
    ids=[];groups=[];prob=[];coords=[];counts={}
    with zipfile.ZipFile(archive) as z:
        for c in systems:
            file='catalogues/'+GC_ARCHIVE_ALIASES.get(c['member_name'],c['member_name'])+'.txt';n=0;core=0
            with z.open(file) as f:
                for line in io.TextIOWrapper(f):
                    if not line.strip() or line.startswith('#'):continue
                    v=line.split();p=float(v[-1])
                    if p>=.5:
                        ids.append(int(v[0]));groups.append(c['group']);prob.append(p);coords.append((float(v[1]),float(v[2])));n+=1;core+=p>.9
            counts[c['member_name']]={'probable_members':n,'core_members':core,'reference_name':c['reference_name']}
    ids=np.array(ids,dtype=np.int64);groups=np.array(groups);prob=np.array(prob);coords=np.array(coords)
    order=np.argsort(ids);ids,groups,prob,coords=ids[order],groups[order],prob[order],coords[order]
    unique=np.r_[True,ids[1:]!=ids[:-1]]&np.r_[ids[1:]!=ids[:-1],True]
    ambiguous=ids[~unique];ids,groups,prob,coords=ids[unique],groups[unique],prob[unique],coords[unique]
    np.savez_compressed(OUT/'all_reference_member_ids.npz',source_id=ids,ambiguous_ids=ambiguous)
    names={r['group']:r for r in systems}
    rv=fitsio.read(str(MWS),ext='RVTAB',columns=['TARGETID','PRIMARY','RR_SPECTYPE','RVS_WARN','SN_R','VSINI','TEFF','TEFF_ERR','LOGG','LOGG_ERR','SUCCESS','FEH','FEH_ERR'])
    ga=fitsio.read(str(MWS),ext='GAIA',columns=['SOURCE_ID'])
    assert len(rv)==len(ga)
    pos=np.minimum(np.searchsorted(ids,ga['SOURCE_ID']),len(ids)-1);member=(ids[pos]==ga['SOURCE_ID'])&(ga['SOURCE_ID']>0)
    good=(member&rv['PRIMARY']&(np.char.strip(rv['RR_SPECTYPE'].astype('U'))=='STAR')&(rv['RVS_WARN']==0)&rv['SUCCESS']&
          np.isfinite(rv['SN_R'])&(rv['SN_R']>=10)&np.isfinite(rv['VSINI'])&(rv['VSINI']>=0)&(rv['VSINI']<30)&
          np.isfinite(rv['TEFF'])&(rv['TEFF']>0)&(rv['TEFF']<7000)&np.isfinite(rv['TEFF_ERR'])&(rv['TEFF_ERR']>0)&
          np.isfinite(rv['LOGG'])&np.isfinite(rv['LOGG_ERR'])&(rv['LOGG_ERR']>0)&(rv['LOGG']+rv['LOGG_ERR']<3.2))
    k=good&(rv['TEFF']>=4200)&(rv['TEFF']<5000)
    candidates=[]
    for sid in np.unique(ga['SOURCE_ID'][k]):
        ii=np.flatnonzero(k&(ga['SOURCE_ID']==sid));i=ii[np.argmax(rv['SN_R'][ii])];j=pos[i];c=names[int(groups[j])]
        metal,valid=rvs_metallicity(np.array([rv['FEH'][i]]),np.array([rv['TEFF'][i]]),np.array([rv['LOGG'][i]]))
        candidates.append(dict(name=str(int(sid)),source_id=str(int(sid)),member_system=c['member_name'],reference_name=c['reference_name'],
          pmember=float(prob[j]),ra=float(coords[j,0]),dec=float(coords[j,1]),type_source='DESI DR1 MWS RVS',
          targetid=str(int(rv['TARGETID'][i])),teff=float(rv['TEFF'][i]),teff_error=float(rv['TEFF_ERR'][i]),logg=float(rv['LOGG'][i]),
          snr=float(rv['SN_R'][i]),feh_raw=float(rv['FEH'][i]),feh_error=float(rv['FEH_ERR'][i]),feh_corrected=float(metal[0]),
          abundance_calibration_valid=bool(valid[0]),phase='unknown',n_qualified_spectra=len(ii),historical_diagnostic=False))
    for c in systems:
        g=c['group'];ix=member&(groups[pos]==g)
        counts[c['member_name']].update(mws_member_spectra=int(ix.sum()),reliable_giant_spectra=int((ix&good).sum()),K_spectra=int((ix&k).sum()),
             unique_K_members=sum(r['member_system']==c['member_name'] for r in candidates))
    # Historical literature identities are separate and already unblinded.
    old=json.load(open('manifests/kgiant_external_inventory_v1.json'))
    for r in old['rows']:
        if r['cluster']=='Terzan7' and r['identity_status']=='unique_position_match' and r['nominal_K']:
            assert r['source_id'] not in {x['source_id'] for x in candidates}
            candidates.append(dict(name=r['name'],source_id=r['source_id'],member_system='Terzan_7',reference_name='Ter 7',pmember=r['pmember'],
              ra=r['matched_ra'],dec=r['matched_dec'],teff=r['teff'],type_source='Sbordone2005 B-V temperature; low gravity/RV',
              phase='unknown',historical_diagnostic=True,measurements=[dict(release=r['release'],grz_quality=r['grz_quality'],
              w1_quality=r['w1_quality'],ebv=r['ebv'],photometry=r['photometry'],quality_raw=r['quality_raw'])]))
    write_json(OUT/'typed_candidates.json',dict(rows=candidates,systems=counts,ambiguous_member_ids=len(ambiguous),
       source_sha256={str(p):sha(p) for p in [REF,archive,Path(__file__)]},mws_path=str(MWS),mws_size=MWS.stat().st_size))
    print('Typed candidates',len(candidates),'; new',sum(not r['historical_diagnostic'] for r in candidates),flush=True)
    # Directed reads of Gaia-ID columns, then only matching rows. No skywide flux scan.
    new=[r for r in candidates if not r['historical_diagnostic'] and r['reference_name']]
    sweep_provenance=[]
    for region in ('south','north'):
        root=Path('/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11')/region/'sweep/11.0'
        for path in sorted(root.glob('sweep-*.fits')):
            r0,d0,r1,d1=tile_bounds(path)
            targets=[r for r in new if r0-.01<=r['ra']<r1+.01 and d0-.01<=r['dec']<d1+.01]
            if not targets:continue
            lookup={int(r['source_id']):r for r in targets}
            with fitsio.FITS(str(path)) as hdus:
                h=hdus[1];cols=h.get_colnames();small=h.read(columns=['REF_ID','REF_CAT'])
                ix=np.flatnonzero(np.isin(small['REF_ID'],list(lookup))&(np.char.strip(small['REF_CAT'].astype('U'))=='G3'))
                primary=['BRICK_PRIMARY'] if 'BRICK_PRIMARY' in cols else []
                if len(ix):
                    raw=h.read(rows=ix,columns=CORE_COLUMNS+PHOT_COLUMNS+primary);ph=derive_photometry(raw)
                    for i,v in enumerate(raw):
                        if primary and not v['BRICK_PRIMARY']:continue
                        r=lookup[int(v['REF_ID'])]
                        r.setdefault('measurements',[]).append(dict(release=int(v['RELEASE']),brickid=int(v['BRICKID']),objid=int(v['OBJID']),
                          grz_quality=bool(ph['grz_clean'][i] and str(v['TYPE']).strip()=='PSF'),
                          w1_quality=bool(ph['w1_clean'][i] and ph['snr_w1'][i]>=5),ebv=float(v['EBV']),
                          ra=float(v['RA']),dec=float(v['DEC']),photometry={k:float(ph[k][i]) for k in BANDS+ERRORS+('g','r','z','w1','snr_w1')},
                          quality_raw={k:np.asarray(v[k]).tolist() for k in ['MASKBITS']+[f'{k}_{b}' for k in ('RCHISQ','FRACMASKED') for b in ('G','R','Z')]}))
            sweep_provenance.append(dict(path=str(path),size=path.stat().st_size,matched_rows=len(ix)))
            print(region,path.name,'matches',len(ix),flush=True)
    for r in candidates:
        r['galactic_b']=float(galactic(r['ra'],r['dec'])[1][0]);r['old_latitude_scope']=abs(r['galactic_b'])>30
        measurements=r.get('measurements',[])
        # Remove exact repeats from overlap tile reads, retain any true ambiguity.
        seen={}
        for m in measurements:seen[(m['release'],m.get('brickid'),m.get('objid'))]=m
        r['measurements']=list(seen.values());south=[m for m in r['measurements'] if m['release']==11010]
        r['unique_decam']=len(south)==1;r['eligible']=False;r['reasons']=[]
        if not r['reference_name']:r['reasons'].append('no_published_distance')
        if r['pmember']<=.9:r['reasons'].append('secondary_membership_only')
        if len(south)!=1:r['reasons'].append('no_unique_DECam_measurement')
        if len(south)==1:
            m=south[0];r['selected_measurement']=m;r['old_dust_scope']=0<=m['ebv']<.03
            if not m['grz_quality']:r['reasons'].append('fixed_grz_quality_failed')
            if r['reference_name']:r['Mg_ref']=m['photometry']['g0']-distance[r['reference_name']]['mu']
        r['eligible']=not r['reasons']
    inventory=dict(rows=candidates,systems=counts,sweeps=sweep_provenance,
        reference_sha256=sha(REF),typed_candidates_sha256=sha(OUT/'typed_candidates.json'),
        n_scored_systems_design_aim=3,host=socket.gethostname(),phase='inventory_frozen_before_prediction')
    write_json(OUT/'inventory.json',inventory)
    print('FROZEN',len(candidates),'eligible',sum(r['eligible'] for r in candidates),flush=True)
if __name__=='__main__':main()
