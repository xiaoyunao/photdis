#!/usr/bin/env python3
"""Complete physical audit columns for the original QRF-only member roster."""
from photdis.stage1 import require_compute
from stream_common import *
from sv18_common import index_join,cached


def main():
    require_compute();d=folder('sequence');inputs={};table=[r for r in csv.DictReader((d/'qrf_only_identity_audit.csv').open()) if not r['system'].startswith('GD1')]
    with np.load(d/'predictions.npz') as a:t={k:a[k] for k in ('ls_id','system','ref','sequence','new')}
    for name in ('M5','M13','M30','M92'):
        z=read(ROOT/f'data/v3_14/audit/{name}',('native_join.npz','TYPE','ls_id','g0','r0','z0','phase_box'),inputs);ref=float(np.median(t['ref'][t['system']==name]));mu=10+5*np.log10(ref)
        rows=[r for r in table if r['system']==name];j=index_join(np.array([int(r['ls_id']) for r in rows]),z['ls_id']);assert (j>=0).all()
        predicted=index_join(np.array([int(r['ls_id']) for r in rows]),t['ls_id'])
        for row,at,p in zip(rows,j,predicted):
            assert z['TYPE'][at]=='PSF';gr=float(z['g0'][at]-z['r0'][at]);mr=float(z['r0'][at]-mu)
            row.update(gr=gr,rz=float(z['r0'][at]-z['z0'][at]),reference_Mr=mr,reference_kpc=ref,member_reference_CMD_phase=z['phase_box'][at],
                outside_old_Mr3_8=not(3<=mr<=8),old_d50=float(t['sequence'][p,2]) if p>=0 else np.nan,new_d50=float(t['new'][p,2]) if p>=0 else np.nan,
                native=int(name in ('M13','M92')),fixed_reference_phase='lower_MS' if .3<=gr<=.85 and 4.5<=mr<=8 else 'turnoff' if .15<=gr<.3 and 3<=mr<4.5 else 'other')
    member=read(ROOT/'data/v3_18/members',('posteriors.npz','TYPE','ls_id','science','quantiles_kpc'),inputs)
    roster={int(r['ls_id']):r for r in csv.DictReader((ROOT/'data/v3_18/members/roster.csv').open())}
    refcache=cached(inputs,('TYPE','ls_id','source_id','m','region'));use=member['science']&np.isfinite(member['quantiles_kpc']).all(axis=1)
    ii=np.flatnonzero(use);jj=index_join(member['ls_id'][ii],refcache['ls_id']);assert (jj>=0).all()
    with np.load(d/'predictions.npz') as a:prior={k:a[k] for k in ('ls_id','common','new_common','sequence','new')}
    pp=index_join(member['ls_id'][ii],prior['ls_id'])
    for i,j,p in zip(ii,jj,pp):
        assert member['TYPE'][i]=='PSF' and refcache['TYPE'][j]=='PSF';rr=roster[int(member['ls_id'][i])];ref=float(rr['desi_kpc']);mag=refcache['m'][j];gr=float(mag[0]-mag[1]);mr=float(mag[1]-10-5*np.log10(ref))
        table.append(dict(TYPE='PSF',system='GD1_'+rr['group'],ls_id=int(member['ls_id'][i]),source_id=int(refcache['source_id'][j]),r=float(rr['r']),mask=False,
            old_sequence_parent=bool(p>=0),old_common=bool(p>=0 and prior['common'][p]),new_common=bool(p>=0 and prior['new_common'][p]),native=int(refcache['region'][j]==3),
            gr=gr,rz=float(mag[1]-mag[2]),reference_Mr=mr,reference_kpc=ref,member_reference_CMD_phase='MS/TO roster; individual phase unmeasured',outside_old_Mr3_8=not(3<=mr<=8),
            old_d50=float(prior['sequence'][p,2]) if p>=0 else np.nan,new_d50=float(prior['new'][p,2]) if p>=0 else np.nan,
            fixed_reference_phase='lower_MS' if .3<=gr<=.85 and 4.5<=mr<=8 else 'turnoff' if .15<=gr<.3 and 3<=mr<4.5 else 'other'))
    del refcache
    assert len({int(r['ls_id']) for r in table})==len(table)
    cross={}
    for row in table:
        key=(row['system'],int(np.digitize(float(row['r']),[14,18,20,21,22,23])),str(row['member_reference_CMD_phase']),str(row['fixed_reference_phase']),str(row['mask']),int(row['native']),str(row['old_common']),str(row['new_common']))
        cross[key]=cross.get(key,0)+1
    csv_rows(d/'full_member_cross_counts.csv',[dict(zip(('system','r_bin_index','member_reference_CMD_phase','fixed_reference_phase','mask','native','old_common','new_common'),key),n=n) for key,n in cross.items()])
    old=list(csv.DictReader((OLD/'sequence/sequence_scale.csv').open()));current=list(csv.DictReader((d/'distance_scale.csv').open()));matched=[]
    for row in old:
        if row['r_lo']=='20' and row['r_hi']=='21' and row['quality']=='unmasked_low_crowding':matched.append(dict(cohort='frozen_v4_three_method',system=row['system'],method=row['method'],n=row['n'],bias=row['bias'],nmad=row['nmad']))
    for row in current:
        if row['r_lo']=='20' and row['r_hi']=='21' and row['phase']=='all' and row['include_masked']=='False':matched.append(dict(cohort='v5_same_current_common_IDs',system=row['system'],method=row['method'],n=row['n'],bias=row['bias'],nmad=row['nmad']))
    csv_rows(d/'original_and_current_r20_21.csv',matched)
    archive=folder('execution')/'qrf_identity_before_physical_columns.csv'
    if not archive.exists():archive.write_text((d/'qrf_only_identity_audit.csv').read_text())
    csv_rows(d/'qrf_only_identity_audit.csv',table);r=json.loads((d/'complete.json').read_text());r['full_member_audit']=dict(code=sha(Path(__file__)),inputs=inputs,scope='all original QRF-only admitted members, including outside the old reference-Mr3..8 interval; reference CMD phase is not an independent spectroscopic label')
    r['outputs']={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'};save_json(d/'complete.json',r)


if __name__=='__main__':main()
