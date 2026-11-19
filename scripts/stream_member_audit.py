#!/usr/bin/env python3
"""Complete physical audit columns for the original QRF-only member roster."""
from photdis.stage1 import require_compute
from stream_common import *
from sv18_common import index_join


def main():
    require_compute();d=folder('sequence');inputs={};table=list(csv.DictReader((d/'qrf_only_identity_audit.csv').open()))
    with np.load(d/'predictions.npz') as a:t={k:a[k] for k in ('ls_id','system','ref','sequence','new')}
    for name in ('M5','M13','M30','M92'):
        z=read(ROOT/f'data/v3_14/audit/{name}',('native_join.npz','TYPE','ls_id','g0','r0','z0','phase_box'),inputs);ref=float(np.median(t['ref'][t['system']==name]));mu=10+5*np.log10(ref)
        rows=[r for r in table if r['system']==name];j=index_join(np.array([int(r['ls_id']) for r in rows]),z['ls_id']);assert (j>=0).all()
        predicted=index_join(np.array([int(r['ls_id']) for r in rows]),t['ls_id'])
        for row,at,p in zip(rows,j,predicted):
            assert z['TYPE'][at]=='PSF';gr=float(z['g0'][at]-z['r0'][at]);mr=float(z['r0'][at]-mu)
            row.update(gr=gr,rz=float(z['r0'][at]-z['z0'][at]),reference_Mr=mr,reference_kpc=ref,member_reference_CMD_phase=z['phase_box'][at],
                outside_old_Mr3_8=not(3<=mr<=8),old_d50=float(t['sequence'][p,2]) if p>=0 else np.nan,new_d50=float(t['new'][p,2]) if p>=0 else np.nan,
                fixed_reference_phase='lower_MS' if .3<=gr<=.85 and 4.5<=mr<=8 else 'turnoff' if .15<=gr<.3 and 3<=mr<4.5 else 'other')
    archive=folder('execution')/'qrf_identity_before_physical_columns.csv'
    if not archive.exists():archive.write_text((d/'qrf_only_identity_audit.csv').read_text())
    csv_rows(d/'qrf_only_identity_audit.csv',table);r=json.loads((d/'complete.json').read_text());r['full_member_audit']=dict(code=sha(Path(__file__)),inputs=inputs,scope='all original QRF-only admitted members, including outside the old reference-Mr3..8 interval; reference CMD phase is not an independent spectroscopic label')
    r['outputs']={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'};save_json(d/'complete.json',r)


if __name__=='__main__':main()
