#!/usr/bin/env python3
"""Label historical roles and scope without altering the raw observational census."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=Path(sys.argv[1]);d=json.loads(source.read_text())
    original=json.loads(source.read_text())
    roles={'NGC_5272':'historical_RGB_training','NGC_5024':'historical_RGB_training',
           'NGC_7006':'historical_test_then_development','NGC_5466':'historical_unblinded_test',
           'NGC_4147':'historical_unblinded_test','NGC_2419':'historical_north_RGB_audit'}
    for r in d['rows']:
        key='_'.join(r['member_system'].split('_')[:2])
        r['historical_role']=roles.get(key,'first_in_this_DESI_coverage_audit_not_claimed_globally_untouched')
        r['optical_reference_qualified']=r.pop('eligible')
        r['scoring_status']='not_scored_in_this_audit'
        r['original_K_prediction_exists']=r['source_id']=='4464703441466064512'
    for new,old in zip(d['rows'],original['rows']):
        assert all(new[k]==v for k,v in old.items() if k not in ('historical_role','eligible'))
        assert new['optical_reference_qualified']==old['eligible']
    d['metadata_revision']=dict(all_original_observational_fields_unchanged=True,raw_inventory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changes='Resolve M3/M53 membership-name aliases for historical roles; clarify optical-only flag is not K-model scoring eligibility; observations unchanged')
    (ROOT/'manifests/giant_parameter_inventory_v1.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
