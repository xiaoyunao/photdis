import json
import numpy as np
import pytest
from photdis.v3_1.freeze import Inventory, file_hash, verify_prediction_inventory, verify_freeze


def test_freeze_authenticates_nested_outputs_and_detects_later_tamper(tmp_path):
    folder = tmp_path/'data'; folder.mkdir()
    value = folder/'pdf.bin'; value.write_bytes(b'full tails')
    obj = folder/'object.json'
    obj.write_text(json.dumps(dict(outputs={'pdf.bin':file_hash(value)})))
    marker = folder/'complete.json'
    marker.write_text(json.dumps(dict(objects={'object.json':file_hash(obj)})))
    inv = Inventory(tmp_path); inv.marker(marker)
    assert len(inv.hashes) == 3
    freeze = tmp_path/'freeze.json'
    freeze.write_text(json.dumps(dict(predictions_frozen=True,outcomes_opened=False,hashes=inv.hashes)))
    assert verify_freeze(tmp_path,freeze) == file_hash(freeze)
    value.write_bytes(b'changed tail')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_freeze(tmp_path,freeze)


def test_fixed_inventory_cannot_replace_missing_or_failed_object():
    ids = np.array([9007199254740993,9007199254740995],dtype='i8')
    record = dict(n=2,references_read=False,final_outcomes_opened=False,numerical_passed=False,
                  objects={f'object_{i}/complete.json':'hash' for i in ids})
    assert verify_prediction_inventory(record,ids,family='candidate') is False
    record['pilot'] = True
    with pytest.raises(ValueError,match='Pilot'):
        verify_prediction_inventory(record,ids,family='candidate')
    record['pilot'] = False
    record['objects'].pop(f'object_{ids[0]}/complete.json')
    with pytest.raises(ValueError,match='inventory'):
        verify_prediction_inventory(record,ids,family='candidate')


def test_freeze_rejects_outside_checkout_and_inexact_identifiers(tmp_path):
    with pytest.raises(ValueError):
        Inventory(tmp_path).add('../outside')
    with pytest.raises(ValueError,match='integer identities'):
        verify_prediction_inventory({},np.array([1.,2.]),family='candidate')
