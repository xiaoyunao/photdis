"""Resume completed radial quadrature rows under the same input identity."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp
import os

from v2_1_near import atomic_npz, load, save_json, sha


def compute_rows(folder, metadata, identities, worker, *, workers, keys):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = folder/'inputs.json'
        if spec.exists():
            if json.loads(spec.read_text()) != metadata:
                raise ValueError(f'Radial checkpoint input identity changed: {folder}')
        else:
            save_json(spec, metadata)
        outputs = [None]*len(identities)
        missing = []
        for row, identity in enumerate(identities):
            payload = folder/f'row_{row:03}.npz'
            receipt = payload.with_suffix('.json')
            if not receipt.exists():
                missing.append(row)
                continue
            record = json.loads(receipt.read_text())
            if record['identity'] != int(identity) or record['TYPE'] != 'PSF' or record['row'] != row:
                raise ValueError(f'Checkpoint source row changed: {receipt}')
            if sha(payload) != record['sha256']:
                raise ValueError(f'Checkpoint payload changed: {payload}')
            data = load(payload)
            outputs[row] = tuple(data[key] for key in keys)
        print(f'RADIAL ROWS {len(identities)-len(missing)}/{len(identities)} verified', flush=True)
        if missing:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
                # Receive rows as soon as possible even if the first row is
                # slower than subsequent contexts. Keep outstanding work
                # bounded by the small frozen catalog size.
                pending = {pool.submit(worker, row): row for row in missing}
                for i, future in enumerate(as_completed(pending), 1):
                    row = pending[future]
                    result = future.result()
                    if not isinstance(result, tuple):
                        result = (result,)
                    if len(result) != len(keys):
                        raise ValueError('Unexpected checkpoint result keys')
                    payload = folder/f'row_{row:03}.npz'
                    atomic_npz(payload, **dict(zip(keys, result)))
                    save_json(payload.with_suffix('.json'), dict(identity=int(identities[row]),
                              TYPE='PSF', row=row, sha256=sha(payload), job=os.getenv('SLURM_JOB_ID')))
                    outputs[row] = result
                    if i % 8 == 0 or i == len(missing):
                        print(f'RADIAL ROWS {len(identities)-len(missing)+i}/{len(identities)} verified', flush=True)
    return outputs
