"""Auditable file and identity closure before opening prospective outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Inventory:
    """Authenticate declared artifacts without reading reference values."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}
        self.markers = set()
        self.stamps = {}

    def add(self, path, expected=None):
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        # Frozen validation products and programs are owned by this checkout.
        name = str(path.relative_to(self.root))
        before = path.stat()
        stamp = (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        if name in self.hashes:
            if stamp != self.stamps[name]:
                raise ValueError('Artifact changed during inventory: ' + name)
            if expected is not None and expected != self.hashes[name]:
                raise ValueError('Artifact hash mismatch: ' + name)
            return path
        digest = file_hash(path)
        after = path.stat()
        if stamp != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Artifact changed while hashing: ' + name)
        if expected is not None and digest != expected:
            raise ValueError('Artifact hash mismatch: ' + name)
        if name in self.hashes and self.hashes[name] != digest:
            raise ValueError('Artifact changed during inventory: ' + name)
        self.hashes[name] = digest
        self.stamps[name] = stamp
        return path

    def unchanged(self):
        for name in self.hashes:
            self.add(name)

    def marker(self, path, expected=None):
        path = self.add(path, expected)
        record = json.loads(path.read_text())
        if path in self.markers:
            return record
        self.markers.add(path)
        for name, digest in record.get('inputs', {}).items():
            if name.startswith(('data/', 'src/', 'scripts/', 'docs/', 'configs/', 'manifests/')):
                self.add(name, digest)
        for name, digest in record.get('outputs', {}).items():
            self.add(path.parent / name, digest)
        for name, digest in record.get('objects', {}).items():
            self.marker(path.parent / name, digest)
        return record


def verify_prediction_inventory(record, identities, *, family):
    """Full fixed-identity inventory, including failed numerical outcomes."""
    ids = np.asarray(identities)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique exact integer identities required')
    if record.get('references_read') is not False or record.get('final_outcomes_opened') is not False:
        raise ValueError('Prediction must precede outcome opening')
    if record.get('pilot', False):
        raise ValueError('Pilot cannot certify full prediction freeze')
    expected = {f'object_{int(identity)}/complete.json' for identity in ids}
    if set(record['objects']) != expected:
        raise ValueError('Prediction inventory differs from frozen identities')
    count = record.get('objects_checked') if family == 'reference' else record.get('n')
    if count != len(ids):
        raise ValueError('Prediction count differs from frozen identities')
    # Numerical failure is retained, not an invitation to remove the object.
    return bool(record['numerical_passed'] if family == 'candidate' else record['passed'])


def verify_freeze(root, path):
    record = json.loads(Path(path).read_text())
    if not record.get('predictions_frozen') or record.get('outcomes_opened') is not False:
        raise ValueError('Full pre-outcome freeze required')
    inventory = Inventory(root)
    for name, digest in record['hashes'].items():
        inventory.add(name, digest)
    inventory.unchanged()
    return file_hash(path)
