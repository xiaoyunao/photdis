# Native reservation I/O amendment

The original application joint inference and independent-SED recovery completed in allocation61123779. Its remaining nativeTYPE reservation used CFITSIO column reads with many small I/O calls; after about13minutes it had not completed the first field. An allocated sequential-read measurement obtained128MiB in0.477seconds. The job was deliberately stopped after recovery step completion; this is not a scientific/model failure.

Preserve all application/recovery chunks and their frozen source hashes. The original population and recovery programs remain unchanged. Replace only the pending reservation reader with `v2_4_reserve_fast.py`: sequential fixed-width FITS rows, explicit no-scaling/header-width assertions, independent CFITSIO row checks, and atomic per-field checkpoints. Preserve the same2500identity-ranked candidates, observable cuts, verifiednativePSF gate and first250admitted objects per field. Do not read predictions or associate reference outcomes for the reserved identities. No change to the target, models, labels or scientific selection.

The replacement has its own code/amendment freeze. Resume only after the original job has terminated. Repoint the existing dependent validator to the one reservation job; do not duplicate prediction or recovery jobs.
