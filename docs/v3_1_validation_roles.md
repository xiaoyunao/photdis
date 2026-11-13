# v3.1 prospective roles — before correction fitting

Freeze actual identities using `v3_1_freeze_roles.py` after native availability
and history audits, before fitting a correction or reading model residuals.
Availability includes reference values/precision and labels; this is an unused
source/block validation design, not a claim that all reference outcomes were
hidden during acquisition. No selection uses Photdis prediction errors.

All newly acquired APOGEE field rows outside historical identities, imaging
identities, footprints and published systems are held out. Keep two named roles:
`new_sky_field` (unused coarse sky cells) and `new_source_seen_sky_field`.
The second role does not establish new-sky independence. Exclude published
cluster/association/region/wide-binary annotations from these field roles.
Do not filter by successful distance or class evidence: eligibility is independent
per reference channel. No new rows become correction training or tuning data.

Freeze admitted NGC1261 strict published members as `new_common_scale`.
NGC1851 and NGC7492 remain empty under the fixed cuts. Any historical source or
Legacy imaging-ID overlap quarantines an entire common-scale system. Membership
is conditional on published Gaia astrometry. The HST-kinematic scale and its
common uncertainty are specified in `v3_1_common_scale_protocol.md`; thousands
of members are one independent system scale, not thousands of scale replicates.
An NGC1261 structure application is a demonstration sharing this validation
system; its background sample must be separately frozen before its fit.

Use every available `ls_id` in the nine upstream historical inputs in addition
to Gaia identities. Decode negative historical `identity=-ls_id` as a second
guard. Original availability products stay unchanged; write a new role artifact
with exact hashes, admission/exclusion counts and block assignments. Reference
files are hashed but are not passed to prediction workers. Prediction export
must verify native TYPE=PSF again and contain source/order/model/prior hashes.

Freeze the eventual method and all predictions before opening paired residuals.
Compare all pre-registered eligible channels, with sky-block resampling for field
rows and common-system distance draws for members. AstroNN distances share Gaia
training and do not provide an independent absolute scale. Weak or negative Gaia
parallax stays in observed space; never invert it. Retain every failed numerical
flag, sparse stratum and missing evidence cell. No tuning after final opening.
