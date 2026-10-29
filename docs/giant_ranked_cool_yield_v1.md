# DESI-first cool-giant field yield

The [pre-registered ranking](../manifests/desi_cool_giant_field_ranking_v1.json)
selected two southern 10° development cells (RA 190–200 and 200–210,
Dec 0–10) and a separated northern 10° reserved cell (RA 210–220,
Dec 40–50). The ranking used qualified 4200–5000 K RVS giant spectra,
high latitude and low MWS reddening, before reading the DR11 photometry in
these cells. The original eleven-field roles, cold-v4 and XD models and
historical final-test identities were unchanged.

| Selected-field stage | Development | Reserved | All |
|---|---:|---:|---:|
| Conservative 4200–5000 K RVS giants | 225 | 59 | 284 |
| Clean DR11 grz and `E(B−V)<0.03` | 173 | 11 | 184 |
| Valid SpecDis distance | 166 | 10 | 176 |
| Valid SpecDis and DESI spectral S/N≥60 | 4 | 0 | 4 |
| Valid SPDist luminosity | 42 | 2 | 44 |

The 284 measured RVS giants nearly reproduce the ranking's 286 projected
objects; the primary yield loss occurs after requiring clean DR11 optical
photometry and then SPDist quality. For all 44 valid SPDist K giants, the
median approximate Legacy `M_g` uncertainty is 0.218 mag. Valid SpecDis K
giants have median fractional distance error 0.316, corresponding to a
first-order 0.69 mag luminosity uncertainty before any additional
method-specific errors. These two labels share Gaia-related training
information and are not independent truths. The clean `Teff<4200 K` group
contains 13 objects and zero valid SPDist labels.

The optical loss is concentrated among brighter giants. All 284 K giants
meet the field's `E(B−V)<0.03` condition; the three field quality counts
are:

| Field (RA, Dec) | K giants | Base grz | Fraction clean | Mask clean | Final clean | Median rejected / retained `g0` |
|---|---:|---:|---:|---:|---:|---:|
| 190–200, 0–10 | 92 | 92 | 65 | 70 | 64 | 15.82 / 17.73 |
| 200–210, 0–10 | 133 | 133 | 109 | 121 | 109 | 17.10 / 18.04 |
| 210–220, 40–50 | 59 | 33 | 11 | 12 | 11 | 15.53 / 18.28 |

In the northern reserved cell, 47 of 59 fail both the fraction and mask
conditions; separately, 26 fail base grz. The quality loss is strongly
associated with brighter objects. The aggregate flags do not
identify the specific mask bit for every rejected star, so no saturated
photometry is reinstated from this audit.

The reserved cell has only **two** SPDist K labels. Its 11 clean grz K
giants are the decisive validation bottleneck, not the original DESI
pre-selection count of 59. A K-giant photo-only model trained on the new
42 development labels could be fit numerically, but a two-star reserved
check would not establish transfer precision. The previously processed
eleven fields contain 42 additional valid SPDist K labels; combining them
would enlarge fitting data without fixing this reserved-field evidence gap.
The independent NGC 2419 comparison also has two of three `Teff<5000 K`
SPDist members below its RR Lyrae modulus by >0.5 mag.

**Decision:** do not fit or release a K-giant distance model from this
three-field set. Keep the exact-identity matched stars and their quality
states on NERSC for later development. A future cool-giant calibration
needs an independent system with more clean K members, stronger external
distance scale checks, and a reserved sky sample large enough to measure
scatter and tails. Expanding sky coverage alone will not resolve the
current SPDist quality and optical-selection losses.

Evidence: [field-role VAC profile](../manifests/giant_ranked_vac_profile_v2.json),
[exact-match VAC inventory](../manifests/giant_ranked_vac_v1.json),
[per-field optical audit](../manifests/giant_ranked_k_photometry_v1.json),
[joint class evidence](../manifests/joint_giant_ranked_v1_evidence_v2.json),
[NGC 2419 external system](../manifests/giant_ngc2419_reference_v1.json).
