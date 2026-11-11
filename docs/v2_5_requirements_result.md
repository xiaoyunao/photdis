# v2.5 requirement and evidence ledger

Review source: [full latest Pro review](pro_review_de4904b.md).
Protocol: [frozen scope](v2_5_protocol.md). Actual status follows execution
manifests and [result report](v2_5_result.md), not this implementation checklist.

| Review requirement | Implemented evidence | Scientific limit |
|---|---|---|
| Red transfer without PIT selection | `red_matches.csv`, `red_migration.csv`; full711, controls nonfit, colors/r/errors matched | Historical versus application selection remains different; reused controls clustered |
| All signed Gaia in original metal pairs | `metal_gaia_objects.csv`, `metal_gaia_pairs.csv`, `metal_gaia_availability.csv` | No weak-parallax inversion; source/fit overlap explicit; teacher and selection dependence remain |
| Stellar-group band gains | `band_gain_by_class.csv` | Each class has its own sky interval; no faint evolved labels or physical information upper bound |
| Explicit target population inference | `population_v2_5.py`, response/posterior caches | Four population tilts, not free20cell fit; no full catalog/deployment |
| Selection normalization | `selected_probabilities`, independent rejection-sampling test, response alpha | Limited observed-r/error selection; no detector/crowding/PSF-generation selection |
| Near bins and far contaminants | 1/3.5/7.5/15/30kpc nodes, six scoring bins | Discrete toy distance model; >50QRF mass retained in error |
| Source-separated forward and recovery templates | cal split versus224held-out test identities, exact fit-ID exclusions | Same teacher families and sky systematics are not independent absolute truth |
| Missing class labels keep luminosity information |32cal missing-label templates, four-way soft memberships | Missing labels are marginalized, not a fifth class; kernel-imputation assumption stressed with labeled-only response |
| Distinguish one-factor shifts | reference-like, giant-only, poor-only, distance-only plus four original mixture definitions | Quota templates and cal-derived reference mix are not Galaxy priors |
| Same-object point/PDF/prototype comparison |192fresh catalogs ×128objects, full QRF posterior chunks | Selection domain versus complement stress reported separately |
| Actual repeated-catalog coverage |12independent noise/identity draws per design/regime, Wilson bounds | Small number of catalogs; parameter and predictive count intervals not conflated |
| Separate uncertainty sources | parameter posterior, independent catalogs,16template bootstraps,±teacher sigma | Sensitivities not a fully calibrated combined uncertainty distribution |
| Retain degeneracy | correlations, boundary mass, quadrature ESS/grid differences | Wide or unstable posterior cannot be published as a precise physical ratio |
| Prospective3250 | outcome hash preserved, equal-field unused-population estimand and metrics frozen | Remains unopened until model/selection suitability; no new-sky/final-validation claim |
| Existing deployment and morphology | original model hashes checked, PSF admission/export | No full-parent rerun, model search or automatic adoption |
| English scientific plots | eight source-linked figure groups, visual verification recorded separately | Formatting checks do not imply scientific acceptance |

Progress and actual failed/accepted criteria belong in WORKLOG, execution records,
and the measured report. No task is called complete solely because code exists.
