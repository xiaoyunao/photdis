# v3.0 request-to-evidence ledger

Status: final production is in progress. The authoritative complete review is
[pro_review_b59e18b.md](pro_review_b59e18b.md). This ledger distinguishes implemented
and executed work from pending output; it is not the final acceptance marker.

| Requirement | Implementation and actual evidence | Remaining delivery / limitation |
|---|---|---|
| A: verified PSF, exact IDs, isolated roles | `v3_0_prepare.py`, `photdis.v3.data`; preparation/role manifests;612114 admitted references | Teacher/source selection remains conditional; no claim of random Galactic representation |
| A: remove Gaia-dependent SpecDis flag truncation | Rejoined original rows, BINARY_FLAG retained;10298 qualified non-1 flags; primary source fixed before analysis | Source dependence is recorded; alternative VAC/Gaia likelihoods are not multiplied |
| A: full FeH/missing-class/native bands | Native six-band records;32768 fitted identities include5653 missing-class and12270 missing-FeH sources | Missing labels marginalized, not a fourth physical class; low-support populations remain limited |
| B: continuous empirical forward model |128-component error-aware fit,43iterations; native-flux continuous quadrature, correlated teacher errors | Gaussian intrinsic-flux approximation has negative-support diagnostics; not a positive-flux physical stellar model |
| B: weak/negative/missing/upper-limit flux | One native likelihood per measured band; projected missing coordinates, censored likelihood tests | WISE quality differs from low SNR; no all-WISE admission requirement |
| B: modes, binary/discrepancy, dust/system | Fixed10% equal-flux binary proxy,1% discrepancy; shared dust/calibration; fullnative kernels/tails retained | No binary mass-ratio inference, unrestricted dust-law fit or north-as-DECam substitution |
| B: near/far continuous range |0–2/2–5/5–10/10–20/20–50/>50 integrals, native log-PDF and reweightable joint support | Output bins are not distance nodes; far science remains reference-limited |
| C: source scale/scatter/tails |8192source-calibration objects per VAC,32shared sky-bootstrap draws and32full SED refits | Gaia-dependent teachers do not become independent absolute scales |
| C: constrained population/selection | Field-shrunk class×continuous-distance bases and within-class SED tilt; selection-normalized26000 sample | Target likelihood is design-rescaled; reference density is not a universal target prior |
| C: correlated actual counts | Shared-draw responsibilities, int64 latent counts, conditional/parameter/response covariance | 26000sample joint/count ensemble passes numerical gates and7329preservation checks; finite-response ESS=1 remains unresolved support |
| D: same-object method/band/probe | Matched32768-identity QRF,grz/griz/multiband, no-color/color probes;47899primary distance and42950class development | Gains and losses retained; log-score tail sensitivity prevents misleading overall-score claims |
| D: continuous recovery/migration |224nominal conditional realizations;3584full shared-response/catalog fits;14scenarios×8catalogs | 112/112 final numerical passes; narrow4kpc count coverage remains0/8; repeat noise does not create new physical templates |
| D: fixed failure strata |2–5kpc,red,poor-metal dwarf,other;distance/PDF/Gaia/class/count metrics | Narrow4kpc and poor-metal biases remain scientific failures |
| E: freeze before real outcomes | `method_freeze.json`,SHA14d3ff1d...; source/core/priors/metrics locked | Numerical-only refinements preserve physical assumptions and earlier failed products |
| E: original3250 reservation | All predictions and exactassociation complete;3231Gaia IDs allmatched but failfrozenquality;19noID | Zero qualified distances/VAC/classes: no accuracy conclusion and no sample replacement |
| E: genuinely new sky | Photometric-first2048;1070signedGaia including108negative,247precise | Same-response paired target transfer complete; interval undercoverage and1/2048 Monte Carlo flag retained |
| E: independent scale/science |8192photometricNGC288 field;1442post-associated members; shared9.03kpc dynamicalscale;5inspectedfigures | Matched target transfer and1442member science example complete; target68coverage47.43%; membership Gaia-conditioned, one common scale |
| F: measured scaling/full qualified parent |45882field benchmark437.96s/3.927GB; full1183990candidate reference production | 333084reference checks complete; verified ensemble available; conditional fullcatalog transfer running, counts/export pending |
| F: complete posterior interface | Compact candidate table plus indexed fullPDF/joint support/sharedparameter/count products | No automatic replacement ofv1; output scope and numericalflags preserved |
| F: coverage and science claims | Parent statuses, newskyavailability, sampledesign diagnostic and fullunitweightcounts separated | No same-selectionallDR11denominator/effectivearea; real population contamination remains unmeasured |
| F: final scientific delivery | Development/recovery/prospective figure families, methods, transfer scope and catalog guide | Finalcountfigures, inspectedatlas, resultreport andactualdeliveryaudit remain |

Historical known-model, same-template and independent-transfer regressions remain
preserved, alongside new continuous simulations. There is no model-family search,
new periodic monitoring, subagent execution or deployment replacement.
