# Photdis v3.0 candidate-catalog interface

Status: production and validation are in progress. This page defines the candidate
interface; final quantities and acceptance belong to the delivery audit. Historical
v1 application products remain unchanged.

## Location and scope

Canonical root on NERSC:
`/pscratch/sd/y/yunao/photdis_pipeline/data/v3_0/catalog/`.
The existing 14-field parent contains 9,390,518 unique objects. The qualified v3
candidate envelope contains 1,183,990 native PSF objects in 13 DECam fields.
The northern field's 491,184 objects remain explicitly unmodeled. A same-selection
whole-DR11 denominator and effective masked area are unavailable.

Candidates require the frozen native clean grz/high-latitude/low-extinction
admission, r<20, maximum grz magnitude error<0.1 mag and reference color support.
They are not selected using Gaia, spectroscopic class, reference distance or a
near-probability threshold. Every original row has a `parent_status.npz` entry;
rows outside the envelope are not silently treated as distance predictions.

Each `field_NN/` contains:

- `prepared.json`, `inputs.npz`, `parent_status.npz`: exact admission and exclusions.
- `predictions/`, `reference_audit.json`: nominal reference-prior predictions and
  their actual full-product audit.
- `population_transfer/refit_DD_sS/`: the conditional target-prior joint products,
  matched-response reference predictions, shared parameters and count realizations.
- `candidate_v3_0.npz`, `candidate_v3_0.json`: compact per-star candidate table,
  column schema, version hashes, units, scope and numerical-failure counts.
- `candidate_v3_0_chunks.csv`: checked links to native full PDFs, reweightable
  joint support, coefficients, source normalization and shared count products.

`transfer_plan.json` records the retained **global response** models and rigorous
omitted-mass bounds. It never removes stellar modes or SED kernels. If the plan
retains one overwhelmingly weighted response, the table is that conditional
product, with the explicit finite-mixture approximation bound. A small discarded
weight does not establish adequate continuous reference-model support.

## Per-star quantities

`identity` and `ls_id` are exact integer identifiers; `TYPE` is verified `PSF`.
`RA`, `DEC`, `field`, `photometric_system`, `band_mask` and `band_state` retain
native provenance and distinguish legal weak flux from unavailable observations.
The canonical native band order is G,R,I,Z,W1,W2.

`mu` contains distance-modulus quantiles 0.025,0.16,0.5,0.84,0.975. Convert with
`d_kpc = 10**((mu-10)/5)`. `modes`, `mode_mass` and `minor_mode_mass` preserve
multimodal summaries; the full native log-PDF is linked, including distant tails.

`probability[N,3,6]` is the joint class×distance mass. Class order is
`dwarf,evolved,other`; distance bins are 0–2,2–5,5–10,10–20,20–50,>50 kpc.
`class_probability`, `distance_probability` and `qnear` are integrals of this
same joint distribution. Do not form a joint probability by multiplying the two
marginals. Missing spectroscopic class is not an extra physical class.

`reference_mu`, `reference_qnear` and `prior_delta_*` use the **same response and
calibration draw** as the target product. They isolate prior sensitivity. The
separately stored nominal-reference model is a different comparison.

Flags are descriptive, not externally calibrated quality guarantees:

- `class_ambiguous`: maximum class probability below0.8.
- `multimodal`: at least two distance modes with probability mass above0.05.
- `prior_sensitive`: absolute change in qnear above0.05.
- `supported`, `extrapolation`: fixed reference support.
- `discrepancy_probability`, `negative_intrinsic_support`: model-inadequacy and
  intrinsic Gaussian-flux-support diagnostics; legal negative observed flux is a
  separate issue.
- `numerically_converged`: reference quadrature and final joint half-chain error
  below0.005, including the global response omission bound.

Numerical failures remain present and flagged. No statement of calibrated
accuracy follows from numerical convergence alone.

## Shared counts and reuse

`summary/counts.npz` and per-field count files retain the full class×distance
covariance, latent int64 count draws and shared parameter identities. Mean counts
sum per-star joint probabilities. Covariance is accumulated across fields using
common parameter indices before marginalization; fields must not be combined as
independent posterior estimates. Latent membership RNG namespaces differ across
fields so that accidental common random numbers do not create extra covariance.

These are unit-weight counts for the actual qualified observed catalog. The
separate 26,000-object sampled-parent Horvitz–Thompson diagnostic has design
uncertainty; it is not the candidate catalog's posterior count interval.

The candidate distribution is a modular conditional transfer from the target
sample. Shared population parameters are not re-estimated using every remaining
candidate. It is not a full-catalog Bayesian generative refit, an intrinsic Galactic
density reconstruction or an independently calibrated census. Finite-response
concentration, source-scale dependence and poor interval coverage in some real
samples constrain scientific use.

For alternate priors use the linked joint latent support/coefficients and the
explicit reference prior, rather than reweighting only the distance median or
class marginals. Preserve source hashes, global draw indices and normalization.
For observed-space log scores, an omitted-response probability bound alone is
insufficient; the transfer evaluator exports Gaussian-likelihood density bounds.
