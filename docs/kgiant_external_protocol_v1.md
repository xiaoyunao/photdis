# Fixed K-giant external check (registered 2026-11-02)

Source: [complete Pro review](pro_review_a399c3e.md). This protocol precedes
external distance prediction. Historical a399c3e products remain frozen.

## Fixed scope and identity

- Pal12: Gaia DR3 6817950249538781952, the unique cached P>.9, PSF,
  clean grz member with g0−16.3<.5. Cached g0=16.52601, W1 invalid.
  Position from membership catalog: 326.625880, −21.267698 degrees.
  Match external literature identity before assigning temperature; never infer
  a literature star number from brightness alone.
- Terzan7: Sbordone2005 Table1 stars1272,1282,1515,1665,1708 only.
  J2000 coordinates and Teff are transcribed in configs/kgiant_external_v1.json.
  Nominal K scope4200≤T<5000;1665 at3945K is a retained scope failure.
  1282/1708 are boundary-sensitive. Temperatures derive from B−V and E=.07,
  not independent spectroscopic thermometers; ±100K is a sensitivity bracket,
  not a reported Gaussian error. Radial velocities157.8–160.6km/s support membership.
- Match DR11 south within2arcsec of literature position; report nearest/second
  nearest and reject ambiguous multiple matches within2arcsec. Preserve exact
  Gaia identity and release. Native optical system must be DECam.

## Independent quality and evidence flags

Retain original grz_clean and PSF requirements, W1_clean separately, and EBV<.03
as an independent old-scope flag. No saturation or blending threshold changes.
External K identity needs a verified literature counterpart with giant/member
evidence; phase may remain unknown for the mixed-stage K model. No RGB-only claim.
Color-box support is reported per teacher/model without residual-based rejection.
Pal12 cache is reused; only the five Terzan7 positions require a sweep scan.

## Frozen prediction and reference

Use existing population.npz, qualified W1 training pool per teacher, positive
finite original teacher width, exclude cluster_group!=0 and all reference IDs.
Original SPDist/SPECDIS width definitions and bandwidth .12mag / Mg floor .15mag
remain unchanged. Record training IDs/hash before scoring. Within each teacher,
prior/A_grz/B_grzW1 share training identities. Query grz suffices for prior/A;
query W1 is required only for B. Do not tune using external residuals.

Pal12 reference μ=16.3±.1: [Musella2018](https://arxiv.org/abs/1709.08509).
Terzan7 μ=17.05, d=25.3±.6kpc (σμ=5/ln10×.6/25.3):
[Siegel2011 Table1](https://arxiv.org/abs/1108.6276).
Both are model-dependent CMD/isochrone scales, not geometric truths.
[Cohen2004](https://arxiv.org/abs/astro-ph/0311187) is the Pal12 identity source;
[Sbordone2005](https://arxiv.org/abs/astro-ph/0505307) is the Terzan7 source.

Retain catalog transmission baseline. Perturb the *catalog SFD coordinate* by
δEBV=±.01 using A_band/EBV from each cached transmission, simultaneously in g0
and colors. This is an explicit sensitivity range, not substituting physical
literature E=.03/.06 into SFD or claiming a measured reddening error.
If scoring is possible, marginalize a common δEBV~N(0,.01²) per system and a
common reference δμ with the stated σμ (independent approximation; covariance
unavailable). Save shared draw IDs, never divide systematic errors by sqrt(N).
Report baseline and sensitivity distributions separately, not adjusted to agree.

## Endpoints

Deliver independent flags for six targets, accepted per-star mixtures and
quantiles/relative reference offsets, support and dust responses, plots and
provenance. No aggregate accuracy/coverage calibration from this small sample.
Quality or identity failures are valid final outcomes; no additional targets.
SPDist extrapolation at cool K temperatures must remain explicit.
Also correct coverage/teacher error interpretation using190-row evidence and
the ideal noisy-center example Var=2σ² (raw68% coverage≈84%).
