# Bounded photo-only evolved-star test, registered before scoring

## Scientific scope

The corrected exact-identity inventory contains 2,487 clean, low-extinction
RVS conservative giants. SPDist supplies 2,044 quality-valid luminosity
labels, but only 42 lie at 4200–5000 K and none below 4200 K. This first
test therefore targets the **5000–7000 K spectroscopically identified evolved
population**. Its inference inputs are solely dereddened Legacy `(g-r)0` and
`(r-z)0`; temperature, surface gravity, iron abundance, Gaia and spectra
determine the reference sample but are not prediction features. The cool RGB
and RC branches remain separate feasibility questions.

## Labels and statistical method

- Join SPDist v1.0 on unique exact `(TARGETID, Gaia SOURCE_ID)`. Require
  `FLAG_GOOD`, `QFLAG_INPUT`, finite percentile magnitudes and usable Legacy
  `g0`; exclude CMD/spectral conflicts and known reference-system members.
- The label is `M_g,Legacy = g0,Legacy - (G0,Gaia - MG_50,SPDist)`.
  Its uncertainty includes Legacy `g`, Gaia `G`, half the SPDist 16–84%
  percentile span and the 0.167 mag intrinsic precision recommended by
  SPDist, in quadrature. SPDist and SpecDis are both trained with Gaia
  information and will not be counted as independent distance truths.
- A two-color Gaussian kernel gives weights to training stars. Its covariance
  includes the measured errors of both stars, including the negative shared
  `r` covariance between `(g-r)` and `(r-z)`. Each weighted training star
  contributes its full Gaussian luminosity component, retaining more than one
  luminosity solution at the same color. The component width includes its
  catalog label error and a fitted smoothing floor.
- Search color bandwidths 0.06, 0.12, 0.24 mag and luminosity floors 0.15,
  0.30, 0.60 mag. Select by mean log predictive density with complete
  development fields left out one at a time. Include the held-out label
  error when scoring that label. The reserved fields and M3 play no part in
  selection.

## Isolation and evaluation

The existing seven-field and four-targeted-field `development`/
`reserved_validation` roles remain frozen. Train the chosen mixture on all
development stars, then evaluate once in the reserved fields. Report
per-field median `M_g` residual, NMAD, `|residual|>1` mag fraction, 68/95%
interval coverage of catalog point labels, log density and local training
support. The support flag requires the query to be inside the development
color convex hull with effective kernel sample size at least 20. Report all
rows and the supported subset separately.

M3 (NGC 5272) is a whole-system holdout outside those eleven fields. Its
five qualified warm evolved members have a near-infrared RR Lyrae reference
distance modulus `15.041 ± 0.017(stat) ± 0.036(sys)` mag. Compare each
photo-only distance distribution and their median with that reference, using
actual DR11 color uncertainties. M3 tests an external distance scale at
roughly 5300 K and does not validate the cooler K giant branch.

A second, cooler system is registered before inspecting DR11 member
measurements: NGC 2419, RA 110–120°, Dec 35–45°. The published member list
is matched by exact Gaia source ID to DR11 south sweeps; all four 5° sweep
tiles are examined on allocated CPU. Its independent RR Lyrae modulus is
`19.71 ± 0.08` mag. The cluster has E(B−V)≈0.08, so report its measured
reddening and optical quality per member rather than applying the training
sample's E(B−V)<0.03 requirement. This system checks cool-giant VAC labels
and later any cool-giant photo-only model, with the dust difference explicit.

This is a bounded conditional-luminosity experiment, not a type classifier
or an all-sky giant distance release. Internal SPDist comparisons measure
agreement with that catalog; the M3 check measures agreement with an
independent system distance. Report both distinctly.

Sources: [DESI SPDist documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-spdist/),
[DESI SpecDis documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-specdis/),
[M3 RR Lyrae distance study](https://arxiv.org/abs/2008.11745).
NGC 2419 reference: [Di Criscienzo et al. 2011](https://arxiv.org/abs/1011.5398).
