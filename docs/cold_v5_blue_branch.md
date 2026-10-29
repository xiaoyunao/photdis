# Metal-poor blue-side training definition

This is a separate science branch under development. It does not change the
frozen cold-v4 optical box, distance model, WISE20 route or candidate flags.
The diagnostic boundary is the v4 lower edge `(g−r)₀=0.9090`; its
`(r−z)₀` lower edge is `0.4007`. Being blue of either edge is an observed
failure reason, not a calibrated metal-poor classification rule.

## Evidence already available

The [v4 DESI boundary audit](cold_v4_type_boundary.md) uses one unambiguous,
quality-qualified DESI spectrum per source, finite atmospheric parameters,
`3000≤T_eff<5000 K`, and `log g≥3.5` for cool dwarfs. Its calibrated
`[Fe/H]<−1.5` subset has the narrower abundance validity region
`4200≤T_eff<5000 K` after intersecting with the cool-dwarf definition.

| Field | Poor dwarfs in valid domain | Rejected by optical box | Rejected with clean grz | Rejected with Gaia parallax S/N≥10 |
|---|---:|---:|---:|---:|
| RA205–215 | 40 | 25 | 20 | 2 |
| RA150–160 | 8 | 5 | 5 | 0 |
| RA220–230 | 14 | 14 | 10 | 2 |

All 44 rejected objects are below the `g−r` edge; 17/25, 2/5 and 8/14
also fall below the `r−z` edge. Median rejected `(g−r)₀` is approximately
0.75–0.76. Four of the 44 have a Gaia-derived Legacy `M_g` with parallax
S/N≥10. The spectral targets are selected by DESI and do not measure
photometric-catalog class fractions. An inventory of strict Gaia anchors,
brightness/field coverage, and overlapping giants/nonstars is being run
separately; its aggregate results will be added here.

## Proposed label and anchor contract

1. **Spectroscopic reference:** retain unique, unambiguous DESI matches with
   accepted stellar/parameter quality. Use the calibrated abundance only
   inside its `T_eff`, `log g` and error validity domain; report raw and
   calibrated abundance separately. `[Fe/H]<−1.5` within this domain
   identifies the first metal-poor diagnostic subset. Keep cool giants,
   nonstars, other stars and uncertain spectra as distinct controls.
2. **Geometric luminosity anchor:** use Gaia-corrected high-S/N parallax,
   validated Gaia color/astrometry, low `E(B−V)<0.03`, clean Legacy `grz`
   and finite band errors. Gaia supplies `M_g`; DESI abundance labels the
   small overlapping subset. Blue Gaia anchors without abundance cannot be
   assumed metal-poor solely from their colors.
3. **Application inputs:** keep Legacy `grz` as the minimum observable set;
   W1/W2 may add information when quality-qualified. The target catalog
   cannot require Gaia or DESI. A future classifier must allow uncertain
   stellar type and a separate luminosity relation for the blue branch.
4. **Coverage and holdout:** measure anchor and DESI label counts jointly
   across `(g−r,r−z)`, `r`, sky block and parallax S/N. Reserve distinct
   sky fields and source identities before fitting a color–luminosity
   relation; validate cold-dwarf completeness, giant/nonstar leakage and
   absolute-distance residuals in those held-out fields. Do not infer a
   class probability from the DESI-targeted class balance.

The [Bochanski et al. subdwarf work](https://arxiv.org/abs/1211.6104)
shows that `g−r` relative to `r−z` can carry a metal-poor M-subdwarf
signal. Its SDSS relationship requires recalibration in Legacy/DECam
photometry before use here. The [Schmidt et al. relation](https://arxiv.org/abs/1605.03732)
is measured for `3550–4200 K` and approximately `−0.5<[M/H]<0.2`;
it cannot serve as a metal-poor halo calibration. Neither study requires
that this project's first blue-branch step have a `u` band or APOGEE.
