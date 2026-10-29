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
photometric-catalog class fractions.

## Four-field blue-side inventory

The allocated-NERSC inventory uses the observed dereddened region
`(g−r)₀<0.9090`, low dust and clean `grz`. Its strict Gaia luminosity anchors
also require corrected-parallax S/N≥20. Counts below are by field, not a
spectroscopic estimate of the full photometric candidate composition.

| Field | Strict blue Gaia anchors | Of these at `18≤r<20` | DESI metal-poor cool dwarfs | DESI cool giants | DESI known nonstars |
|---|---:|---:|---:|---:|---:|
| RA205–215 | 404 | 12 | 25 | 16 | 1,894 |
| RA0–10 | 1,037 | 69 | 0 | 0 | 0 |
| RA150–160 | 1,624 | 47 | 5 | 5 | 1,323 |
| RA220–230 | 1,142 | 14 | 14 | 7 | 1,584 |

The four fields supply 4,207 strict blue Gaia anchors, including 142 at
`18≤r<20` and none in the recorded `20≤r<26` bins. None of the 44 DESI
metal-poor cool dwarfs
overlaps those *strict* S/N≥20 anchors; the four with Gaia `M_g` noted above
meet the looser S/N≥10 diagnostic. The original v4 training fields contain
78 cold-anchor rows blueward of the later box edge, distributed 12/22/42/2
across its four training fields. These rows show the available color range;
they do not define a metal-poor training branch.

The [color and luminosity figure](../figures/cold_v5/01_blue_inventory.png)
shows the DESI poor-dwarf locus near the optical boundary and a separate
faint Gaia sequence in the blue region. The latter requires a white-dwarf
check before assigning blue Gaia anchors to a cool-dwarf luminosity relation.
DESI quality-qualified blue giants and known nonstars are present in the same
color region, especially nonstars at `20≤r<22`. The counted DESI nonstars
are spectroscopic controls, not a contamination fraction for the photometric
survey. Detailed field/brightness/color counts and label overlap are in
`manifests/cold_v5_blue_inventory.json`.
The RA0–10 field has no DESI class labels in this matched input, so its zero
DESI counts describe coverage rather than a class absence.

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
