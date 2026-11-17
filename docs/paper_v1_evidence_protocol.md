# Scientific checks for the Photdis v1 manuscript

Scientific baseline `8706940843975e9f9af657ea298aa5b7c94f64bc`. This protocol
uses archived DR11 source rows and posteriors. A native DR11 `TYPE=PSF` is
required. The manuscript source is an input to the figure handoff.

| Question | Denominator | Readout | Stop condition |
|---|---|---|---|
| Where can the F15 result be compared with the imaging catalogue? | 43,347,612 PDF rows, the two core tiers separately, and the exact two weak-Gaia F15 pairs | Disjoint native component cells in observed r, intrinsic g−r/r−z, maximum grz magnitude error, E(B−V), and nested equatorial HEALPix NSIDE=2 | Read every owned original partition once and conserve each cohort |
| Does the scatter advantage depend on reference boundaries or missing astrometric quality? | Identical Photdis and BJ stars for each reference, with states 3 and 4/5 split | Published full 2–5 kpc comparison, fixed 2.2–4.7 kpc buffer, quoted ±1σ modulus range inside 2–5, and all reference distances | Compare the four fixed definitions with the archived combined comparison |
| How usable are the reported posterior intervals? | Native strictly qualified 18–20 mag spectral parent with a full PDF and qualifying distance uncertainty, split by reference, class, survey, and r | Containment of reference centers, Gaussian reference-error overlap, reference center CDF, and predictive CDF for a Gaussian modulus measurement; separately use Gaia signed-parallax CDF conditional on its measured signal-to-noise window | Use the original probability mass and zero point, keep sparse counts and the separate near-width companion |
| What leads sources to the wrong distance layers? | Same observable-selected spectra before imaging class-score/median gates; conditional LF tests in the original spectral frame | Native color–absolute-magnitude and color–color median residual maps, matched observables strata, class counts, large errors, and their intersections | Use the two frozen reference products and three fixed LF scenarios |
| Can a subsequent user replay the measurement? | Frozen core FITS and PDF index, explicit source IDs and a calibrated companion example | ID/key, tier, checksums, full raw histogram, coverage cell and paired reference count | Verify on NERSC and deliver the portable reader and manifest |

For coverage, each reference is counted separately. A cell with 20 or more
F15 pairs is marked as having common observable support. The remaining
occupied cells retain sparse or absent evidence categories. Neither training
support nor the presence of a known class label is sufficient for this flag.
The 20-pair count is an analysis reporting choice. The weight is the
cell's target-to-pair ratio, normalized over paired stars in the overlapping
cells and clipped to [0.1, 10]. Report Kish effective sample size with the
reweighted residuals. This accounts for the stated observables within the
overlap only. It leaves spectroscopic targeting, success, and classification
dependence on stellar properties unresolved.

For the reference-error overlap, integrate a zero-centered Gaussian around
the archived reference modulus over the two Photdis interval boundaries.
For the forward CDF, convolve the Photdis PDF (uniform in each original
modulus bin) with the quoted reference modulus error and evaluate at the
reference center. Both are sensitivity diagnostics of a Gaussian
measurement approximation to the quoted errors. SpecDis and SPDist retain
their calibration and model histories. No reference posterior is used as an
independent likelihood. When a native histogram grid has fewer intervals
than another part, add only zero-mass bins for batching. For signed Gaia
in states 4 and 5, propagate `10^((10−mu)/5)` in mas, add each source's
reported parallax error, and condition on the selection window in measured
parallax signal-to-noise. Count negative parallaxes directly. State 3 is
included only in the paired-distance sensitivity, because a positional
identification alone does not establish a usable parallax likelihood.

For P1, infer neither population fractions outside the spectroscopic
frame nor a single confidence interval from the three LF scenarios.
Class labels and severe-error indicators can overlap. Each sensitivity
scenario is applied to an already targeted and quality-selected
spectroscopic template, and the predictive counts share denominators.

Photdis distance inference uses the two dereddened native grz colors and
their propagated errors. The bright-red near-tier admission also requires
the existing primary photometric gate, `dwarf_score >= .80` and
`pooled-SED P(d < 10 kpc) >= .90`, plus its frozen W1-state, quality, r,
color and QRF median cuts. The auxiliary SED uses the valid native
`grizW1W2` measurements or explicit partial measurement states.
The intermediate core has the original working-state score and
`QRF P(2 <= d < 10 kpc) >= .50` followed by `2 <= d50 < 5 kpc`.
Working, core-by-prediction and reference-distance slices are distinct.
