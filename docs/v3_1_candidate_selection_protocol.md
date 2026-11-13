# v3.1 candidate-envelope normalization pilot

Registered before candidate-normalization outcomes or selection-matched refitting.
The fixed candidate envelope is native PSF/DECam, observed r<20, all three grz
magnitude errors<.1, and the frozen grz QRF support predicate. The26000sample
contains3540such observations. No class, distance or residual enters admission.

For a response12 target-relative calibration c in magnitudes, the likelihood
coordinate is F'=F/T*10^(.4c), with corresponding reported error E'. The cuts
become F'_band>(2.5/ln10)/.1*E'_band and F'_r>10*10^(.4c_r)/T_r.
Evaluate QRF support on22.5−2.5log10(F')+c, the original extinction-corrected
magnitudes. Applying a new calibration to the original admission decision would
select a different catalog and is prohibited.

Intrinsic Gaussian kernel covariance plus reported measurement covariance gives
a3D Gaussian in observed grz flux. Sequential survival-weighted truncated-normal
Sobol integration handles the positive flux thresholds. Apply the exact original
QRF support predicate to the surviving draws; preserve both binary states and
all128SED kernels. This inherits the original response's conditional treatment
of reported errors/transmission and fixed dust corrections. It is not an area,
morphology or survey-completeness correction.

Numerical pilot: among the3540fixed candidate rows, sort by the mean log calibrated
grz error, with identity tie-breaks; select minimum, median and maximum positions.
These are recorded observed contexts, not fitted label/residual strata. Use the
fixed response12model and calibration12. On each context integrate128 and512
Sobol draws with seed317101 and512independent draws with seed317102, mu[-15,45]
in.1mag increments. A.05mag sidecar on the median context checks broad-basis
quadrature. Preserve original artifacts and flags. Require each5-basis,
per-kernel entry integral discrepancy<.003 and candidate entry no greater than
the same-context originalgrzSN5 entry within.003. This is a numerical pilot,
not a representative-domain calibration or permission to apply coarse context
compression without its own audit. Failed results require bounded numerical
refinement before any selection-matched population comparison.

The eventual comparison keeps the sameSED/response and prior family across
all26000, the3540subset and whole held-out fields. Selection normalization must
match each sample. Context compression, fitting and predictive transfer are
still pending; no population scientific acceptance follows from this pilot.
