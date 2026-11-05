# Fixed weak WISE flux contrast

Source: GPT Pro review of 0001fd8. Three arms share the same joint templates,
spatial split, optical noise and evaluation rows. Use only new split-0 frozen
expanded A distance-training rows with valid high-S/N (>=5) W1 and W2 as
templates. This selection limits transfer, particularly to faint stars.
Frozen expanded A is a separate benchmark with different data and estimator.

Templates retain (Mg, W1/z flux ratio, W2/z flux ratio) together. For each of
32 shared grz Sobol perturbations, find 96 nearest templates in training-
standardized grz colors, with Gaussian weights and bandwidth half the 96th
distance (minimum 0.03 standardized units). Each optical draw has equal prior
weight. Predict positive observed WISE fluxes using the perturbed z0 flux,
template ratio and the target WISE MW transmission. Flux likelihood uses
observational variance plus a fixed 5% fractional template term. Jointly
normalize over neighbors and optical draws, preserving optical/IR covariance.
Use common 0.1 mag bins with linear deposition. Output full PDFs for all arms.

W1 arm uses valid W1 (states 3/4). W1/W2 arm uses each valid band independently;
the strict incremental W2 comparison conditions on eligible W1, and the
W2-only fallback has its own row. Missing/failed bands contribute no likelihood.
Retain signed flux and report zero and negative counts, per-band and joint
weighted measurement coverage, 18–20 weak bands, high S/N, failed/missing,
each class and calibrated-domain metal-poor dwarfs. Compare refined signed Gaia
scores and chosen/source residuals, widths, coverage and paired cell bootstrap.
Use fixed development splits and no threshold search. Independent sky is needed
to validate any promising candidate.
