# Faint signed Gaia validation

This registration implements the full review message `3beaa71f-2263-40c4-8355-e434be5c3962` at Photdis `fcd59ee9`. The frozen native `grz` QRF, SFD correction, PSF admission, Gaia astrometric quality, zero-point correction, and historical exclusions are inherited from the nine-field archive. The paper and production catalog are outside the experiment.

## Pilot and expansion criterion

L032 and M249 both have supported weak Gaia parallaxes in [20,21) and [21,22). They provide contrasting environment and sample size. In each field and magnitude bin, use a stable identity-ranked sample of at most 6,000 eligible PSFs, retaining both positive and negative measured parallaxes. Join the original faint parent to reproduce exact Gaia uniqueness and compare counts against the published inventory. Save identities, native PDFs, and signed outcomes in separate artifacts. The power code reads identifiers, uncertainties, and PDFs. Actual signed outcomes are used after the power decision.

A shift (Deltamu) transforms each native histogram quadrature point to (10^{(10-mu-Deltamu)/5}) mas. Convolve with the star's calibrated Gaia measurement error. The likelihood and PIT re-normalize at the inherited measured cut (y/sigma_arpi<5). Detection and quality flags enter by holding the selected ancillary roster fixed. Positive and negative measurements enter the same likelihood. Gaia (G), color, and ecliptic-latitude zero-point correction follows the original catalog implementation. [Lindegren et al.](https://doi.org/10.1051/0004-6361/202039653)

For the scale injection, project the difference between measured and predicted conditional mean parallaxes onto the PDF response at (pm0.10) mag. The working covariance combines selected per-source variance with a 0.010 mas shared field component and a 0.010 mas component shared in each 0.30° spatial block. The alternate calculation doubles both scales. These explicit RMS assumptions exercise correlated response and systematic sensitivity. The eventual assessment should distinguish robust signal from signal sensitive to this assumed floor. [Vasiliev & Baumgardt](https://doi.org/10.1093/mnras/stab1475)

Draw latent moduli from the unchanged piecewise-uniform histograms, then Gaussian noise and shared offsets. Within each offset realization, require the same (y/sigma_arpi<5) cut for each simulated object. Calibrate a two-sided 5% test with 512 zero-shift realizations, verify it with an independent 512 null draws, and use 384 draws at each of −0.20, −0.10, +0.10, and +0.20 mag. A magnitude bin advances only if both pilot fields have at least eight blocks, realized null rejection in [2%,8%], and directional power ≥80% in all four injections at the nominal offset scale. Keep stress results and failures. If admitted, expand that bin to all eligible objects in the nine fixed fields. Pilot scores for an underpowered bin are descriptive.

Assess the real distribution with same-star log-score contrasts, conditional PIT/SNR histograms, the negative-parallax fraction, and the scale projection at both systematic assumptions. Resample the same spatial blocks across PDF variants. Report injection error and actual outcome diagnostics separately. The test concerns the selected ensemble's parallax distribution. Parallax inversion and population extrapolation have no role in it.

## Population comparison

Reuse the archived captured Gaia sample and SFD full-range (A), equal-map-support SFD (B), and converged photometric Bayestar (C) PDFs. Retain common reference support and apply the historical split. Within field, use fixed equal-mass overlap weights for dwarf proxy and uncertain Gaia CMD classes in the same SFD (g-r), SFD (r-z), observed (r), reference modulus and SFD color-excess cells. Widths are in the config. Report losses, effective sample sizes, balance, CMD and two-color scatter, changes in complete-distribution CRPS, and PDF mass at least 1 mag on either side of the reference. Spatial bootstrap keeps both groups and the B/C responses attached to their observed blocks. Cross-tabulate inherited reliable optical spectra and APOGEE classes against the Gaia proxies. The mass tails are distance diagnostics. Gaia CMD proxy groups alone cannot identify the physical luminosity branches.

## Independent scale inventory

Keep only NGC 6362, NGC 6723 and M55. The published public grid provides an imaging feasibility screen, not a membership audit. Compare exact Gaia IDs with membership probability greater than 0.90 in a fixed 0.5° aperture and preserve Type, photometry, QRF support and historical losses. An independent prospective comparison requires at least 50 supported history-clean members, including at least 20 fainter lower-sequence proxies with (20le r<22). The proxy requires (M_{r,0}ge5) and ((g-r)_0ge0.40). Literature distance moduli use the W1 RR Lyrae relation with an HST parallax zero-point, independent of the target-system Gaia weak parallaxes. The publication reports a sizable alternate calibration offset. Keep the published statistical errors and treat a common 0.10 mag as an additional calibration sensitivity for any passing candidate. [Dambis et al., Table 6](https://academic.oup.com/mnras/article/439/4/3765/1165609)

## Execution

Submit `scripts/run_faint_signed_20261120.sh` from the canonical Photdis checkout. The script requires a NERSC CPU allocation and stops on errors. Per-source arrays stay under `data/faint_signed_20261120`. The public-only cluster table can be reproduced locally with:

```sh
python3 scripts/faint_signed_clusters.py --public manifests/faint_signed_20261120/cluster_public_screen.csv
```
