# Native CMD filter control

Fixed setup in `configs/paper_v2_filter.json`. M30 supplies DECam and M92
supplies BASS/MzLS. M30 uses archived low-crowding native DR11 PSF members.
The M92 strict member-only roster has fewer than 20 histogram inputs.
Before evaluating either test field, the M92 construction was switched
to the v3.16-style native target minus control Hess, with fixed depth-22
unmasked random area and nonnegative cells. Training and calibration
identities are excluded.
Approximate ages are 13.25 Gyr and metallicities −2.3 dex. The empirical
absolute magnitudes use the fixed 8.49 and 8.50 kpc scales. The main-sequence
support is restricted to the template magnitude bins. The shifts span
modulus 5–16 (0.10–15.85 kpc), so the filters test both displayed distance
layers. In the nearby layer only combinations of distance and apparent
magnitude with overlap in the empirical CMD contribute signal.

The test subtracts the effective-area-scaled control Hess, smooths the member
CMD with predeclared widths that include representative measurement errors,
and computes the normalized Gaussian variance-weighted contrast at every
shift. M5 uses disjoint quadrants of its existing field and its own native
random areas. GD-1 uses the DESI-aligned and independently offset archived
ribbons with common usable segments. Both compare the target with the same
sky and depth cuts in the off-target regions. The search carries a one-sided
Bonferroni correction across shifts, templates' width settings, depth cuts
and systems. A positive score additionally faces the held-out blank score
and the stability rule. These are tests on previously studied control
structures using a new fixed control design. They do not constitute blind
discovery of an unknown structure.

Age/metallicity context is from the published cluster analyses.
[M30](https://academic.oup.com/mnras/article/499/2/2390/5903728) ·
[M92](https://doi.org/10.1088/0004-637X/775/2/134)
