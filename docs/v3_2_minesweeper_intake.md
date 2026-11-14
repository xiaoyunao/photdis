# DR20 BOSS MINESweeper acquisition

The frozen DR19 APOGEE intake admitted zero objects. DR20 adds BOSS
observations from Apache Point and Las Campanas, while its APOGEE
release repeats the DR19 data. The DR20 MINESweeper value-added catalog
targets distant and metal-poor halo stars, making its native-PSF overlap
worth measuring for the 2–5, 5–10 and 10–50 kpc objectives.
[DR20 MWM release](https://sdss.org/dr20/mwm/) ·
[MINESweeper catalog and methodology](https://www.sdss.org/dr20/data_access/value-added-catalogs/?vac_id=10022)

The acquisition job 61267444 downloaded `minesweeper_v1.2.2.fits` and
completed 0:0. It verified the published byte length (76,224,960) and SHA-1 checksum
`b905615b4b4c08c195e5efd5c644c3d46644e7e1` before publishing the
archive path and recording SHA-256
`65bab7406656d17d335beb69ee56bcd7499cb2130fc96b783f27ac7bb0a30b15`.
[Science Archive directory](https://data.sdss.org/sas/dr20/vac/mwm/minesweeper/) ·
[FITS data model](https://data.sdss.org/datamodel/files/MWM_MINESWEEPER/minesweeper.html)

The first census selects LCO rows with at least one exposure, positive
Gaia DR3 identifiers and observing MJD no later than the DR20 cutoff
60708. It reads six identity, coordinate and observation-provenance
columns. It checks exact Gaia DR3 IDs, sky history and
known-system exclusions, then applies the frozen Legacy DR11 primary,
release, TYPE=PSF, imaging history and photometric gates in the native join.
Keep the MINESweeper and DR19
receipts separate. The acquisition cannot establish native coverage.

MINESweeper fits BOSS spectra, photometry and Gaia parallax jointly.
Its distance and stellar-parameter outputs support a teacher comparison
under that selection. Parent contamination and independent absolute
distance calibration require other evidence. Keep the reference
outputs closed until the cohort, candidate and comparison rules are frozen.

For teacher scoring, require `FLAG=0` and finite positive heliocentric
distance and uncertainty on both sides. The Gaussian scoring scale is
the larger of the two errors, in kpc. The precise subset requires that
scale to be below 20% of the distance. For cool stars, assign dwarf
or giant labels only when the lower gravity bound exceeds 4.2 or the
upper bound is below 3.2, respectively. Keep the middle group as
unknown. Qualified metallicity uses finite values and positive errors.
Retain the separate signed Gaia channel and report the dependence of
MINESweeper on that same astrometric source.

Reference count checks use the same qualified rows as the reliability
bins. Report an exact Poisson-binomial interval under independent
Bernoulli draws at the frozen marginal probabilities. This interval is
conditional on those probabilities and omits shared model uncertainty.
Retain the sky-block interval on the mean count residual alongside it.
