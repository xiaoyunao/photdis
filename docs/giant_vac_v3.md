# DESI giant distance resources: corrected identity and bounded use

## Exact same-star inventory

The first two inventory artifacts gave an apparent shortage of giant distance
labels. After deduplication the original program sorted the DESI target IDs
but used a Gaia source-ID array from before that sort. A direct catalog audit
found 3,636 exact SpecDis and 3,687 exact SPDist `(TARGETID, SOURCE_ID)`
pairs among 3,694 unique qualified RVS giant-evidence stars. SPDist repeats
168 of those target/source pairs, so the strict unique-pair join accepts
3,519. The corrected v3 program keeps both IDs aligned with the selected
rows; its pair identity is asserted before attaching labels.

| Sample | All qualified | Clean low-dust | Valid SpecDis | Valid SPDist | Both valid |
|---|---:|---:|---:|---:|---:|
| RVS conservative giant evidence | 3,694 | 2,487 | 2,217 | 2,044 | 1,852 |
| `Teff<4200 K` | — | 17 | 12 | 0 | 0 |
| `4200≤Teff<5000 K` | — | 135 | 108 | 42 | 30 |
| `5000≤Teff<7000 K` | — | 2,335 | 2,097 | 2,002 | 1,822 |
| `logg<1.8` | — | 62 | 36 | 8 | 4 |

The clean sample requires Legacy `grz_clean`, E(B−V)<0.03 and no conflicting
CMD/spectral evidence. Its original RVS giant label requires a reliable
spectrum and `logg+sigma_logg<3.2`; it includes some warm evolved or
subgiant objects. Temperature and gravity select evaluation strata but are
not photo-only inference features.

## Distance and luminosity definitions

SpecDis v2.1 predicts distance directly from DESI spectra. For valid
positive `DIST` and `DISTERR`, `BINARY_FLAG=1`, and finite Legacy `g0`, set

`M_g,Legacy = g0 - 10 - 5 log10(DIST/kpc)`.

Its first-order uncertainty proxy is
`sqrt(sigma_g² + [5 DISTERR/(ln(10) DIST)]²)`. The sample's median
`DISTERR/DIST` is 0.491; the median proxy in `M_g` is 1.067 mag. Because
those distance errors are often large, the Gaussian/linearized proxy is
only a quality summary, not a full posterior transformation.

| DESI spectral S/N | Valid clean SpecDis giants | Median `DISTERR/DIST` |
|---|---:|---:|
| <20 | 645 | 0.594 |
| 20–60 | 1,421 | 0.471 |
| 60–100 | 123 | 0.281 |
| ≥100 | 28 | 0.140 |

SPDist v1.0 predicts Gaia `M_G` using spectroscopic atmospheric parameters,
iron abundance and Gaia colors. Require its `FLAG_GOOD` and `QFLAG_INPUT`
plus finite `G0`, `MG_16`, `MG_50`, `MG_84`. For a Legacy-band label use

`M_g,Legacy = g0,Legacy - (G0,Gaia - MG_50,SPDist)`.

The median label uncertainty is 0.209 mag after combining Legacy `g`, Gaia
`G`, half the SPDist 16–84% range and the catalog's recommended 0.167 mag
method precision in quadrature. The exact label distribution need not be
Gaussian; the first baseline treats this width as an approximation and
checks an independent system. Only one SPDist row among 2,045 otherwise
quality-flagged clean matches fails the additional finite/range checks.

Among 1,852 clean stars with both valid labels,
`M_g,SpecDis - M_g,SPDist` has median −0.207 mag and NMAD 0.397 mag. In the
4200–5000 K group, 30 paired labels have median −0.346 mag. Both VACs use
Gaia information during training, and their individual predictions share
stellar measurements. These offsets are a method comparison, not an
independent validation or a justification for a universal correction.

## External system checks and next experiment

The whole M3 system is isolated from the eleven development/reserved fields.
Five qualified DR11 members have an independent RR Lyrae modulus of
`15.041±0.017(stat)±0.036(sys)` mag. Their median catalog-minus-RR moduli
are +0.041 mag (SpecDis) and +0.078 mag (SPDist), with five stars in each
comparison. They are all near 5300 K, so this checks the warm evolved
regime only.

NGC 2419 provides a second, cooler whole-system check against an independent
RR Lyrae modulus of `19.71±0.08` mag. All 1,563 probable members have DR11
north-sweep counterparts. Eighteen have qualified RVS giant spectra and 15
have clean grz; their measured `E(B−V)` is 0.058–0.061, higher than the
`<0.03` development selection. Thirteen of the clean giants have valid
SPDist labels and five have valid SpecDis labels. For the 13 SPDist giants,
the catalog-minus-RR modulus median is +0.007 mag (NMAD 0.216 mag). The
temperature split is consequential:

| NGC 2419 SPDist subset | Stars | Median `μ_cat−μ_RR` | NMAD |
|---|---:|---:|---:|
| `5000≤Teff<5200 K` | 10 | +0.030 mag | 0.194 mag |
| `Teff<5000 K` | 3 | −0.543 mag | 0.053 mag |

The three cooler members have individual offsets +0.116, −0.543 and
−0.579 mag; their published membership probabilities are all >0.997.
The five SpecDis labels have median offset −0.525 mag and NMAD 2.516 mag.
This is an external scale warning for cool giants, with only three cool
SPDist members; the pooled 13-star median does not establish a calibrated
K-giant label. The individual catalog errors, extinction, temperature and
membership evidence are retained in the manifest. The registered warm
photo-only experiment will assess M3, and will report NGC 2419 only as an
out-of-selection diagnostic.

The registered grz-only warm evolved-star experiment is described in
`docs/giant_photo_protocol_v1.md`. The cold RGB/RC data are tabulated here
but are not yet a calibrated photo-only K-giant distance relation.

Evidence: `manifests/giant_vac_v3.json`,
`manifests/giant_vac_profile_v3.json`,
`manifests/giant_vac_quality_audit_v3.json`,
`manifests/giant_m3_reference_v1.json`,
`manifests/giant_ngc2419_reference_v1.json` and
`figures/giant_vac/02_ngc2419_catalog_scale.png`.

Primary catalog descriptions: [DESI SpecDis](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-specdis/)
and [DESI SPDist](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-spdist/).
M3 distance: [near-infrared RR Lyrae study](https://arxiv.org/abs/2008.11745).
NGC 2419 distance: [RR Lyrae study](https://arxiv.org/abs/1011.5398).
