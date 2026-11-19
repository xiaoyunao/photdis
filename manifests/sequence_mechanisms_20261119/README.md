# Five fixed sequence-mechanism controls

Baseline `20513877`; production `87069408` unchanged. This package separates
native-locus geometry, along-locus measure and early post-MS luminosity support
in M13/M5/M30. It does not edit the paper or the completed sky products.

**The result is mixed:** matched loci and early post-MS support improve M5/M13,
whereas M30 does not recover the same scale. IMF-only effects are small. All
six lower-MS/turnoff E groups on common support still miss the registered
0.10 mag absolute-scale target. No new production correction is adopted.

- [Scientific results and decision](science_decisions.md)
- [Figure index and definitions](figures.md)
- [Likelihood, mass measure, scoring and reproduction](../../docs/sequence_mechanisms_20261119_methods.md)
- [Registered comparison and model hashes](../../configs/sequence_mechanisms_20261119.json)
- [Original full review and prompt](../../docs/v7_inputs)

The frozen primary native-PSF parents contain 249/585/99 members; every source
has a status row for all five controls and QRF. Primary support, pairwise common
support and the unchanged v5 cohort are distinct. The main scale panel uses
the intersection of all five controls within v5, keeping identical stars at
every point. Original v5 lower-MS counts 166/397/32 and turnoff 37/85/32 remain
available, including unsupported states. A original/old-calibrated replay is
separate from the unshifted A–E comparison.

| Evidence | Contents |
|---|---|
| `prepare/` | Admitted identities, denominator, actual model intervals and measures |
| `predict/` | Exact legacy/QRF replay, retained quadrature trials and convergence |
| `curves/support_status.csv` | All 5,598 parent-by-method status rows |
| `evaluate/summary.csv` | Raw/centered scales, widths, inclusion, CDF, scores, CRPS, shared CIs and fixed color/Mg bins |
| `evaluate/paired.csv` | Every model pair on exactly common support; signed changes and shared 95% intervals |
| `evaluate/phase_gaps.csv` | Turnoff-minus-lower-MS scale, shared-reference uncertainty |
| `data/` | Compact portable plot arrays, 11 selected full-PDF examples and selection metadata |
| `validation/` | Original score/zero replay, direct half-grid comparison and v3.8 pixel regression |
| `release.json` | Source-bound output hashes, executed code, production/style/sky preservation |
| `delivery/` | Portable pixel checks and actual visual/PDF review |

`primary` reports the frozen parent with each method's support; `v5` retains
the original cohort with failures; `v5_commonABCDE` is the additional common
support display subset. A supported-only performance statistic is not an
accuracy estimate for its unsupported parent members.

Scores are means of per-star marginal prediction densities convolved with the
same reference measurement kernel, in modulus coordinates. They are not a
joint cluster likelihood. Higher log score is better; lower CRPS is better.
The reference is shared across each system's member bootstrap. Centering uses
the evaluated group and remains diagnostic. The main comparison has no new
zero fitted to B–E. Every original component and tail is retained privately.

Private full-member PDFs, shared draws, per-star scores, initial and final
exports, logs and source inputs remain on NERSC at
`/pscratch/sd/y/yunao/photdis_pipeline/data/sequence_mechanisms_20261119`.
The public compact arrays redraw all five plates without raw catalog access:

```bash
PYTHONPATH=scripts:src python scripts/seq7_figures.py \
  --source manifests/sequence_mechanisms_20261119/data \
  --out /tmp/photdis_v7_redraw
```

The selected sky overview remains B5–10/G10–20/R20–100 kpc, density sigma0.3°,
background sigma5°, with unchanged fixed gains, Galactic bands and dashed
Sagittarius/Pal5 tracks. Its PNG SHA256 remains
`411b0086e9d4c6f9f208805f02f064e2c24bb117d61293083b9bb51dc7faaed5`.
