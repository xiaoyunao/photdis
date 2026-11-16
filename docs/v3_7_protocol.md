# Photdis v3.7 scientific protocol

The development comparison uses the frozen v3.6 evidence at
`34422f1` and the [2026-11-16 Pro review](chatgpt-conversation://6af9bbfe-5288-83ea-9e15-f09b45daea62).
The scientific question is whether photo-only selection can retain a useful
2–10 kpc dwarf sample with quantifiable class contamination and distance
migration. All candidate work preserves the released QRF distance PDF and
requires verified native Legacy DR11 PSF morphology.

## Selection contrasts

The workpoints are frozen in [the configuration](../configs/v3_7_selection.json).
Keep primary and strict as anchors. Compare one relaxation in dwarf score,
one in nominal SED near probability, their combination, and their combination
at observed r < 18. These contrasts bracket the current thresholds. They are
development comparisons against known evidence. Mark every historical exposure
and require a separate claim basis for independent evidence.

Reuse computed nominal probabilities. For rows that can pass one of the
contrasts and have no stored probability, run the same nominal SED likelihood
and joint integration on a restricted reference and stratified catalog frame.
Quote measured reference retention and pass fractions with each source's
denominator. For catalog estimates, apply the sampling weights and show
median-bin counts and PDF masses. Keep a candidate only where retained far
dwarfs and the associated errors and leakage support a scientific use. Allow
the answer to vary across observed domains.

## Supporting comparisons

- Compare the frozen full classifier with a fit that omits sky coordinates and
  keeps the other features, labels, identities and training settings. Inspect
  retention across reference distance and sky bins. In the blue samples compare
  neighborhoods matched on both optical colors, optical errors and r. Use the
  same objects for paired i or WISE comparisons when those bands qualify.
- Join Gaia, SpecDis and SPDist by verified native identity. Compare distances
  on the triple overlap and pairwise when the triple has little support. Use
  each source's quality flags and preserve missing FeH and gravity separately.
- Evaluate every supported row with corrected signed Gaia S/N < 5 and QRF
  median distance in [5, 10) kpc. Keep the published primary and strict
  subsets, negative parallaxes and recorded history flags. Score QRF and the
  same broad-distance baseline in conditional parallax space. Report paired
  sky-block intervals and the fixed v3.6 scale, zero-point and far-odds grid.
- Vary the total giant/dwarf odds by 0.25, 1 and 4 at fixed other-branch
  odds relative to the dwarf. Keep the five v3.6 LF cases and verify known
  blue foreground and far-giant branches. Report hard-assigned and posterior
  weighted selection leakage separately.
- Use the same qualified non-QRF-training stars in principal performance,
  retention and migration panels. Annotate PDF colors and hard counts.
  Produce a photo-only use table with direct, conditional, known-failure and
  missing-evidence domains, and seven to eight figures in the established
  English figure style. Keep the v3.6 atlas accessible for the appendix.

## Reproducing the delivery

Run science stages from the canonical NERSC checkout in allocated compute
steps. The 120-CPU allocation uses separate Slurm steps for each diagnostic
and parallel steps for the missing nominal probabilities. The script resumes
completed stages from their receipts.

```bash
sbatch scripts/run_v3_7_diagnostics.sh
```

The separate weak Gaia, same-identity reference and LF-odds stages use
`scripts/v3_7_weak_gaia.py`, `scripts/v3_7_reference_comparison.py` and
`scripts/v3_7_lf_odds.py`. Their verified results live beneath
`data/v3_7` on NERSC. The published CSV projections retain their
complete-file hashes, and the derived composition table records the hash of
its source reference table. The six new plates can be rendered from the
committed inputs.

```bash
python scripts/v3_7_paper_figures.py --inputs manifests/v3_7
PYTHONPATH=scripts python scripts/v3_7_validate.py
```
