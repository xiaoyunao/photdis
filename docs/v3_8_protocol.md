# Photdis v3.8 science protocol

The starting point is the frozen v3.7 delivery at `afd25ff` and the
[latest Pro review](chatgpt-conversation://6af9bbfe-5288-83ea-9e15-f09b45daea62).
The published southern QRF and catalog provide the common comparison baseline.

1. Rebuild the paper figures around direct distances, same-star absolute CMDs,
   individual-coordinate Aitoff maps and heliocentric X–Y, X–Z and Y–Z views.
   Plot the three disjoint projected distance intervals 0–1, 1–5 and 5–10 kpc
   and the union 0–10 kpc. All points admitted to new science views require
   verified native DR11 PSF morphology. The footprint and the source-coordinate
   layer have distinct meanings.
2. Test the fixed 0.65 dwarf-score and 0.90 nominal SED near-probability
   candidate on the same weighted draw, with the frozen QRF PDF. Measure its
   own signed Gaia predictive distribution and LF scenarios. Use the independent
   per-source quality definitions on the matched reference frame. Production
   adoption requires a scientifically interpretable population and migration
   comparison before completing previously unevaluated nominal probabilities.
3. Decompose the correct and alternative distance branches in fixed
   representative examples. Use the native forward flux model when comparing
   likelihood, prior and posterior. Any alternative branch representation
   must be tested against unchanged controls.
4. Evaluate reference-error-aware interval calibration with separate fitting
   and test identities. Keep the hard-list model leakage, QRF PDF expected
   count and QRF–SED cross-weighted diagnostic distinct. Report stellar labels,
   Redrock evidence and the qualified reference coverage of validated domains.
5. Pilot DR11 north in its native BASS/MzLS optical system and compare matched
   north–south objects. Survey documentation specifies its own natural
   passbands and products. [DR11 data model](https://www.legacysurvey.org/dr11/files/)
   · [DR11 photometry and coverage](https://www.legacysurvey.org/dr11/description/)

Formal AAS editing follows the user's separate confirmation. The working text
and verified paper results can be assembled while science jobs run.

## Execution and products

The NERSC checkout runs the direct figures with
`sbatch scripts/run_v3_8_figures.sh`. The comparison wrapper accepts selected
stages as arguments. The north report reads the completed north pilot receipt.

```bash
sbatch scripts/run_v3_8_science.sh candidate reliability branches redrock north
sbatch scripts/run_v3_8_science.sh candidate_index north_report
```

The science outputs and full-coordinate inputs stay beneath `data/v3_8` in
the NERSC checkout. Each finished stage has a checksum receipt. From a staged
copy of the compact products, use `scripts/v3_8_collect.py` to verify them.
Local-rendered figures use the plot inputs named in their table receipts.

```bash
python3 scripts/v3_8_collect.py /path/to/staged/v3_8 --verify
```
