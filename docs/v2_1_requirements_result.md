# v2.1 review requirement ledger

Source: [complete corrected Pro review](pro_review_602727c_near_probability.md).
Prespecified acceptance: [frozen protocol](v2_1_protocol.md), unchanged after production.
Scientific interpretation: [result](v2_1_result.md). The previous v2.0 inputs and atlas remain intact.

| ID | Implemented evidence | Status / interpretation boundary |
|---|---|---|
| A1 | near.py PSF admission; near_complete/information_admission/optical_probe_complete metadata; every NPZ admission/export validated | Done. Legacy nonPSF/unverified members only in exact frozen audit |
| A2 | distance_class.csv: all six distance bins, class counts, bias/NMAD/tails/coverage/width/interval score, separately for chosen/Gaia/SpecDis/SPDist | Done; no display clipping |
| A3 | migration.csv: actual prediction-only near selection; class, brightness, errors, WISE, sky, FeH and crossed strata | Done; reference-selection contamination is not application contamination |
| A4 | boundary.csv; probability_scores/reliability/risk_retention point-reference versus certain1sigma | Done; crossing and unknown-error rows retained |
| A5 | application.json, weighted existing catalog predictions, native PSF export | Done;13 DECam fields/8899334 parent, baseline_v1, not expanded_A or all DR11 |
| B1 | frozen_training_audit.npz; joint_reconstruction.json; JointReferenceForest | Done; exact317003 positions, class/source marginals and original training matrix verified |
| B2 | joint_freeze/chunks/complete,128 shared-g draws, full histogram mass and quantile reproduction | Done;763 chunks/97650 objects; failed first accumulation attempt diagnosed and explicitly migrated before accepted production |
| B3 | probability.npz, branch_attribution.npz/CSV, class_conditional.csv/summary, six deterministic examples | Done; joint reference contributions, not a physical posterior; no independent-marginal multiplication |
| C1 | full independent-color covariance, bounded64-candidate local_neighbors; matched and broad CSVs | Done;2000 matched per arm plus2000 broad. Candidate-limited, not information ceiling |
| C2 | shared-template target-noise tables with actual weighted errors/routes; weak_flux_inventory/probability using frozen signed-flux full PDFs | Done;149 common-route matched templates;2000 broad templates. Simulations use training templates, never new validation |
| C3 | information_scores and optical_probe: one fixed forest framework, no-color prior, controlled QRF, identical internal labels/evaluation/calibration | Done; original frozen partition-training confound disclosed; no family search |
| C4 | four same-label classification arms;4000 same-identity five-model band_probability full-PDF comparison | Done; matched grz/griz same original trainIDs; ABC distance training pools differ. Coverage explicitly reported |
| C5 | teacher full-density log score (zero-density events retained), interval scores/width/coverage; Gaia observation-space scores incl4135 qualified negative parallaxes | Done; teacher estimates, observed-parallax score and true distance accuracy separated. No double convolution of reference error |
| C6 | reliability/Brier/risk-retention at0.5/0.8/0.95; near_rule.json cal choice | Done; cal-selected0.5 fails dev5% contamination target.0.95 remains prespecified diagnostic, not a posthoc replacement rule |
| C7 | evolved contribution weights0.5/1/2; full near selection stress tables | Done; not measured prior or population count uncertainty |
| C8 | six distance bins/crossed quality strata, raw/domain/corrected FeH,12 bounded zero-point/dust/error scenarios | Done; sparse faint and new-sky physical evidence remain insufficient |
| C9 | exact DR1 primary zcatalog quality/position controls, nonstellar_complete/near CSV;49 old anomalies unchanged | Done for assessed pipeline controls; not visual confirmations or population-complete control. WD candidates not training labels |
| C10 | reserve.py excludes all historical train/eval/quarantine IDs and occupied sky tiles+neighbors before candidate evaluation | Reservation audit done: only5 IDs/2tiles. System/cluster membership and adequate independent references unavailable; independent final validation NOT completed |
| D | exactly one prespecified W1 class-partition candidate versus equal-ID/hyperparameter regression control; all class outcomes and six cal guards exported | Done, candidate rejected by cal all/evolved-tail guards. No further search or deployment |
| Figures | six groups/eight plates, each PNG300dpi/PDF/CSV/JSON, English text and inherited Pal5 style | Final format/hash and visual review recorded in delivery.json |

Remaining scientific evidence is explicitly itemized in result section8. This ledger does not turn an unavailable independent test into a passed check.

## Reproduction boundaries

Full data, join, model and probability programs run under the `run_v2_1_*.sh` Slurm wrappers.
NERSC path: `/pscratch/sd/y/yunao/photdis_pipeline/data/v2_1`.
Do not resubmit completed stages just to recreate this report. Full candidate PDF caches and models remain there.
Local compact rendering: `python3 scripts/v2_1_figures.py`.
Focused validation: `PYTHONPATH=src python3 -m pytest tests/test_probability_v2_1.py tests/test_information_v2_1.py tests/test_partition_v2_1.py -q`.
Final compute artifact validation: `scripts/run_v2_1_final_validate.sh`; frozen inputs must still match, no stale checkpoint reuse.

No timers, subagents, model-family searches, catalog reruns, automatic northern expansion or replacement deployment were added.
