# v2.3.1 correction requirements

Source: [complete Pro review](pro_review_569d6ff.md).
Contract: [frozen amendment](v2_3_1_protocol.md).
This corrects the previously inspected frozen experiment, not a new blind test.

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Correct shared-g sign | density_v2_3.conditional uses T=Mg+epsilon_g | Implemented; independent tests pass |
| Independent forward tests | Known SED unequal/equal errors; non-diagonal Gaussian mixture in native grz band space; mean/variance/weight/PDF/qnear |8density/scoring +2selection tests pass |
| Preserve fitted models/training/QRF | Journaled migration, immutable old archive; candidate-only run; byte hashes checked | Complete;8124production checks,3316correction/selection checks |
| Invalidate stale predictions despite unchanged model hash | Every candidate chunk requires inference version and density/runner source hashes | Implemented; old candidate archives retained |
| Recompute all candidate-dependent quantities and original cal guards | Development/cal/reserved/application full PDFs, mu,qnear,selection,scores,far diagnostics | Completed61081256 (9:32) |
| Explain correction by measurement error | inference_correction_by_error.csv, same identities before/after | Completed61081737 (1:05); first schema failure61081403 repaired |
| Replay why reserved objects were unused | Exact historical manifests and source qualifications; class/distance/other/excluded reasons | Completed61081737 (1:05); first schema failure61081403 repaired |
| Raw and justified selection-conditional Gaia scores | Error/source-dependent anchor admission, fixed non-parallax covariates; no globalSNR<20 | Complete;3502upper-SNR20 and94untruncated objects; original raw-cal guards unchanged |
| Forward simulation through selection | Direct full-PDF+measurement-noise draws; reapply distance admission; independent CDF halves and empirical-vs-analytic checks | Completed61081737 (1:05); first schema failure61081403 repaired; not real calibration evidence |
| Include SPDist new-ID near evidence | Per-source reserved_distance/risk and main report/plate | Main evidence, shared-teacher and unknown-class limits retained |
| Distinguish full far PDF from class tagging | Full probability within1mag and convolved score on identical reference objects, alongside class-conditional results | QRF leaf-joint and XD component-class approximation explicitly different |
| Preserve metal-poor pairs under strata | Original2055pairs/354unique controls; both/one/neither counts; pair-conserving alternatives | No causal or independent-repeat claim |
| Explain coverage denominator change | Same selections with dev-only and cal+dev evidence counts | No attribution of all coverage improvement to selection |
| English figures, report and original evidence |16PNG/PDF/CSV/JSON plates; retain prior buggy results as superseded audit | Complete;16English plates visually reviewed, representative PDF rendered; delivery validation linked below |
| No new model or blind-test claim | Same32components/179EM; already-viewed reserve; no deployment/search/timer/full-catalog run | Enforced |

Population targeting and correlations between spectroscopic/source quality and
observed parallax are not modeled by holding those covariates fixed. Historical
admission conditioning does not certify an independent distance scale or a
representative application population. Poor-metal scales and far branch weights
remain scientific questions; infer their status from corrected results only.

[Final delivery checks](../manifests/v2_3_1/delivery_validation.json) · [Visual review and hashes](../manifests/v2_3_1/visual_review.json)
