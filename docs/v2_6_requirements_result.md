# v2.6 requirement and limitation ledger

Source: [full Pro review](pro_review_39e6e4f.md). Frozen implementation scopes:
[protocol](v2_6_protocol.md), [luminosity design](v2_6_luminosity_design.md),
[numerical-only amendment](v2_6_luminosity_numerical_amendment.md).

| Requirement | Executed artifact / evidence | Interpretation and limit |
|---|---|---|
| A: per-object class×distance joint posterior | 192 `data/v2_6/observed/*.npz`, 20 components, exact original IDs/observations | Five discrete nodes; broad/multiple branches retained; no continuous-distance claim |
| A: actual-catalog counts with common population parameters | 8192 joint count draws/catalog, full count covariance; `observed_counts.csv` | One shared theta per draw, then conditional per-star labels; not population×N and not independent marginal draws |
| A: separate population truth and realized counts | `observed_summary.csv`, `uncertain_population.csv`; figure01 | Same catalog identities for QRF point/PDF and new counts; >50kpc QRF tail included |
| B: known response, same template, independent transfer | 108 restricted catalogs; `response_controls.csv`; figure02 | Three designs only; 12 repeats each; same-template independent response MC truth is approximate |
| B: reachability and interval endpoints | `reachable_range.csv`, `reachable_truth.csv`, `interval_endpoints.csv` | Eight odds corners plus scalar numeric tilt extrema; all 80 individual-bin truths inside; not simultaneous full-vector proof or certified interval arithmetic |
| B: preserve four-parameter family and fixed priors | Original v2.5 weights and [-3,3]^4 bounds | Fixed-point Bayesian coverage need not be exactly95%; known-response control supports migration diagnosis |
| C: poor/other cal/test support | `template_shapes.csv`, `template_color_luminosity.csv`, `response_cell_residuals.csv`; figures04/08/09 | Includes joint color-Mg and observed-cell differences; outside-display mass retained |
| C: more independent sources, separate noise MC | Nested256/768/2048, same ID-hash order; `nested_response.csv` | 256 versus128 noise draws per source; replicates are not independent stars; missing-class soft kernel still uncalibrated |
| C: response uncertainty enters main inference | 16 finite Bayesian-bootstrap-prior responses, evidence-weighted `uncertain_grid.npz`, 36 `uncertain_observed/*.npz` | Uses joint target likelihood, not equal normalized posteriors; finite response prior and ESS/convergence limits reported |
| C: do not infer calibration from wider errors alone | Parameter and count coverage separately; `response_integration.csv` | Remaining undercoverage, finite-prior and template-transfer limits prevent scientific acceptance |
| D: cal-role signed Gaia luminosity location/scatter | `luminosity_refined/luminosity_posterior_{source}.npz` | Source-conditioned VAC scale experiment, not direct QRF refit; full signed likelihood, no inverse weak parallax |
| D: source scale / selection | SpecDis and SPDist separate, ordinary source-offset nuisance | VAC and Gaia dependence/selection not fully modeled; no universal physical scale claim |
| D: real identity and reused-control isolation | All2055original pair groups excluded from cal;1024cal/source;2049SpecDis and1185SPDist eligible pairs | No cal/evaluation or fitted-identity overlap; duplicated ordinary controls bootstrap by group |
| D: same-object likelihood, PIT and intervals | `luminosity_changes.csv`, `luminosity_distance_agreement.csv`, `luminosity_QRF_same_objects.csv`; figures06/07 | Gaia predictive coverage distinct from teacher-point distance agreement; full per-object table remains on NERSC |
| D: unknown metallicity latent at application | Fixed grz empirical-kernel joint branch draws, all four groups, uncertainty in source/poor shift and scatter | Application does not read evaluation FeH/class; teacher-conditioned and pure-photometric checks kept separate; no deployment |
| Numerical reliability | Independent adaptive-convolution tests; original/fine luminosity grids retained;32/64-node checks | SPDist coarse zero-width interval repaired; SpecDis residual quadrature discrepancies reported, not assumed exactly zero |
| Preserve accepted diagnostics | Prior red/metal/band audits untouched | No repeat red pipeline or reinterpretation of historical sample as representative parent |
| W1 | Deferred as review requested | No unrequested W1 experiment or model-family search |
| Prospective identities | Frozen3250reservation hash unchanged; outcomes/predictions unopened | No new final blind/sky validation claim |
| Delivery | English PNG300dpi/PDF + CSV/JSON, actual artifact tests and visual review | Passing engineering checks does not imply scientifically calibrated population intervals |

## Boundaries kept explicit

The population inference and the source-conditioned luminosity trial are separate
controlled experiments, not one validated new deployed model. The discrete
response family omits the intermediate metallicity range. Repeated catalogs share
224 real test SEDs, and the largest calibration library has2048 real identities.
Reference teachers, missing-label assignment, finite response prior, simplistic
photometric noise/selection and source dependence remain. No full DR11 rerun,
prospective outcome release, timers, agents or replacement of the existing model.

Initial failure61129243 was an output-directory writer error before any posterior
product was saved; explicit source-hash migration preserves frozen inputs.
Corrected population61129359 and luminosity61129363 completed. Initial validation
61129443 passed1639checks. Luminosity-only numerical refinement61129820 completed
and passed1648checks; final delivery records the later compact-support export
validation separately. No silent overwriting of initial scientific evidence.
