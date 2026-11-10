# v2.3 requirement ledger (execution in progress)

Authority: [complete Pro review](pro_review_83e1fe8.md), implementation contract:
[frozen protocol](v2_3_protocol.md). Latest job state is in WORKLOG/PLAN and
manifests/v2_3/execution/jobs.json. This document is not a completion claim.

| Pro requirement | Implementation / evidence | Status |
|---|---|---|
| Cache-only noise prior correction | noise_baselines.csv; uniform/design/per-class/macro; paired v2.2 changes preserved | Actual completed;2English correction plates checked |
| Separate training/evaluation/signed/weak reference qualifications | reference_inventory.csv;173chosen,3596Gaia=2952SNR<5+644SNR>=5 | Actual completed; no reserved predictions used |
| Freeze near scope before new-ID outcomes | PSF, native clean grz/high-latitude/low dust; r<20,q>.95,error<.1,common QRF support; WISE optional | Protocol frozen; production validation pending |
| Preserve missing-class distance support |266774original-owner PSF objects,43914unlabeled; exact source uncertainty join | Actual preparation complete, no luminosity cut |
| One error-aware multimodal candidate | Fixed XD32 in(g-r,r-z,Mg), per-source covariance/reference errors, analytic shared-g conditioning |6math/scoring tests pass; model61078462 running |
| Same qualification baseline | Same266774PSF identities and grz; unchanged QRF configuration; historical admission-effect contrast separate | Code complete; production pending |
| Freeze before reserve; no evaluation-to-training feedback | models_frozen hash checks, all42076new IDs evaluation-only; no calibration | No outcomes yet opened |
| Full signed Gaia and distance PDF scores | Full PDF checkpoints, observed-error likelihood/PIT, negative values; reference-convolved scores,68/95intervals | Production pending |
| Selection versus accuracy; conditions and completeness | qnear risk, chosen/reference source distances, magnitude/color/error/WISE/sky; selected quality, retained denominators | Production pending |
| Poor-metal mechanism | Same poor-metal objects across models; source-exact color/temperature/error matched ordinary controls; matching quality/counts | Production pending; descriptive, not causal |
| Far10–20 versus20–50 shape/weight separation | Oracle conditional distributions, source disagreement and class contributions; no global class multiplier | Production pending; existing teachers only |
| Actual-application scope and unvalidated weights | Existing26000weighted PSF sample only; same candidates and support; evidence-cell occupancy not reliability cut | Production pending; no whole catalog rerun |
| PSF nonstellar controls | Risk-flagged reserved/development controls through actual predicted cuts | Code complete; not representative population contamination |
| Independent18–22/noWISE, poor-metal scale,20–50giant evidence | Reserved IDs add some noisy distance/parallax evidence; no newly certified independent scale/system membership | Still outstanding; cannot certify by relabeling old teachers |
| Preserve old models, no automatic deployment/search/timer | Frozen old inputs; one candidate only; prespecified cal guards | Enforced; adoption not authorized |

The error-aware candidate follows the heteroscedastic Gaussian-mixture method
of [Bovy,Hogg & Roweis2011](https://arxiv.org/abs/0905.2979). Gaussian reference
errors are approximations, and reference selection remains embedded in the
learned density. It is not a prior-free likelihood for Galactic populations.
