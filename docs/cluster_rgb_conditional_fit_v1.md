# Frozen development fit: DECam conditional RGB v1

The fit ran on NERSC Slurm job 60497547, using only M3 lower-RGB, M53 upper/lower-RGB, and NGC 7006 upper-RGB Gaia members. The independent NGC 5466 Legacy photometry was not read. The model artifact is [the frozen manifest](../manifests/cluster_rgb_conditional_model_v1.json); it records the configuration SHA-256, input hashes, development-source hash, baseline coefficients, 48 shared-nuisance fits and the registered support cells.

| Model tier | Development stars | Supported upper RGB `M_g` | Supported lower RGB `M_g` |
|---|---:|---|---|
| gr, all gr-clean stars | 1,041 | −1.5 to +0.5 | +0.5 to +3.0 |
| gr, grz-common stars | 850 | none | +0.5 to +3.0 |
| grz, same grz-common stars | 850 | none | +0.5 to +3.0 |

The upper grz sample contains only 19 NGC 7006 stars distributed across the magnitude bins, so no upper bin meets the preregistered five-per-development-cluster support criterion. Gr-only gains an upper-RGB range; the gr/grz information comparison is limited to the same 850 development stars and the supported lower RGB. NGC 5466 will be scored only after this model is committed, using its registered phase screen, distance, abundance and prior sensitivity.

The baseline lower-RGB gr color model has a central metallicity coefficient `+0.0438 mag/dex` on the common training stars; the grz model's gr and rz coefficients are `+0.0417` and `+0.0375 mag/dex`. These are regularized conditional associations under the adopted RRab abundance and external cluster scale. The baseline grz intrinsic color covariance has gr/rz standard deviations about `0.0436/0.0288 mag`, with strong positive correlation. The 48 coherent cluster-scale draws encode additional model uncertainty. No per-cluster free intercept was fitted.

This experiment is not a general RGB distance calibration: a conditional color model is meaningful only on the registered overlap and with a specified abundance prior. In particular, the narrow upper-RGB grz overlap is an observed support limitation, not a reason to lower the minimum count after looking at the test system.
