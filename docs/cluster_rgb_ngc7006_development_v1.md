# NGC 7006 as a new DECam development system

This audit uses the cached gr-selected and same-row grz-selected DR11 member tables on a NERSC Slurm CPU allocation. NGC 7006's former role as the once-only test of the **old pooled A/B model** is preserved in that model's historical report. It is now development data for the newly registered conditional experiment.

| Upper RGB Gaia candidate | M53 | NGC 7006 |
|---|---:|---:|
| `gr_clean` | 246 | 56 |
| `grz_clean_same_row` | 156 | 19 |
| gr 5–95% `M_g` overlap | −1.007 to +0.471 mag | −1.007 to +0.471 mag |
| grz 5–95% `M_g` overlap | −0.296 to +0.513 mag | −0.296 to +0.513 mag |

The local gr color-residual NMAD is 0.073 mag for the 50 NGC 7006 stars inside its own binned curve; for the 11 grz stars inside their own curve it is 0.033 mag. These are color residuals relative to curves estimated from the same small sample, not independent distance errors. NGC 7006's gr fit-quality split (`RCHISQ_G+RCHISQ_R` median 105.6) gives residual centers +0.020 and −0.020 mag on its two sides. The radial halves differ by about 0.017 mag in center. These trends and the sparse grz locus require a shared-scale and quality sensitivity in the new fit; they do not justify a global new quality threshold.

The independent modulus uncertainty is 0.074 mag for the entire cluster. The published NGC 7006 reddening estimate has ±0.05 in `E(B−V)`. A coherent 0.05 shift changes `M_g` by 0.161 mag, gr by 0.052 mag and rz by 0.048 mag. Those are common cluster shifts, not independent errors on 56 stars.

The plotted upper-RGB sequence is a descriptive Gaia-CMD candidate sequence. The 19 grz objects cover a narrower common range with M53 than the 56 gr objects. The model will therefore report separate gr and grz support and score band gains on identical grz-qualified objects. Source data, detailed radial/quality halves and covariance values: [development manifest](../manifests/cluster_rgb_ngc7006_development_v1.json) and [curve figure](../figures/cluster_rgb/05_ngc7006_upper_rgb_development.png).
