# v7 fixed sequence-mechanism experiment

This is a development experiment on already studied systems, registered in
[`sequence_mechanisms_20261119.json`](../configs/sequence_mechanisms_20261119.json).
It preserves production `87069408`, the original full PDFs, every sky display,
and the Pal5/v3.8 rendering sources. It does not edit the manuscript.

## Scientific comparison

The unchanged low-crowding, role-excluded, unmasked native-PSF parents contain
M13/M5/M30 = 249/585/99 members. Missing historical predictions do not remove
members from this parent. Every method has a status row for every parent member.
Performance tables distinguish the full parent, each pair's common support,
and the original v5 cohort. The scale panel additionally intersects the latter
with the support of all five controls, so every point uses the same stars.

| Control | Native locus | Along-locus measure | Evolutionary support |
|---|---|---|---|
| A | Original FOS MIST1.0, 12 Gyr, FeH −2.17 | Original grz arc length | Phase 0 |
| B | Archived MIST1.2, same age and FeH | Same arc rule | Phase 0 |
| C | Archived system-matched MIST1.2 | Same arc rule | Phase 0 |
| D | Exactly C's physical nodes | Kroupa initial-mass cell integrals | Phase 0 |
| E | D plus connected early post-MS | Same initial-mass integral | Phase 0 and connected phase 2 |

All controls use native 3 ≤ Mr ≤ 8, the frozen north transformation/support,
the same measured magnitudes/errors/extinction, a 0.001 mag error floor,
0.10 mag gray scatter, and a flat distance-modulus measure. The MIST1.2
tables contain no phase 1. E includes the bridge from the last MS row to the
first phase-2 row; the disjoint bridge interval is assigned to post-MS.
Actual EEP, initial-mass and native magnitude endpoints are in `model_measures.csv`.
No brighter SGB/RGB support is implied.

D/E integrate the continuous [Kroupa (2001) IMF](https://doi.org/10.1046/j.1365-8711.2001.04022.x)
over each subdivided initial-mass interval: slope 1.3 below 0.5 solar masses,
2.3 above, with continuous normalization. They do not count table rows or
multiply by an additional lifetime factor. This is a fixed population
sensitivity measure; a real cluster's mass function and member selection
can differ. C and D have identical nodes, likelihoods and support.

The system label chooses a predeclared age/FeH curve. Reference distance,
reference absolute magnitude and phase boxes enter evaluation only. These
known-system conditional controls do not define an application-time field-star
mixture. Age/FeH vary together; fixed alpha/rotation and north transfer remain
part of the tested model, not independently identified causes.

## Normalized likelihood and evaluation

For node j, integrate the common gray amplitude analytically. The conditional
modulus is Gaussian with inverse-variance weighted mean of m−M_j and variance
1/sum(1/s_b²)+0.10². Its weight is the node prior times the two-color Gaussian
likelihood. Shared-r color covariance is negative. Keep every normalized
component, including color-multimodal and distant-tail solutions. Support uses
the unchanged minimum color chi-square ≤25 rule. Quantiles are analytic-CDF
inversions; the exact old A4/Sobol128 quantiles remain a separate legacy replay.

Frozen QRF uses the historical float64 `mag_grz/error_grz/identity`, its original
full piecewise-uniform modulus PDF and quantiles. Rounded display columns never
enter its replay. Historical model/training roles remain excluded.

Let p_i(mu) be a normalized prediction and R_s(mu−mu_ref) the common Gaussian
reference measurement kernel, whose modulus sigma comes from the frozen
system-common distance draws. The marginal score is

`log [(p_i * R_s)(mu_ref)]`, in natural logs per magnitude.

The reported mean of per-star marginal scores is not a joint cluster likelihood.
The same reference kernel is used for all five controls and QRF. CRPS is the
positive loss `integral [F_conv(u) − 1(u≥mu_ref)]² du` on that same modulus
coordinate; lower is better. Its definition follows the proper-score framework
of [Gneiting & Raftery (2007)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
No log-density floor or Gaussian replacement of a QRF PDF is used.

Each bootstrap replicate reuses the same whole-parent member multiplicities
and one system-common reference-center draw across methods and groups.
The v5 cohort reuses its original multiplicities. It is a parametric-observation
bootstrap of the scoring statistic: the same measurement-convolved forecast
is evaluated at the common perturbed observation. It does not broaden the
prediction a second time. Empty resampled diagnostic bins give missing draws,
not zero-valued statistics. Diagnostic group centering is recomputed in every
replicate and is explicitly conditional on the evaluated group. The original
M13 zero and its coupled reference draws enter legacy replay only.

Report raw/centered scores, asymmetric 68/95% widths, reference-center inclusion,
reference-convolved CDF inclusion, sample median/NMAD and shared 95% intervals.
Phase gaps and paired contrasts reuse the common draws. Fixed continuous color
and reference-Mg bins accompany the phase boxes. All support counts and actual
r/Mr/color ranges are in the tables.

## Numerical and graphical preservation

The registered 4→8→16→32 integration sequence is retained. The main common
resolution is selected by predeclared median, paired score and per-star
quantile tolerances, without choosing another model. Each normalized mixture
and QRF PDF stays on NERSC. A common 0.0025 mag grid encloses all components and
QRF edges, with at least ten Gaussian sigmas and the range 0–30 mag.

CDF summation loses at most one machine epsilon in total probability. For
log density, interval-specific component bounds limit relative omitted sums to
less than exp(−40), preserving components that dominate far tails. Original
parameters are never truncated. Cubic interpolation is used for log density;
CDF/CRPS interpolation is linear. Independent direct, unbounded mixture sums
on a half-step grid check these approximations. CRPS uses cumulative trapezoid
integration over the whole grid. Initial and refined outputs remain available.

The five new plates use the shared PAPER/Pal5/v3.8 settings and locked sources.
PDF examples follow the registered ID rules; separate small panels show the full CDF while
the main panel resolves the central density. Dotted curves are outside the
method's declared color support. Complete example parameters, full-grid curves,
original quantiles and reference kernels accompany them. They explain mechanisms;
aggregate claims come from the whole-member paired statistics.
The gray reference profile has its displayed height scaled to 80% of the largest
prediction peak; its width is unchanged. This prevents a narrow reference kernel
from flattening the prediction curves. Scientific scores use the normalized
reference kernel. Branch-case main windows follow D/E quantiles; separate full-CDF panels
retain the broader QRF support. Initial plot exports are archived separately.

## Reproduction

On allocated NERSC resources in `/pscratch/sd/y/yunao/photdis_pipeline`:

```bash
export PYTHONPATH=scripts:src:.deps:.deps/v39_dust
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/global/common/software/desi/perlmutter/desiconda/20260910-3.1.0/conda/bin/python scripts/seq7_run.py
```

`--from-stage` resumes a stage, retaining attempt logs. Completed prediction and
curve checkpoints are reused. Source catalogs, full-member component/PDF caches,
per-star scores and shared member draws remain under
`data/sequence_mechanisms_20261119` on NERSC. Compact evidence is published under
`manifests/sequence_mechanisms_20261119`; figures under the matching figure path.
Portable rendering, without any source catalog:

```bash
PYTHONPATH=scripts:src python scripts/seq7_figures.py \
  --source manifests/sequence_mechanisms_20261119/data \
  --out /tmp/photdis_v7_redraw
```

The predeclared practical target is an unshifted absolute group median modulus
residual ≤0.10 mag. Improvement and target attainment are separate conclusions.
This task stops after the five controls and a scientific decision; it does not
fit per-system corrections, tune a sixth model or replace production.

Pixel regression is enforced within the recorded Matplotlib/Pillow/FreeType/NumPy
build runtime. A cross-runtime redraw is checked and reported separately rather
than claiming pixel identity across rendering-library versions. PDF text/page
checks run locally with pypdf; the compute-side pixel replay uses `--pixels-only`.
Public per-source status rows explicitly carry TYPE=PSF/native verified against
the sealed identity roster; the deterministic enrichment is recorded in the
release receipt and does not change source admission or support.
