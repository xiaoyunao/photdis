# Whole-cluster Legacy RGB calibration audit

## Design and status

The input is the full Vasiliev–Baumgardt Gaia membership list for each
registered cluster, joined by exact 64-bit `source_id` to DR11 `G3` Legacy
photometry. A member enters the high-probability phase audit at
`Pmember>0.9`. Gaia `G−μ` and observed `BP−RP` define broad phase-candidate
regions before any Legacy RGB fit. A second `0.5≤Pmember≤0.9` layer remains
available for sensitivity checks. The phase boundaries are in
[`configs/cluster_rgb_phase_regions_v1.json`](../configs/cluster_rgb_phase_regions_v1.json).
The published membership rows include Gaia parallax, proper motions,
their errors and `qflag`; they have no per-row radial-velocity column.
The published membership probability is the primary astrometric identity
evidence here. Existing DESI radial velocities and spectral classes remain
subset checks, not an eligibility condition for the full member sample.

Legacy colors use the project's measured DR11 dereddening
`m₀=m−R_m E(B−V)`, with `R_g=3.214`, `R_r=2.165`, `R_z=1.211`.
An RGB candidate's external absolute magnitude is `M_g=g₀−μ_cluster`.
All members of a cluster share its external `μ` uncertainty. This is a
measured optical sequence on Gaia-selected member candidates, without a
per-star DESI spectrum or SPDist label requirement.

| Cluster | P>0.9 upper/lower RGB candidates | Clean upper/lower | Clean RGB `M_g` 5–95% | Clean RGB W1+W2 |
|---|---:|---:|---:|---:|
| M3 | 387 / 579 | **2 / 422** | +1.14 to +2.76 | 9 |
| M53 | 357 / 313 | **156 / 253** | −0.19 to +2.73 | 22 |
| NGC 2419 | 458 / 93 | **391 / 78** | −1.79 to +0.89 | 10 |

The M3 upper RGB loses 385 of 387 high-probability candidates under the
existing `grz_clean` rule: 381 have base grz, only five pass the optical
fraction condition, six the mask condition and two pass both. Within one
arcminute of M3's center, ten of 141 candidate RGB stars remain clean.
The per-bit audit identifies `SATUR_Z` on 373 of 381 M3 upper-RGB base
records and z-band `FRACMASKED` failure on 373; causes overlap. For M53 upper
RGB, 158 of 354 base records carry `SATUR_Z` and 151 fail the z-band
`FRACMASKED` criterion; 156 remain clean. These counts locate a substantial
bright-end z-band loss. The [quality-cause manifest](../manifests/cluster_rgb_quality_causes_v1.json)
also breaks out `SATUR_G/R`, `ALLMASK`, each optical fraction criterion,
and the reserved system's quality counts.
NGC 2419 retains 391 of 458 upper-RGB candidates, but its lower candidate
region overlaps the red horizontal branch in the Gaia CMD and is a
sensitivity layer. The secure M3 and NGC 2419 luminosity ranges barely
meet. M53 supplies 156 clean upper and 253 clean lower candidates across
the gap. Its median map reddening is `E(B−V)=0.02078`.
The [registered roles](cluster_rgb_system_registration_v1.md) keep
NGC 7006 as a whole-system holdout. The reserved system has 84
high-probability upper-RGB candidates, 27 with base grz, 19 clean,
and 20 clean lower-region candidates; its Legacy
luminosities and colors remain sealed until the
[fixed transfer protocol](cluster_rgb_experiment_protocol_v1.md) and
development model are committed.

The existing WISE clean counts refer to the same clean RGB candidate stars.
Nine, 22 and ten two-band objects in M3, M53 and NGC 2419 are too few
for a standalone WISE RGB relation.

## Shared scale and dust

The external RR Lyrae moduli used here are M3 `15.041±0.017(stat)±0.036(sys)`
mag and NGC 2419 `19.71±0.08` mag. Their uncertainties shift all member
`M_g` values in that system together. The median DR11 map reddening for
the clean RGB candidates is `0.01347` and `0.06084`, respectively.
The cited NGC 2419 RR study reports `E(B−V)=0.08±0.01`; changing its DR11
map value by +0.01916 would move `M_g` by −0.0616 mag,
`(g−r)₀` by −0.0201 mag and `(r−z)₀` by −0.0183 mag.
This comparison is a shared dust-scale sensitivity, not independent
per-star scatter.

The Gaia phase diagram uses observed `G` and `BP−RP`; it is an approximate
evolutionary screen, particularly in NGC 2419. The RGB, HB and possible
AGB labels are candidates, and no model residual or photometric-distance
agreement is used to assign them.

## Evidence

- [`M3 phase manifest`](../manifests/cluster_rgb_m3_phase_v1.json)
  [`M53 phase manifest`](../manifests/cluster_rgb_m53_phase_v1.json)
  and [`NGC 2419 phase manifest`](../manifests/cluster_rgb_ngc2419_phase_v1.json)
  include member, optical fraction/mask, radius, WISE and color-bin counts.
- [`M3 phase figure`](../figures/cluster_rgb/02_phase_locus_m3.png)
  [`M53 phase figure`](../figures/cluster_rgb/02_phase_locus_m53.png)
  and [`NGC 2419 phase figure`](../figures/cluster_rgb/02_phase_locus_ngc2419.png)
  show the measured Legacy loci alongside Gaia phase candidates.
- [`NGC 7006 phase manifest`](../manifests/cluster_rgb_ngc7006_phase_v1.json)
  contains only Gaia-CMD candidate and DR11 quality counts.
- [`Development loci comparison`](../figures/cluster_rgb/03_development_loci.png)
  displays 0.5-mag bin medians and within-bin color widths for the three
  development systems. At overlapping magnitudes the loci differ, so a
  pooled photo-only curve requires a whole-system transfer check.

## Frozen low-degree transfer diagnostic

The preregistered cluster-balanced quadratic A/B experiment has now been
fitted to 422 M3 lower, 409 M53 upper/lower and 391 NGC 2419 upper RGB
candidates. The selected W1+W2 counts are 9, 22 and 10, respectively.

| Frozen model | M3 development `Δμ` median / NMAD | M53 | NGC 2419 |
|---|---:|---:|---:|
| A: `(g−r)₀` quadratic | +0.688 / 0.361 | −0.051 / 0.451 | −0.580 / 0.744 |
| B: `(g−r)₀,(r−z)₀` quadratic | +0.544 / 0.398 | −0.130 / 0.388 | −0.495 / 0.610 |

All quantities are mag and `Δμ=μ_pred−μ_external`. These are fitted-system
scores, so their large offsets already show a failure of a single
cluster-independent point relation under the chosen equal-cluster weights.
The same [frozen model file](../manifests/cluster_rgb_transfer_model_v1.json)
also records leave-one-system-out checks; for A, omitting M3 changes its
median offset to +1.201 mag. This model remains a diagnostic and is not a
field-star distance product.

### Once-only whole-system NGC 7006 check

The complete NGC 7006 system was scored **after** the training model and
protocol commits. The registered upper-RGB test contains 19 clean stars,
all inside both models' development color rectangles.

| Model | Holdout `Δμ` median | NMAD | `|Δμ|>0.5` | Source-bootstrap 95% median interval |
|---|---:|---:|---:|---:|
| A, primary `(g−r)₀` | +0.166 mag | 0.323 mag | 5/19 | −0.072 to +0.223 mag |
| B, `(g−r)₀,(r−z)₀` | +0.152 mag | 0.293 mag | 4/19 | −0.046 to +0.333 mag |

The 20 clean lower-region candidates, kept separate because of possible
red-HB overlap, have median offsets −0.117/−0.171 mag for A/B. The
registered upper-RGB sample contains **no** clean W1 or W1+W2 measurement,
so an IR same-star model comparison is unavailable in this holdout.

These are one-system scores with 19 sources, not a population-wide distance
precision. The external NGC 7006 modulus has ±0.074 mag uncertainty shared
by all 19 stars. In the fixed A sensitivity experiment, perturbing the
NGC 2419 development reddening by ±0.020 shifts the NGC 7006 median by
−0.111/+0.076 mag; perturbing the NGC 7006 reddening by ±0.010 shifts it
by about ±0.112 mag. The corresponding development NGC 2419 modulus
perturbation of ±0.080 moves the median by ±0.066 mag. These correlated
scale effects are distinct from the 0.323 mag source-level NMAD.

The [whole-system score manifest](../manifests/cluster_rgb_transfer_holdout_v1.json)
retains both model outputs and all registered scale sensitivities. Because
the development systems already have large, opposite center offsets and
the holdout is small and scale-sensitive, **neither quadratic is adopted**.
The measured Legacy sequences remain the useful result of this round.

The earlier frozen warm-star grz experiment remains a separate diagnostic:
its five qualified M3 members had a +0.367 mag median modulus offset to
the independent M3 scale. Its 5000–7000 K target is not interchangeable
with the Gaia-selected RGB branches used here, and it did not select or
reject any of these cluster members. The frozen cold-v4 relation describes
the cold-dwarf hypothesis and was not applied to RGB members.
- External scales: [M3 RR Lyrae](https://arxiv.org/abs/2008.11745),
  [NGC 2419 RR Lyrae](https://arxiv.org/abs/1011.5398).
