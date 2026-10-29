# Whole-cluster Legacy RGB calibration audit

## Design and status

The input is the full Vasiliev–Baumgardt Gaia membership list for each
registered cluster, joined by exact 64-bit `source_id` to DR11 `G3` Legacy
photometry. A member enters the high-probability phase audit at
`Pmember>0.9`. Gaia `G−μ` and observed `BP−RP` define broad phase-candidate
regions before any Legacy RGB fit. A second `0.5≤Pmember≤0.9` layer remains
available for sensitivity checks. The phase boundaries are in
[`configs/cluster_rgb_phase_regions_v1.json`](../configs/cluster_rgb_phase_regions_v1.json).

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
NGC 2419 retains 391 of 458 upper-RGB candidates, but its lower candidate
region overlaps the red horizontal branch in the Gaia CMD and is a
sensitivity layer. The secure M3 and NGC 2419 luminosity ranges barely
meet. M53 supplies 156 clean upper and 253 clean lower candidates across
the gap. Its median map reddening is `E(B−V)=0.02078`.
The [registered roles](cluster_rgb_system_registration_v1.md) keep
NGC 7006 as a whole-system holdout. The reserved system has 19 clean
upper-RGB candidates and 20 lower-region candidates; its Legacy
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
- External scales: [M3 RR Lyrae](https://arxiv.org/abs/2008.11745),
  [NGC 2419 RR Lyrae](https://arxiv.org/abs/1011.5398).
