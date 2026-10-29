# Whole-system RGB development and reserved roles

This role registry was written after the M3/NGC 2419 Gaia-CMD coverage
audit and before reading M53 or NGC 7006 DR11 member photometry. Selection
uses sky position, independent RR Lyrae distances and the measured lack of
bright/faint RGB overlap between the first two systems. No distance-model
residual enters system choice.

| System | Role | External distance | Reason |
|---|---|---|---|
| M3 | Development diagnostic | RR Lyrae `μ=15.041±0.017(stat)±0.036(sys)` mag | Existing close low-dust cluster; clean lower RGB, very little clean upper RGB. |
| NGC 2419 | Development diagnostic | RR Lyrae `μ=19.71±0.08` mag | Existing distant cluster; clean bright RGB, E(B−V) map ≈0.06. |
| M53 | New development bridge | NIR RR Lyrae `μ=16.403±0.024(stat)±0.033(sys)` mag | Intermediate distance and independent scale may bridge the M3/NGC 2419 Legacy luminosity gap. |
| NGC 7006 | **Reserved whole-system test** | RR Lyrae `d=41.2±1.4 kpc`, equivalent `μ=18.074±0.074` mag | Intermediate distant system with an external scale; reserve entire cluster before any fit. |

For NGC 7006, inspect membership, Gaia CMD, DR11 match and clean-image
coverage only. Keep its Legacy RGB luminosity sequence and model residuals
unread until a candidate method and scope are frozen. If the reserved
system has too few clean Gaia-CMD RGB candidates, report the measured
shortage and register a replacement system before inspecting model offsets.
The M53 and NGC 7006 reference distances are separate measurements; all
members of a system share that system's distance uncertainty.

The second pair was extracted on allocated Slurm job `60493137` after this
role registration. M53 has 9,512 `Pmember>0.9` DR11 matches, of which 6,550
pass clean grz. NGC 7006 has 568 and 145, respectively. The NGC 7006
inventory plot displays only its independent Gaia CMD and clean-coverage
overlay. Broad Gaia CMD phase boundaries for both systems were then
registered in `configs/cluster_rgb_phase_regions_v1.json` before fitting
or opening either system's phase-specific Legacy luminosity relation.

References: [M3 RR](https://arxiv.org/abs/2008.11745),
[NGC 2419 RR](https://arxiv.org/abs/1011.5398),
[M53 NIR RR](https://arxiv.org/abs/2012.13495),
[NGC 7006 RR](https://arxiv.org/abs/2212.05021).
