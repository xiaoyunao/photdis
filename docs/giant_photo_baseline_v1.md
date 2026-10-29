# First bounded photo-only evolved-star experiment

The [registered protocol](giant_photo_protocol_v1.md) used only dereddened
Legacy `(g-r)0,(r-z)0` for inference. Its spectroscopic RVS giant class and
SPDist luminosity label defined the reference sample; they were never
prediction inputs. Existing eleven-field development/reserved identities and
the whole M3 system remained separate. The selected conditional Gaussian
kernel mixture used color bandwidth 0.06 mag and luminosity floor 0.15 mag,
chosen by leaving out complete development fields. These are the lower
edges of the registered grid; no further search was made after evaluation.

| Check | Result |
|---|---:|
| Development / reserved objects, 5000–7000 K | 1,403 / 588 |
| Reserved in color hull with effective training count ≥20 | 576 / 588 |
| Reserved median predicted minus SPDist `M_g` | −0.060 mag |
| Reserved residual NMAD | 0.464 mag |
| Reserved `|ΔM_g|>1` fraction | 5.3% |
| Reserved 68% / 95% interval coverage of SPDist point labels | 81.1% / 98.6% |
| Same-star predicted minus SpecDis `M_g`, 538 pairs | +0.171 mag median; 0.542 mag NMAD |

The wide reserved intervals exceed nominal coverage of the **SPDist point
labels**. This does not measure coverage of true geometric distance.
The color relation mixes distinct evolved populations: reserved `logg<2.5`
objects have +0.285 mag median residual, while `logg≥2.5` objects have
−0.227 mag. The 5000–5500 K subset has 0.524 mag NMAD, compared with
0.267 mag at 5500–7000 K. Metal-poor objects have 0.566 mag NMAD. Each
comparison is conditional on the spectroscopic reference selection.

The independent M3 RR Lyrae modulus is `15.041±0.017(stat)±0.036(sys)` mag.
All five qualified M3 members are inside the training color hull with high
effective training counts. Their photo-only median is **15.408 mag**, or
**+0.367 mag** (+18.4% in distance) above the RR Lyrae scale. Four of five
individual nominal 68% intervals and all five 95% intervals include the
reference because the intervals are broad. M3 catalog medians are much
closer to the RR scale: SpecDis +0.041 and SPDist +0.078 mag. Thus the
photo-only color relation, rather than a uniform M3 catalog-distance shift,
is the main observed discrepancy in this whole-system check.
The development-label neighborhood at each M3 color has weighted median
`M_g≈1.19–1.23`, while the five M3 RR-based absolute magnitudes are
`1.57–2.12`. Roughly 29% of each color neighborhood's training weight is
at `logg<2.5`. This directly locates the offset in the conditional
color–luminosity mix; the present experiment cannot assign it uniquely to
gravity, metallicity, age or the parent catalog's selection.

**Decision:** retain this run as a reproducible diagnostic and do not adopt
its distance estimates or intervals as a calibrated giant product. Optical
grz colors alone do not resolve the RGB/subgiant mixture in these data, and
the independent M3 center check fails. No correction is fit to M3. The
cooler K giant branch is separately constrained by the NGC 2419 audit:
its 13 SPDist labels have pooled median near the RR reference, but the three
`Teff<5000 K` members differ by −0.543 mag median.

The DESI-first ranking of previously untouched complete DR11 10° cells
identified 227 prospective clean low-dust 4200–5000 K RVS giants in two
southern cells (RA 200–210 and 190–200, Dec 0–10), and 59 in a separated
northern cell (RA 210–220, Dec 40–50). These are **pre-DR11-match counts**.
The three cells were registered as two development fields and one reserved
field before inspection. Their actual clean Legacy/SPDist/SpecDis yield will
determine whether a small cool-giant experiment is feasible. Existing cold
v4 model, XD decision and historical final-test identities remain frozen.

Evidence: [model manifest](../manifests/giant_photo_baseline_v1.json),
[field ranking](../manifests/desi_cool_giant_field_ranking_v1.json),
[diagnostic figure](../figures/giant_vac/03_warm_photo_baseline.png),
[M3 color-neighborhood check](../manifests/giant_m3_color_diagnostic_v1.json),
[M3 system audit](../manifests/giant_m3_reference_v1.json), and
[NGC 2419 system audit](../manifests/giant_ngc2419_reference_v1.json).
