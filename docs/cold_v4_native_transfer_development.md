# New-field native transfer: development checkpoint

The original four 10° fields and two supplemental development fields were
registered before their source rows were read. Each was built on a NERSC
allocated CPU node using the unchanged DR11, Gaia and optical pipeline.
The native scores use the frozen cold-v4 forests, support, color box and
`grz_clean`+WISE S/N≥20 route. Independent final-field residuals have not
yet been scored. The small field manifests and development scores are in
`manifests/field_native_*.json`,
`manifests/cold_v4_native_transfer_development.json` and
`manifests/cold_v4_native_transfer_supplemental.json` and
`manifests/cold_v4_native_transfer_supplemental2.json`.

| Development field | Strict Gaia anchors | Cold anchors | Selected | Δμ median | NMAD | Sky-block 95% median range |
|---|---:|---:|---:|---:|---:|---:|
| RA60–70, Dec−35–−25 | 9,431 | 6,821 | 6,594 | −0.006 mag | 0.206 mag | [−0.014,+0.002] mag |
| RA235–245, Dec0–10 | 17 | 10 | 9 | +0.007 mag | 0.120 mag | unavailable |
| Supplemental RA180–190, Dec25–35 | 6,815 | **0** | 0 | unavailable | unavailable | unavailable |
| Supplemental RA230–240, Dec25–35 | 7,128 | 4,746 | 4,554 | +0.042 mag | 0.215 mag | [+0.034,+0.051] mag |

The RA60–70 field gives a useful native transfer check. Its 5,955 selected
C-tier objects have median Δμ=−0.006 mag; 486 A-tier objects have
−0.006 mag. In the corrected-parallax 1–2 mas and ≥2 mas groups, the
medians are −0.014 and +0.007 mag. The 68%/95% distance intervals were not
recalibrated here.

RA235–245 has nearly no accepted sky because the original E(B−V)<0.03
criterion removes that direction. Its nine scored sources remain in the
record and are too few for a field conclusion.

The supplemental high-latitude field has 6,815 strict astrometric/optical
anchors, of which 5,401 have Gaia color 1.3≤BP−RP<3.5 and 6,744 have
Legacy M_g>5. **All 6,815 strict anchors are excluded by the published
external-reference reservation**, before the color/CMD requirements are
applied. The zero cold-branch count is therefore a role-isolation result,
not a stellar-population result. The full frozen-rule cutflow is in
`manifests/cold_v4_anchor_cutflow_development.json`.

The separate `configs/cold_v4_native_transfer_supplement2.json` registers
RA230–240/Dec25–35 as another development field before inspecting its rows.
The two originally registered independent final fields remain fixed.

RA230–240 supplies the second informative development direction. Among its
4,554 selected cold anchors, the corrected-parallax 1–2 mas group has median
Δμ=−0.003 mag (1,928 sources), while the ≥2 mas group has +0.084 mag
(2,606 sources). This reproduces the parallax-bin pattern previously seen
in RA170–180. RA60–70 instead has −0.014/+0.007 mag in those two bins.
The two informative development fields lie at Galactic latitudes about
−45° and +53°. The old RA35–45/RA170–180 crossover also pairs negative
and positive Galactic latitudes. A northern/southern difference is now a
specific testable hypothesis, not an assigned correction.

Before reading any additional final-field source rows, the project registered
`configs/cold_v4_native_transfer_north_final.json`: RA130–140/Dec25–35
adds a disjoint northern final field. The prediction is a more positive
native median there than in the two original southern final fields,
especially at corrected parallax ≥2 mas. The model and route remain fixed.
