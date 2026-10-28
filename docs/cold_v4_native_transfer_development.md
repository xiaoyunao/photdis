# New-field native transfer: development checkpoint

The original four 10° fields and one supplemental development field were
registered before their source rows were read. Each was built on a NERSC
allocated CPU node using the unchanged DR11, Gaia and optical pipeline.
The native scores use the frozen cold-v4 forests, support, color box and
`grz_clean`+WISE S/N≥20 route. Independent final-field residuals have not
yet been scored. The small field manifests and development scores are in
`manifests/field_native_*.json`,
`manifests/cold_v4_native_transfer_development.json` and
`manifests/cold_v4_native_transfer_supplemental.json`.

| Development field | Strict Gaia anchors | Cold anchors | Selected | Δμ median | NMAD | Sky-block 95% median range |
|---|---:|---:|---:|---:|---:|---:|
| RA60–70, Dec−35–−25 | 9,431 | 6,821 | 6,594 | −0.006 mag | 0.206 mag | [−0.014,+0.002] mag |
| RA235–245, Dec0–10 | 17 | 10 | 9 | +0.007 mag | 0.120 mag | unavailable |
| Supplemental RA180–190, Dec25–35 | 6,815 | **0** | 0 | unavailable | unavailable | unavailable |

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
