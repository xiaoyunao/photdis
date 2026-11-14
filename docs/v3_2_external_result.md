# Photdis v3.2 frozen external validation

DR20 job 61273704 bound all four forecasts before opening reference
outcomes and completed in 4m11s. All 1,117 admitted DR11 PSFs have
four forecast rows. The matched photometric gate admits 1,107.
The candidate retains 92 forecast numerical flags and old target one.
[Freeze and evidence](../manifests/v3_2/final_validation_dr20/) ·
[Scores](../manifests/v3_2/final_validation_dr20/scores/)

The native cohort has 1,109 eligible signed Gaia measurements and 990
precise MINESweeper centers, using the registered error rule. The precise
sample has 213 objects at 2–5 kpc and 329 at 5–10 kpc. It has zero
qualified giants at 2–5 kpc and 117 at 5–10 kpc, including 87 metal-poor
giants. At 10–50 kpc there are 397 classified giants, including 192
metal-poor giants. All sources have r0 below 18 and weak W1 has no
coverage. MINESweeper uses photometry and Gaia information, so agreement
with its centers measures teacher agreement.
[Reference coverage](../manifests/v3_2/final_validation_dr20/coverage.csv)

On 1,099 same-object signed Gaia pairs inside the candidate gate, the
candidate scores +1.0965 nat [1.0463, 1.1445] against old target and
+0.1703 nat [0.0677, 0.2773] against QRF. It scores −0.5120 nat
[−0.5577, −0.4649] against nominal reference. In the 2–5 kpc teacher
subset it falls below QRF by 0.4678 nat and nominal by 0.6401 nat.
The candidate has a Gaia PIT lower-tail fraction of 15.74% and upper
tail of 0.09% on the matched gate.
[Paired scores](../manifests/v3_2/final_validation_dr20/scores/paired.csv)

The near-distance reference subset has 982 qualified rows. The candidate
expects 794.10 near objects with a conditional count interval of
[775, 813], while 565 teacher centers are near. For 676 labeled rows,
it expects 122.63 giants with interval [106, 139], while 610 have giant
labels. These intervals condition on the fixed marginal probabilities.
Of 426 objects with qnear above 0.95, 66 of 400 precise teacher centers
are far. Model-expected far count over the 426 is 6.92. This disagreement
is measured on a teacher-selected cohort. Its counts cannot supply
the contamination rate of the application parent.
[Reference counts](../manifests/v3_2/final_validation_dr20/scores/reference_counts.csv) ·
[High-q support](../manifests/v3_2/final_validation_dr20/scores/highq_reference_support.csv)

The frozen HST comparison included 77 47 Tuc and 26 NGC362 members.
Candidate median relative residuals are +19.57% and +23.42%, versus
+12.97% and +14.58% for QRF. Candidate 95% interval agreement is 72.73%
and 88.46%. The 11 candidate numerical flags remain in the comparison.
The score improvements over old target do not establish calibrated
distance or class probabilities for the proposed product.
[HST common-scale table](../manifests/v3_2/final_validation_common/scale_agreement.csv)

[Signed Gaia and HST figure](v3_2_atlas.md#signed-gaia-scores-and-hst-distance-scales)
