# Final candidate admission in external evaluation

Registered before final new-reference opening. Predictions retain every frozen
nativePSF validation object. Primary external evaluation of the3540candidate
must use the identical observed-photometry admission from
`v3_1_target_native_pilot.py`: nativeDECam_DR11/TYPE=PSF, observed r<20,
maximum grz magnitude error<.1, and the unchanged frozen QRF grz feature box and
10th-neighbor-radius support gate. Observed r is not extinction-corrected r0.
No luminosity correction, predicted distance, class, reference or residual is an
argument to this gate. Highq adds finalcandidate qnear>.95 AFTER this admission.

Recompute and prove bit-identical membership on all26000application rows against
the frozen3540candidate mask before applying the same gate to field/common roles.
Preserve every validation identity and rejection reason, including overlapping
brightness/precision/support reasons; native missing/nonPSF exclusions remain in
upstream admission manifests. Do not retrain QRF or widen its support bounds.

Report candidate-domain field strata as primary; separately show all-frozen and
outside-candidate diagnostic summaries. Common-system outputs must distinguish
candidate members, all members, outside-candidate members and the fixed main
candidate highq subset. Each retains one common system scale, irrespective of N.
Domain exclusion cannot hide original failed predictions or reference outcomes.
QRF support only reproduces the existing selection envelope; it does not establish
native-SED support, correct branch weights, empirical coverage or absolute scale.
Those scientific use-domain questions require the separate final evidence map.
