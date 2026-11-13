# Preserve original structure numerical failures and refine only quadrature

The response12 native pilot retains7 failed objects out of512. Initial coarse/fine
log-response differences reach0.411724nat, and class-probability differences reach
0.0456824. These are failures of the preparatory numerical comparison, not a
scientific baseline benefit or permission to delete objects.

For an original failed object, retain its fine grid as the first comparison.
Refine the global[-20,50] spacing to.005/.0025/.00125mag and the same common-scale
region to.001/.0005/.00025mag, stopping only after the existing.02nat/.003 tests
pass for BOTH full likelihood and moment-Gaussian baseline. Store each attempt
and the actual previous/new response arrays as separate sidecars. No change to
photometry, native SED, binary states, calibration, common distance or spatial
priors, class frequency, model choice or scientific admission is allowed.

For initially passing objects, retain the original coarse/fine pair. Complete
sidecar inventories cover all512objects for every response; matrix assembly
uses the authenticated selected pair and retains the initial failure flag.
Unresolved terminal failures block accepted structure inference. Final joint
evidence/count/distance quadrature tests remain independently mandatory.
