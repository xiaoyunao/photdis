# Structure likelihood grids and numerical acceptance

All32 frozen original response states retain all512 selected identities. Native
likelihood grids cover[-20,50] at.02/.01mag, augmented across the union of the
registered common-distance quadrature nodes plus0.2mag on both sides with
.004/.002mag spacing. This local numerical refinement resolves the fixed0.02mag
intrinsic structure width; it does not alter the shared-distance prior or fit
intrinsic depth. Both grids retain all native kernels and binary states.

For every class compute the exact numerical reference evidence and reference
posterior mean/variance underN(15,5). Form the moment-Gaussian likelihood
approximation by explicitly withdrawing that prior, then apply each new prior.
Keep exact and approximate class-integrated background and common-scale
responses for both grids. A Gaussian approximation can fail scientifically even
when its numerical integral is stable.

Per-object checks require relevant class/prior log-response differences<.02nat
(relevant means within30nat of that object's maximum), and normalized class
probability changes<.003 for each radial prior. Failures stay in the inventory;
no object or response is discarded. Final full-catalog evidence, posterior
membership/count and common-distance quadrature transfer checks remain mandatory
and are stronger than these preparatory checks. The first native pilot is
response12 on all512objects; all32 states are ultimately required.
