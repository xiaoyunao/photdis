# Evidence for contamination in a high-q candidate selection

The frozen v3.1 parent contains 3540 candidate-admitted native DR11 PSF
objects, of which 3003 satisfy the fixed high-q rule. Its reported
0.266502% leakage is the model's conditional expectation. The opened
reference cohorts lie outside that finite application sample and have
their own source-selection functions. Their high-q counts and labels
describe the coverage available for a comparison.

| Opened role | High-q | Precise AstroNN | Far distance centers | Range with missing labels |
|---|---:|---:|---:|---:|
| New sky | 120 | 9 | 0 | 0.00%–92.50% |
| New source on seen sky | 323 | 113 | 2 | 0.62%–65.63% |

For a cohort of size $N$, with $K$ qualified labels and $F$ teacher
distance centers at or beyond 10 kpc, the missing-label range is
$[F/N,(F+N-K)/N]$. It holds the observed teacher labels fixed and lets
each remaining object fall on either side of 10 kpc. The range is an
identification bound for the opened cohort under these assumptions. It
excludes reference-distance uncertainty and sampling error. All rows in
these two high-q cohorts inherit the v3.1 main numerical failure.

For strict new sky, using every available AstroNN distance would raise
coverage to 84/120 and include 62 far teacher centers. The resulting
range would be 51.67%–81.67%, which shows sensitivity to the reference
precision cut. No qualified SpecDis or SPDist label supplies coverage
in either high-q subset. Signed Gaia scores use the original observation
likelihood, including weak and negative parallaxes.

[Opened bounds and denominators](../manifests/v3_2/contamination_support/opened_highq_bounds.csv) · [Source receipt](../manifests/v3_2/contamination_support/complete.json) · [Original v3.1 domain assessment](v3_1_domain_result.md)

## Parent-matched measurement

After a candidate and threshold are fixed, define a concrete native
PSF frame and write the high-q identities and their observable strata.
A probability sample from that frame needs positive inclusion
probabilities in all claimed strata. Use only observables to stratify
before reference outcomes, maintaining separate representation by
magnitude, color, WISE availability and sky. Retain the frame and
sample denominators, sampling weights, and all attempts to obtain
independent distance measurements, including failed observations.
Qualification and response rates may differ between near and far
objects even within the same observable stratum.

For a fully observed probability sample with known weights, estimate
the finite-frame number of far objects with inverse inclusion weights
and divide by the frozen high-q frame size. Carry the sampling design
into uncertainty. For missing outcomes, retain weighted lower and
upper contributions and propagate the distance errors for measured
objects. A signed-parallax analysis instead needs an explicit
selection and forward observation model for the sampled frame.
The targeted DR19 channel can report teacher agreement and observable
support for a frozen candidate. Inclusion probabilities for the
complete high-q frame are unavailable in that channel, so the
parent-matched contamination rate remains unresolved.
