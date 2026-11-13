# Structure realized counts and descriptive Gaia comparison

This numerical/report sidecar leaves the frozen32-response structure likelihood,
512photo-only identities, common scale prior and full/Gaussian branches unchanged.
It runs only after both original structure posteriors and numerical gates have
finished. Preserve flagged branches and label their reports accordingly.

For each branch sample65536whole catalogs from the final64x256 shared posterior,
seed320601. Draw ONE(response,common-distance,spatial-odds) index for each catalog,
then independent Bernoulli memberships conditional on that index and the observed
fluxes. The observed-membership probability is qR_s/(qR_s+(1-q)R_b); do not divide
each component by its entry probability. Selection already enters global weights.
Store int64(member,background) counts summing exactly to512, shared state indices,
and empirical2.5/16/50/84/97.5percentiles. These are posterior realized counts,
not confidence intervals around the conditional expected count or Poisson errors.

Compare simulated moments to the exact original conditional+shared covariance.
Require mean difference<5MonteCarlo-standard-errors+.05count, alternating-half
CDF difference<.01, half-quantile difference<=2counts, and half-variance difference
divided bymax(exactvariance,1)<.05. If any fails, retain the first attempt and
make one bounded262144-catalog attempt with the same seed; keep terminal failures.
These are sampling checks, separate from the original quadrature/response/entry
gates. Do not reinterpret a Gaussian/full difference as a model Bayes factor.

Only after the posterior is complete, associate the existing v3.0 frozenNGC288
GaiaEDR3 membership sidecar using exactidentity and originalrow. Missing membership
stays missing, never a nonmember. Describe all512, membershipavailable, published
p>=.9 members, and availablep<.9 separately. Report per-branch model member sums,
probability contrasts and threshold agreements at fixedp>.5/>.9, alongside each
denominator. No Gaia labels alter selection, model, weights, thresholds or priors.
The published probability is an external model comparison in an already-seen
system, not independent hard truth; no binomial accuracy interval or independent
cluster-scale recovery claim is warranted. No catalog expansion or physical
density estimate follows. Figures and scientific interpretation remain required.
