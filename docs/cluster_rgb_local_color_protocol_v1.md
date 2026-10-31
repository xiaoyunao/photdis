# Local conditional-color information (registered before scoring)

Source: [complete Pro review](pro_review_4c203ca.md); base `4c203ca`.
The four existing systems, frozen 48 calibration realizations, four knots,
robust mixture, quality thresholds and phase evidence are retained. Historical
full-sample results and failed numerical flags stay unchanged.

## Fixed question and selection

Use delta_mu = mu-mu_ref uniformly on [-.5,.5] mag, all four clusters identically.
Within the original same-row clean grz lower-RGB identities, retain nominal
1.1 <= g0-mu_ref <= 2.4. Do not tune the window or remove residual outliers.
Count and hash both original and interior identities. Report measurement g
errors and common-g sigma against the nominal .1-mag worst-case margin.
An empty/very small sample is a result, not permission to scan windows.

This uses reference information explicitly and is an exposed local experiment.
It measures updates to a local prior, not distance-blind recovery. The original
[.5,3] latent Mg interval remains a calibration support assumption, not a
measured luminosity function or the physical endpoints of all RGB stars.

## Conditional likelihood and numerical route

For each shared nuisance realization eta, form the product over members first.
Define J(mu)=integral p(eta) product L_gc,i(mu,eta) d eta and
B(mu)=integral p(eta) product L_g,i(mu,eta) d eta. The target is
p(delta_mu|C,G) proportional to local_uniform(mu) J(mu)/B(mu).
It is NOT the prior average of the product of per-star likelihood ratios.

Reuse the tested piecewise-linear exact latent-Mg integration, shared g/color
covariance and exact Gaussian nuisance collapse. For the numerator, sample
local_uniform(mu) J(mu) with shared parameters; its arbitrary evidence constant
cancels when the final conditional posterior is normalized. Divide its marginal
mu density by independently calculated B(mu), AFTER nuisance marginalization.
No RR likelihood is added. The RR reference sets the disclosed local coordinate
and interior selection only.

Brightness alone depends on the common g shift, whose Gaussian variance is
sigma_zero^2 + R_g^2 sigma_E^2. Integrate this single shared shift with 64/128
Gauss-Hermite nodes, retaining the product of each member's finite-support normal
CDF interval. Compare log B(mu) shapes, not underflowing likelihood products.
The normalized Hermite weights divide by sqrt(2*pi).
[Quadrature definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html).

The no-color mathematical control sets J=B exactly and recovers the local
uniform prior. Separately measure B's log range across the local window: flag
> .05 nat as material brightness-boundary dependence. Report the union upper
bound on any member crossing support, using measurement and common-g variance;
this bound does not assume the shared errors are independent across members.

## Finite comparisons

On identical interior stars use robust gr and robust grz, plus no colors.
Reference abundance: the existing cluster Gaussian mean/error. Unknown abundance:
the existing point atoms [-1.46,-2.00] with equal [.5,.5], rich [.8,.2], poor
[.2,.8] weights, unchanged. A SINGLE atom is common to the whole cluster.
Sample the equal mixture jointly over the 48 calibration indices and two atoms;
reweight by the bounded new/equal mixture-weight ratio for the rich/poor cases.
Do not separately normalize each atom's posterior before combining; that would
discard the relative marginal likelihood. Report component occupancy and weight ESS.

Use sigma_E=.005 and .01 in physical reddening units, zero sigma=.01 per band,
with the same already-audited local CCM derivative. No new extinction investigation.
Reference abundance Gaussian variation is absorbed into the collapsed covariance;
unknown point atoms have zero within-atom abundance variance as originally defined.

## Numerical validation and bounded endpoint

Four chains, 3000 discarded warmup, 24000 production steps each; same frozen
likelihood and priors. A reversible model proposal follows the distance/color
degeneracy using fixed per-model offsets and a unit-Jacobian color shear.
All proposal changes are numerical, not changes to the curve or posterior.
Validate inverse/Jacobian and Gaussian-collapse identities on synthetic data.

Check rank/folded split Rhat <1.01 and bulk/tail ESS >400 for distance, color
shifts and log target; also check atom occupancy when nonconstant, independent
chain conditional summaries, and [.005,.0025] grid quantiles (difference <.005 mag).
Preserve a constant atom indicator as a mixing limitation if no chain visits the
other atom, rather than claiming an Rhat for a constant series. One bounded
48000-step refinement of failed new fits is allowed; retain any remaining flags.
Do not extend old broad-window MCMC. Test the marginalized-ratio order with an
analytic shared-nuisance counterexample and exact no-color cancellation.

## Outputs and end decision

Report N, local prior, delta_mu median/16–84/2.5–97.5 quantiles, W68/W95,
KL relative to the uniform local prior (model information, not accuracy), B's
flatness, and edge mass in each outer .05-mag strip. Flag strip mass >.15 or
an edge-strip density maximum; a window-truncated width is not color precision.
Report paired gr/grz changes and sensitivity to FeH weights and dust width.

Conclude one bounded experiment: stable color constraints motivate a specified
target-population model; prior-sensitive constraints identify missing abundance/
reddening information; endpoint-dominated constraints require physically supported
calibration-range expansion. Do not extrapolate the relation or launch those
follow-up projects automatically. Allocate NERSC CPUs for scientific calculations;
sync code, docs, small aggregate results and inspected figures only.
