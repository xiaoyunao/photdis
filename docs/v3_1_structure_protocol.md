# NGC288: explicit structure-prior withdrawal and shared-uncertainty example

This is a seen-system methodological application, not the new NGC1261 test and
not evidence of independent distance accuracy. Use the existing photometric-first
NGC2888192-object nativePSF cohort, with512 identities selected by lowest SHA256
`320201:<identity>`. Keep the original grzSNR5/PSF admission and1degree footprint;
no membership probability or residual selects objects. Preserve original
admission/exclusion records and conditional sampling probability. Count the
observed512objects, not all48230eligible sources or a physical surface density.

This demonstration uses the frozen original32 SED/source-calibration/band/dust
response draws with equal Monte Carlo prior mass, before structure photometry.
All response-specific within-class kernels and binary states remain; class
frequencies are the same frozen nominal frequencies for all responses. The
six-parameter v3.1 response12 correction is NOT silently applied to other SEDs.
This reference-ensemble application has a separately named identity from the
corrected target candidate; its success cannot count as that candidate's final
validation. Main product-matched validation remains separately mandatory.

At each recorded angular radius r, define pre-photometric-entry member odds
logit(q_i)=zeta-2log(1+(r/0.1degree)^2), with zeta~N(0,4). This fixed spatial
shape is an illustrative conditional membership model, not a measured profile
or mask/area-corrected cluster density. Background distance is an equal mixture
of the existing five broad Gaussians, with the same nominal class frequencies.
The structure has fixed intrinsic distance-modulus width0.02mag. No fitted
intrinsic-depth claim follows. All these priors are fixed before new application
scores; Gaia membership is reserved for a descriptive later comparison.

The common structure distance uses the already-seen external dynamical-scale
information from `configs/v3_0_independent_scale.json` as ONE SHARED PRIOR:
median9.03kpc with lower/upper scales0.56/0.48kpc. Define its inverse-CDF by
d=9.03+sigma_side*Phi_inverse(u), choosing the lower/upper scale below/above
u=0.5, then transform to mu. This is an explicit equal-half asymmetric scale
model, not512independent distance truths; never divide its error bysqrt(N), and
never claim recovery of this input anchor as independent validation.

Use the selected-data likelihood
[q_i*R_structure_i(mu_common)+(1-q_i)*R_background_i] /
[q_i*A_structure_i(mu_common)+(1-q_i)*A_background_i].
R is obtained from the native full likelihood after withdrawing the original
reference/target prior. A uses the same native response, local dust, shared band
and dust offsets, binary states and actual grzSNR5 entry. Template/calibration
draw, common distance and zeta are global indices shared by every object.
Update their joint weights by the actual selected likelihood; do not flatten
response weights, stack independent posterior PDFs or sum independent-star
Poisson errors. Report member/foreground count conditional and shared covariance.

The fair baseline approximates EACH CLASS-CONDITIONAL reference posterior by
one Gaussian matching its mean/variance, retaining its exact class evidence.
Withdraw N(15,5) analytically before applying the same structure/background
priors. Thus L_approx(c,mu)=R_reference(c)*N(mu;m_c,s_c)/N(mu;15,5).
Both branches retain identical sample, structure priors, response priors,
selection integrals and common-scale uncertainty. This is a likelihood-summary
approximation to the same model, not a newly normalized generative flux model.
Compare member count, common-scale posterior and foreground mixing; keep all
negative outcomes and distinguish the role of the supplied scale anchor.

Numerical preparation begins with the three fixed min/median/max effective-error
contexts in the512photo-only sample at response12. Contexts are ranked by
geometric-mean calibrated/transmission-corrected grz
flux error (stable sort), selecting rows at ranks0,256,511. The broad numerical
proposal isN(15,6); the narrow proposal isN(mu(9.03kpc),0.8). The narrow pilot
includes the union of common-scale CDF quadrature nodes at16/32/64 orders.
Direct broad/narrow Gaussian entry uses separate proper numerical proposals, with target/proposal ratios
and finite maxima recorded; proposals cancel and are not scientific priors.
Use levels131072/524288/2097152, seeds320211/320212 and the same absolute.0005 /
stabilized1% checks, then verify count sensitivity. No interpolation across
observed error/dust contexts without a separately registered validation design.
Common-distance CDF quadrature orders16/32/64 and zeta-normal orders64/128/256
are numerical-only refinements. Require probability change<.003, count-mean
change<.5, common-mu quantile change<.03mag and log-evidence change<.05nat.
Numerical failures prevent an accepted structure result, not sample/draw deletion.
