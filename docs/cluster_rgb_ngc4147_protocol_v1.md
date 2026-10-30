# Registered independent lower-RGB transfer: NGC 4147

Registered before extracting this system's Legacy RGB or scoring it. Follow
the complete Pro review in `pro_review_c1f5c59.md`; configuration and hashes
are in `configs/cluster_rgb_ngc4147_v1.json`. The primary is frozen **robust
monotone grz**. Quadratic/ordinary monotone, gr and analytic uniform-prior
controls retain their existing fits. No fitting, parameter search or new
quality thresholds belongs to this experiment.

## Independence and eligibility

`git log --all -G 'NGC.?4147'` finds only the earlier system-ranking commit
9133a32. The ranking records one giant spectrum, coordinates and an external
distance; no cluster RGB relation was fitted. No named NGC 4147 extraction
or model product exists in the NERSC data directories. Extraction will
verify source disjointness against the actual frozen 672 development IDs.
This qualifies as a new RGB transfer system; the older rank is disclosed.

Use the complete Vasiliev–Baumgardt member list, existing P>0.9 core and
unchanged Gaia phase region copied from M53. Match Gaia IDs to southern DR11
DECam. Report RELEASE, stage and band-dependent quality counts, gr-only
added coverage and half-mag bins. Only then lock the scoring identities.
The phase screen gives RGB candidates, not individual evolutionary proof.
Reference-Mg selection is an external validation domain, not an application
selection available to unknown-distance stars. Keep the 2–3 mag results.

## External scales fixed before scores

[2018 RRc study](https://arxiv.org/abs/1803.05313) supplies 19.05±0.46 kpc
and E(B−V)=0.01. RRab amplitude modulation makes their Fourier parameters
unsuitable here. Transform its ZW −1.72±0.15 using the same polynomial as
development: UVES −1.689±0.203 dex (linear propagated uncertainty).
Use three quadrature nodes (center±sqrt(2) sigma, center), weights .25/.5/.25,
as in the frozen implementation. Also show the exact-center sensitivity.

[Spectroscopy](https://arxiv.org/abs/1605.03408) gives −1.84±0.02 dex;
0.02 is an error on the mean. Register an assumed 0.10-dex scale floor,
combined in quadrature, for this alternative; it is not an exact scale
conversion. The [homogeneous compilation](https://arxiv.org/html/2306.01175v1)
lists RRc UVES −1.68, ZW scatter 0.26, RRc distance 18.7±0.5 kpc and
E(B−V)=0.02 in its distance table (0.01 in its abundance table).
Register that abundance/propagated scatter, and distance 18.7 kpc, as
separate alternatives. The 2018 I-band RRc PL 18.89 kpc is another distance
sensitivity based on the same stars, not an independent combined constraint.
These literature differences do not select the primary after scoring.

Retain native Legacy corrected photometry; show its E(B−V) distribution.
Apply coherent ±0.01 E(B−V), ±0.01 per-band zero points and ±sigma_mu to
the *same* selected IDs. Hold nominal reference-Mg bins for those contrasts.
Target and calibration scale errors are shared, never divided by sqrt(N).
RR distance and abundance reference covariance is unavailable; simulations
assume independence and the scale grid exposes this assumption's scope.

## Once-only comparisons and target-matched recovery

Score every fixed model on identical grz-clean IDs. Report gr-only added
coverage separately. Main and both alternative abundance distributions,
exact primary abundance, and the three unchanged unknown-FeH mixtures
are all retained. Report center, NMAD, >0.5-mag tail, interval widths,
68/95 coverage and log predictive density overall and by fixed Mg bin.
Prior-only is analytically uniform on [0.5,3], not a training histogram.
Coverage uses noisy reference g−mu. If a perturbed point lies outside the
bounded density support, full log score is undefined; report interior count.
Source bootstrap describes star-sampling contrasts conditional on this
cluster and cannot replace a shared-scale uncertainty.

Simulate the frozen primary only, with actual per-bin band errors and the
target abundance uncertainty. Two true-Mg designs: the registered fixed
grid and the observed reference-Mg distribution. For each of 48 synthetic
clusters draw one of the 48 calibration realizations and one target FeH;
draw residuals from the fixed .95/.05 mixture. Add a single shared reference
modulus, dust and three zero-point perturbations per synthetic cluster,
plus each source's band noise. The same g error enters colors and final
modulus. Compare true-Mg binned recovery and observed-proxy bins explicitly;
retain out-of-domain noisy references for distance/coverage summaries.
One synthetic cluster is the unit of predictive bands and Monte Carlo
uncertainty. Repeat the grid without shared target scale perturbations to
isolate ordinary shrinkage. Use the actual observed-Mg distribution for
the scale-compatible cluster median predictive envelope. This is a plug-in
luminosity/error design, not a fitted Galactic population model.

Answer: (1) external scale compatibility under registered shared errors,
(2) extra luminosity trend beyond simulated shrinkage, (3) gain over
prior-only, (4) changes when target abundance is removed. Retain failure
results and primary identity. The 5% broad component is a robust likelihood
assumption, not a measured class fraction. No field RGB product or universal
5% distance claim follows from one conditional system test.

## Delivery

Allocated NERSC extraction and scoring; source products stay there. Save
coverage/score/simulation manifests, exact identity and file hashes, two
scientific figures and a concise decision report. Validate sign/shared-band
noise, immutable model hashes, sample identity and simulation clustering.
Update PLAN/WORKLOG and synchronize tracked milestones to all three sites.
