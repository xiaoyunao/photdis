# Photdis v3.2 research results

The fixed giant-odds adjustment improves signed Gaia prediction relative
to old target on the frozen DR20 field. The nominal response scores
higher, and the external distance, class and count checks leave no
calibrated release domain. The v3.2 candidate is retained as a research
result. The narrow-structure recovery and NGC288 joint-quadrature
sections remain in progress.

[GPT Pro review of 7699299](pro_review_7699299.md) ·
[Experiment protocol](v3_2_protocol.md) ·
[Requirements and evidence](v3_2_requirements_result.md) ·
[English figure atlas](v3_2_atlas.md)

## Why the historical candidate lost signed Gaia score

The previously opened new-sky diagnostic reproduces a −0.8464 nat
change from old target to historical candidate zero on 166 identical
objects. Removing tilt in the matched broad fit accounts for
−0.6205 nat. Switching the population and its selection denominator
at zero tilt accounts for −0.2194 nat. The two replay-to-historical
steps sum to about −0.0066 nat. The class comparison has 157
reference-labeled objects, 142 of which are giants, and its
uncertainty is conditional on that reference selection.

Restoring tilt gains +0.7788 nat under candidate selection. The
population-by-tilt interaction is +0.1583 nat with a 95% sky-block
interval of [0.1066, 0.1963]. In a single fixed broad subsample of
3,540 PSFs, the equal-size candidate contrast remains negative at
both tilt settings, −0.6621 nat at zero and −0.4409 nat at free tilt.
This equal-size result changes population admission and selection
normalization together. The one broad draw gives a specific
downsampling sensitivity rather than a distribution across draws.

[Opened diagnostic and QRF replay](v3_2_opened_result.md) ·
[Matched arms and size control](v3_2_factorial_result.md)

## Giant probability adjustment

All 142 opened new-sky reference giants retain a giant-conditional
peak beyond 10 kpc under the common prior and each of the four
population fits. Mean giant probability is 0.2918 under the common
prior and 0.1258 under candidate/free. Residual distance errors
persist even given the giant label. On 96 giants with AstroNN
centers, the mean conditional modulus residual is −2.616 mag
for candidate/free.

The registered adjustment moves fitted giant log odds halfway
toward the odds under the common distance prior. It preserves
the normalized class-conditional distance curves and changes no
likelihood factors. In development, 526 PSF forecasts pass the
grid checks. On 166 matched new-sky signed Gaia rows the
adjustment changes the score by +0.3765 nat relative to
candidate/free, +0.3101 versus old target and −1.4529 versus
nominal reference.

Among the 142 reference giants in the same admitted development
cohort, the argmax giant count rises from 13 to 28. Mean giant
probability rises by 0.0788 [0.0544, 0.1295], and mean near
probability changes by −0.0538 [−0.0869, −0.0422].
These are paired sky-block intervals. Reference labels are used
for evaluation of the fixed rule.

[Common-prior evidence and the frozen class probes](v3_2_factorial_result.md#giant-branch-availability-and-probability) ·
[Fixed adjustment registration](v3_2_branch_protocol.md) ·
[Opened adjusted scores](../manifests/v3_2/branch_development/paired.csv) ·
[Compact mechanism audit](../scripts/v3_2_branch_audit.py)

## External validity and release limits

All four forecasts were frozen before the corresponding DR20
and HST reference access. DR20 admits 1,117 native PSFs. Its
1,099 signed Gaia measurements inside the matched photometric
gate give candidate changes of +1.0965 nat versus old target,
+0.1703 versus QRF and −0.5120 versus nominal. The candidate
retains 92 numerical flags and old target one. The 5–10 kpc
giant subset improves over QRF in signed Gaia score and loses
to nominal. Candidate argmax correctly classifies 2 of the 115
admitted, qualified giants in that interval, including 0 of
85 metal-poor giants. The 2–5 kpc giant denominator is zero.
All DR20 objects have r0 below 18 and weak W1 is absent.
At 20–50 kpc, 131 of 179 giant teacher centers fall inside
candidate 95% intervals, and mean modulus residual is −4.207 mag.
MINESweeper shares Gaia and photometric inputs with the comparison.

The two HST samples contain 77 members of 47 Tuc and 26 of NGC 362.
Candidate median distance residual is +19.57% and +23.42%,
respectively. The candidate's 95% intervals agree with the common
reference scale for 72.73% and 88.46% of the respective native
cohorts. Eleven candidate numerical flags remain. Each HST scale
draw is shared within its system and the systems probe two
specific distance scales.

The DR20 matched gate predicts 794.10 near objects on the 982
precise distance references, with conditional count interval
[775, 813] and 565 near teacher centers. For 676 class labels
it predicts 122.63 giants, interval [106, 139], with 610 reference
giants. Among the 426 rows with qnear greater than 0.95, the
precise teacher subset has 66 far centers among 400 labels.
The application parent lacks the matched independent reference
sample required to estimate its contamination rate.

[External score and support tables](v3_2_external_result.md) ·
[Contamination support](v3_2_contamination_support.md) ·
[Compact freeze audit](../scripts/v3_2_external_audit.py)

## Numerical and scientific application evidence

The historical conditional count audit traces the largest change
in each of the 33 failed broad-envelope cases to reweighting the
same 8,192 draws. Worst-bin shifts span 0.5784–0.6655 objects.
Weight ESS exceeds 8,181 despite these failures. In the new
four-arm opened forecasts both alternative entry levels pass
the checked distance-bin count and external-score tolerances.
The population-fit radial-basis count diagnostics retain larger
low-integration changes, up to 0.701 objects for candidate/free.
These checks use different count estimands.

Narrow-structure recovery uses 288 frozen mock catalogs, 256
fixed-truth and 32 prior-predictive. Four fixed model-SED strata
at a 10% generating member fraction have all eight repetitions
archived. At true widths of 0.08 mag (near background), 0.08 mag
(deep background), 0.25 mag (near background) and 0.25 mag
(deep background), local width intervals cover the truth in
4/5, 1/2, 4/4 and 1/1 gate-accepted fits.
The corresponding bounds over all registered repeats are
4/8–7/8, 1/8–7/8, 4/8–8/8 and 1/8–8/8, allowing excluded fits either
coverage outcome. At 30% generating member fraction, all eight local
fits pass in each model-SED stratum. Width coverage in the same order
is 7/8, 8/8, 8/8 and 8/8. Mean 95% interval lengths range from 0.223
to 0.400 mag. The exact 95% binomial lower limits are 0.473 for 7/8
and 0.631 for 8/8. For true width 0.08 mag, every accepted
30% fit has a width median closer to the 0.25 mag prior anchor.
The mean reduction in distance to that anchor is 0.069 mag in
the near background and 0.052 mag in the deep background.
In the same four 30% strata, the local-minus-broad change in
summed absolute joint-count error divided by catalog size is
−5.91, −8.40, −0.88 and −5.08 percentage points. The replicate
bootstrap interval crosses zero for the 0.25 mag near-background
stratum, [−4.98, +3.26] percentage points. At 60% member fraction,
the near-background 0.08-mag model-SED stratum has 8/8 passing
local fits and 8/8 width intervals covering truth. Its mean width
interval length is 0.169 mag, and its mean width median is 0.121 mag.
The paired count-error contrast is −16.41 percentage points
[−21.00, −11.77]. At the same fraction and true width in the
deep background, all eight local fits pass and cover width truth.
The median width averages 0.110 mag, and mean interval length
is 0.127 mag. The paired count-error contrast is −13.28 percentage
points [−19.28, −7.52]. At the wider 0.25 mag, both 60% model-SED
background strata also have 8/8 accepted local fits and 8/8
width intervals covering truth. The mean width medians are 0.232 mag
for near and 0.243 mag for deep. The paired count-error changes are
−15.23 [−18.75, −11.38] and −14.26 [−17.58, −10.74] percentage
points, respectively. At 90% member fraction and 0.08 mag true
width in the near background, all eight model-SED fits pass in both
arms. Local width coverage is 8/8 (exact 95% interval 0.631–1.000),
the mean width median is 0.095 mag, and the mean interval length
is 0.105 mag. All eight medians move toward the 0.25 mag prior
anchor by 0.015 mag on average. The paired count-error change is
−22.85 [−25.88, −19.39] percentage points.
With the same width and member fraction in the deep background,
both arms pass in all eight replicates. Local width coverage is
8/8, the mean width median is 0.102 mag and the mean interval
length is 0.110 mag. Seven medians move toward the prior anchor.
The paired count-error change is −25.49 [−30.41, −21.00]
percentage points.
At 90% member fraction and 0.25 mag true width, local width
coverage is 7/8 in the near background and 8/8 in the deep
background. Mean width medians are 0.262 and 0.245 mag, and
mean interval lengths are 0.190 and 0.187 mag. The paired count
error changes are −27.01 [−31.86, −21.09] percentage points
for seven accepted near-background pairs and −30.37
[−33.69, −26.66] for eight deep-background pairs. The excluded
near-background broad fit has one sampler divergence. All 16
fixed model-SED strata are now complete.
The empirical-SED and prior-predictive
comparisons await the remaining outputs. The NGC288 odds-only
audit passes at 512 to 1024 with 64 distance nodes, after the
256 to 512 step fails. The complete distance-by-odds test
awaits the 32-state joint result. In the first ten archived states,
512 independent-entry-seed checks, 424 previous-entry-level checks
and four Gaussian native checks failed. All 5,120 full-likelihood
native and all 5,120 archived-entry closure checks passed in these
states. NGC288 is a previously opened
system and demonstrates sensitivity of the inference to the
likelihood representation.

[Conditional count audit](v3_2_opened_probe_and_count_result.md#conditional-count-sensitivity) ·
[Narrow-structure protocol](v3_2_radial_protocol.md) ·
[NGC288 joint protocol](v3_2_structure_protocol.md)
