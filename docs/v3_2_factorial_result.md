# Opened-field population and SED tilt experiment

The four matched population fits passed their sampler checks. The common
forecast completed on 526 native DR11 PSFs. The paired report uses 519
candidate admissions, including 167 new-sky and 352 seen-sky objects.
These previously opened samples serve as development diagnostics.
[Fit receipts](../manifests/v3_2/tilt_fits/initial/) ·
[Per-object forecasts](../manifests/v3_2/tilt_predictions/initial/objects.csv) ·
[Paired report and count tables](../manifests/v3_2/tilt_report/initial/)

## Where the historical score was lost

On 166 new-sky rows with signed Gaia data, the old target to historical
candidate-zero change is −0.8464 nat. The matched path below closes for
each object. Intervals resample whole sky blocks within each contrast.

| Step | Mean change (nat) | 95% interval |
|---|---:|---:|
| Old target to matched broad/free | −0.0053 | [−0.0077, −0.0021] |
| Remove tilt in broad | −0.6205 | [−0.6955, −0.5064] |
| Change from broad to candidate at zero tilt | −0.2194 | [−0.3751, −0.0778] |
| Matched candidate zero to historical candidate zero | −0.0012 | [−0.0030, +0.0017] |

Restoring tilt improves the candidate score by +0.7788 nat. The
population-by-tilt interaction is +0.1583 nat [0.1066, 0.1963].
At free tilt, candidate minus broad is −0.0611 nat [−0.2646, +0.0714].
The population contrast at zero tilt combines sample composition, size and
matched conditional selection. A size-matched broad control would be needed
to identify a size effect. Tilt has a directly identified effect within
each fixed sample. [Paired contrasts](../manifests/v3_2/tilt_report/initial/factorial.csv) ·
[Historical path](../manifests/v3_2/tilt_report/initial/historical_decomposition.csv)

Candidate/free remains below the nominal reference by 1.8294 nat on
the new sky and 0.2844 nat on the seen sky. Against old target the
changes are −0.0664 nat [−0.2669, +0.0661] and +0.1081 nat
[+0.0471, +0.1939], respectively. The six-parameter historical correction
and nominal response remain in the comparator table.
[Historical controls](../manifests/v3_2/tilt_report/initial/historical_anchor.csv)

## Giant branch availability and probability

All 142 new-sky reference giants have a conditional giant peak beyond
10 kpc under the common prior and each population fit. The common-prior
mean giant probability is 0.2918. It falls to 0.0269 and 0.0456 in
broad zero and free, and to 0.0804 and 0.1258 in candidate zero and free.
The corresponding giant argmax counts are 0, 0, 7 and 13.

Candidate/free improves the full signed Gaia score for these giants
by 0.8830 nat over candidate/zero. Conditioning on their known giant
labels instead gives a change of −0.1170 nat [−0.1477, −0.0211].
The 96 giants with AstroNN centers retain a mean conditional modulus
residual of −2.616 mag for candidate/free, versus −2.601 mag under the
common prior. AstroNN supplies teacher agreement in this comparison.
The frozen multiband probe minus candidate/free giant class score is
−0.0508 nat [−1.1243, +0.5184], leaving the available information gain
uncertain for this cohort.
[Giant strata](../manifests/v3_2/tilt_report/initial/giant_decomposition.csv) ·
[Confusion](../manifests/v3_2/tilt_report/initial/confusion.csv) ·
[Probe comparison](../manifests/v3_2/tilt_report/initial/probe_factorial_comparison.csv)

## Numerical scope

The opened forecast reports zero grid and sampler flags across all four
arms and 526 objects. Across the two entry alternatives the maximum
expected class-distance bin shift is 0.004055 objects on the admitted
reference cohorts. All checked external-score shifts meet their tolerances.
The larger fitted-population radial-basis shifts remain in the fit
sidecars. Joint realized count intervals condition on each opened cohort
and integrate its shared population draws.
[Forecast gates](../manifests/v3_2/tilt_report/initial/numerical.csv) ·
[Entry sensitivity](../manifests/v3_2/tilt_report/initial/entry_sensitivity.csv) ·
[Joint counts](../manifests/v3_2/tilt_report/initial/joint_counts.csv)
