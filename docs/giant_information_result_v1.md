# Existing-field giant selection and abundance information v1

Protocol: [registered design](giant_information_protocol_v1.md). This is leave-one-whole-field-out evaluation on existing labeled fields, not an untouched final sample or a field-giant distance release.

Stored clean records: 9,390,518; unique LS photometric measurements: 9,390,518. Spectral records: 232,454; connected identity groups: 232,454; conflicting groups: 0; cross-field groups: 0.

The LS measurement count is not an independent-star count. Uncertain spectral classes and raw unassessed Redrock types remain in the audit. Base/clean counts are conditional on the stored diagnostic parent; optical availability does not require WISE.

| Field | Camera | Stored parent | Base | Clean records | W1>=5 | W1W2>=5 | W1W2>=20 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0: dev_cal_ra205_215_dec_m010_p000 | DECam | 319420 | 319095 | 275998 | 96092 | 51840 | 18842 |
| 1: native_dev_ra060_070_dec_m035_m025 | DECam | 767269 | 767112 | 695479 | 180749 | 116824 | 51643 |
| 2: native_final_ra330_340_dec_m035_m025 | DECam | 1152953 | 1128974 | 947518 | 340610 | 196708 | 73879 |
| 3: native_final_ra020_030_dec_m025_m015 | DECam | 1400211 | 1399913 | 1285115 | 273355 | 155579 | 63752 |
| 4: native_north_final_ra130_140_dec_p025_p035 | DECam | 501379 | 500566 | 423308 | 174985 | 103639 | 42543 |
| 5: cold_v4_final_ra150_160_dec_p020_p030 | DECam | 694673 | 694128 | 605103 | 216051 | 122346 | 49877 |
| 6: cold_v4_final_ra220_230_dec_p020_p030 | DECam | 336859 | 336208 | 290836 | 117535 | 67532 | 26830 |
| 7: targeted_ra180_190_dec_p000_p010 | DECam | 1185251 | 1184281 | 1037316 | 332656 | 179623 | 70157 |
| 8: targeted_ra180_190_dec_m010_p000 | DECam | 608979 | 608239 | 527873 | 179270 | 97030 | 37228 |
| 9: targeted_ra190_200_dec_p020_p030 | DECam | 770466 | 769699 | 676024 | 237996 | 128889 | 50283 |
| 10: targeted_ra020_030_dec_p000_p010 | DECam | 498581 | 498372 | 455715 | 109398 | 64249 | 26738 |
| 11: giant_ranked_ra200_210_dec_p000_p010 | DECam | 1099335 | 1098516 | 961065 | 328816 | 177777 | 67571 |
| 12: giant_ranked_ra190_200_dec_p000_p010 | DECam | 817941 | 817298 | 717984 | 233857 | 125895 | 48470 |
| 13: giant_ranked_ra210_220_dec_p040_p050 | BASS_MzLS | 578030 | 567212 | 491184 | 224399 | 133772 | 50854 |

Stored-parent/base columns are field totals (not camera subtotals for a mixed field). Clean and WISE columns are camera-specific.

## Same-star WISE>=5 class comparison

Threshold=0.5 is fixed for class-balanced scores. Values below are TPR / FPR. Wilson intervals, temperature/r/error strata, high-S/N panel, optical-only panel and color extrapolation flags are in the aggregate manifest.

| Field | Camera | Test N | A | B | C |
|---|---|---:|---|---|---|
| 0 | DECam | 642 | 0.800 / 0.265 | 0.900 / 0.098 | 0.950 / 0.085 |
| 1 | DECam | 47 | — / 0.255 | — / 0.064 | — / 0.064 |
| 2 | DECam | 0 | no_test_labels | no_test_labels | no_test_labels |
| 3 | DECam | 188 | 1.000 / 0.199 | 1.000 / 0.070 | 1.000 / 0.065 |
| 4 | DECam | 780 | 0.500 / 0.163 | 1.000 / 0.078 | 1.000 / 0.073 |
| 5 | DECam | 228 | 1.000 / 0.120 | 0.667 / 0.093 | 0.667 / 0.089 |
| 6 | DECam | 305 | 1.000 / 0.248 | 1.000 / 0.116 | 1.000 / 0.119 |
| 7 | DECam | 3093 | 0.724 / 0.206 | 0.828 / 0.086 | 0.828 / 0.080 |
| 8 | DECam | 2559 | 0.810 / 0.245 | 0.857 / 0.098 | 0.857 / 0.087 |
| 9 | DECam | 2236 | 0.714 / 0.320 | 0.714 / 0.113 | 0.786 / 0.104 |
| 10 | DECam | 1234 | 0.750 / 0.194 | 1.000 / 0.065 | 1.000 / 0.058 |
| 11 | DECam | 3675 | 0.782 / 0.295 | 0.901 / 0.125 | 0.911 / 0.116 |
| 12 | DECam | 2268 | 0.655 / 0.247 | 0.836 / 0.109 | 0.873 / 0.101 |
| 13 | BASS_MzLS | 2994 | insufficient_training_classes_or_fields | insufficient_training_classes_or_fields | insufficient_training_classes_or_fields |

## Hidden abundance comparison

Mean log-density gains relative to the training-only Gaussian abundance marginal are evaluated on identical stars. Measurement-error convolution is included; widths and coverage refer to noisy spectral labels. The fitted residual width is not intrinsic population scatter. These fixed simple models omit coefficient uncertainty; finite-sample coverage is an outcome to inspect, not guaranteed calibration.

| Field | Camera | Panel | FeH test N | A gain | B gain | C gain |
|---|---|---|---:|---:|---:|---:|
| 0 | DECam | optical | 18 | 0.352 | — | — |
| 0 | DECam | common5 | 17 | 0.351 | 0.520 | 0.609 |
| 0 | DECam | common20 | 14 | 0.167 | 0.294 | 0.386 |
| 3 | DECam | optical | 2 | 0.014 | — | — |
| 3 | DECam | common5 | 2 | -0.012 | 0.287 | 0.121 |
| 3 | DECam | common20 | 2 | -0.040 | 0.223 | -0.094 |
| 4 | DECam | optical | 2 | 0.005 | — | — |
| 4 | DECam | common5 | 2 | 0.082 | -0.257 | 0.048 |
| 4 | DECam | common20 | 2 | 0.155 | -0.181 | 0.227 |
| 5 | DECam | optical | 2 | 0.281 | — | — |
| 5 | DECam | common5 | 2 | 0.291 | 0.380 | 0.478 |
| 5 | DECam | common20 | 2 | 0.326 | 0.433 | 0.446 |
| 6 | DECam | optical | 2 | 0.171 | — | — |
| 6 | DECam | common5 | 1 | 0.048 | 0.486 | 0.634 |
| 6 | DECam | common20 | 1 | 0.230 | 0.811 | 0.957 |
| 7 | DECam | optical | 50 | 0.282 | — | — |
| 7 | DECam | common5 | 45 | 0.238 | 0.008 | -0.040 |
| 7 | DECam | common20 | 39 | 0.201 | -0.275 | -0.235 |
| 8 | DECam | optical | 12 | 0.366 | — | — |
| 8 | DECam | common5 | 12 | 0.385 | 0.333 | 0.442 |
| 8 | DECam | common20 | 10 | 0.309 | 0.483 | 0.604 |
| 9 | DECam | optical | 10 | 0.750 | — | — |
| 9 | DECam | common5 | 10 | 0.767 | 0.717 | 0.749 |
| 9 | DECam | common20 | 7 | 0.546 | 0.522 | 0.967 |
| 10 | DECam | optical | 4 | 0.487 | — | — |
| 10 | DECam | common5 | 4 | 0.477 | 0.225 | 0.057 |
| 10 | DECam | common20 | 3 | -0.018 | -0.107 | 0.153 |
| 11 | DECam | optical | 97 | 0.258 | — | — |
| 11 | DECam | common5 | 89 | 0.266 | 0.343 | 0.433 |
| 11 | DECam | common20 | 76 | 0.166 | 0.249 | 0.341 |
| 12 | DECam | optical | 52 | 0.045 | — | — |
| 12 | DECam | common5 | 46 | -0.002 | 0.203 | 0.313 |
| 12 | DECam | common20 | 41 | -0.202 | 0.028 | 0.174 |

## Finite transfer decision

A stable abundance gain requires >=3 fields with >=10 test giants each, a positive whole-field bootstrap lower limit, and positive signs in >=75% of those fields. Overlapping training folds make this a descriptive transfer uncertainty, not a new survey-level significance measurement.

- DECam, optical, A: 6 informative fields; gain 0.342, field-bootstrap interval [0.189, 0.523]; passes registered limited-family gain criterion.
- DECam, common5, A: 6 informative fields; gain 0.334, field-bootstrap interval [0.166, 0.526]; passes registered limited-family gain criterion.
- DECam, common5, B: 6 informative fields; gain 0.354, field-bootstrap interval [0.180, 0.527]; passes registered limited-family gain criterion.
- DECam, common5, C: 6 informative fields; gain 0.418, field-bootstrap interval [0.206, 0.598]; passes registered limited-family gain criterion.
- DECam, common20, A: 5 informative fields; gain 0.128, field-bootstrap interval [-0.048, 0.252]; no stable gain established (weak or insufficient evidence).
- DECam, common20, B: 5 informative fields; gain 0.156, field-bootstrap interval [-0.094, 0.361]; no stable gain established (weak or insufficient evidence).
- DECam, common20, C: 5 informative fields; gain 0.254, field-bootstrap interval [-0.005, 0.474]; no stable gain established (weak or insufficient evidence).
- BASS_MzLS, optical, A: 0 informative fields; gain —, field-bootstrap interval [—, —]; no stable gain established (weak or insufficient evidence).
- BASS_MzLS, common5, A: 0 informative fields; gain —, field-bootstrap interval [—, —]; no stable gain established (weak or insufficient evidence).
- BASS_MzLS, common5, B: 0 informative fields; gain —, field-bootstrap interval [—, —]; no stable gain established (weak or insufficient evidence).
- BASS_MzLS, common5, C: 0 informative fields; gain —, field-bootstrap interval [—, —]; no stable gain established (weak or insufficient evidence).
- BASS_MzLS, common20, A: 0 informative fields; gain —, field-bootstrap interval [—, —]; no stable gain established (weak or insufficient evidence).
- BASS_MzLS, common20, B: 0 informative fields; gain —, field-bootstrap interval [—, —]; no stable gain established (weak or insufficient evidence).
- BASS_MzLS, common20, C: 0 informative fields; gain —, field-bootstrap interval [—, —]; no stable gain established (weak or insufficient evidence).

Unscored class cells: DECam/2/optical: no_test_labels; DECam/2/common5: no_test_labels; DECam/2/common20: no_test_labels; BASS_MzLS/13/optical: insufficient_training_classes_or_fields; BASS_MzLS/13/common5: insufficient_training_classes_or_fields; BASS_MzLS/13/common20: insufficient_training_classes_or_fields.

## Population and distance boundary

The evidence applies to DESI-qualified 4200–5000 K low-gravity versus high-gravity stars within these native-camera, clean, high-latitude low-dust samples. The three common-panel models have identical train/test stars. Changing the WISE S/N panel changes its population; its performance difference is not a causal noise-only gain. Pure optical availability remains separate.

The purity-prior figure assumes only two transported classes. Unknown temperatures, intermediate gravities, nonstellar contamination, DESI targeting and different field populations prevent reading it as full-catalog purity. Selected labeled-giant temperature/logg/FeH distributions are provided per fold in the manifest; they characterize that labeled subset only.

No lower-RGB phase identification has been established. A useful WISE score can define a finite-depth evolved-star candidate layer, but RGB/RC/HB separation and an external luminosity calibration for its actual phase/FeH range are still required before distances. Where FeH gains fail the criterion, retain explicit external population priors. Even successful q(FeH|colors) must enter a consistent joint distance model: reusing it as an independent prior with the same color likelihood would double-count information.

The next calibration must target the observed K-giant candidate population and its independently established evolutionary phases, rather than attach the present M3/M53 lower-RGB curve to every low-gravity object. No new field, color-cut search, distance-VAC regressor or RGB-curve tuning was used.

## Artifacts

- `figures/giant_information_v1/01_wise_availability.png`
- `figures/giant_information_v1/02_class_transfer.png`
- `figures/giant_information_v1/03_purity_prior.png`
- `figures/giant_information_v1/04_feh_information.png`
