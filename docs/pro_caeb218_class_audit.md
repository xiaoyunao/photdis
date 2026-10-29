# Class evidence and blue-anchor audit after `caeb218`

This round keeps the cold-v4 distance models, route choice and earlier final
test identities frozen. The versioned catalog edits change class evidence and
blue-branch training eligibility only.

## Physical class versus branch eligibility

The v3 evidence separates four questions:

1. Is a DESI RVS spectrum a qualified stellar atmospheric measurement?
   Require an unambiguous match, `RR_SPECTYPE=STAR`, primary and successful
   RVS result, `RVS_WARN=0`, `SN_R≥10`, `0≤VSINI<30`, finite positive
   temperature and gravity errors. A missing or out-of-domain iron abundance
   does not invalidate temperature and gravity as class evidence.
2. Is that star a conservative cool dwarf or giant? For `T_eff<7000 K`,
   require `logg−σ_logg>4.2` for a dwarf or `logg+σ_logg<3.2` for a giant.
   These physical class definitions do not impose the 4200–5000 K blue-branch
   range. A qualified giant conflicts with the cold-dwarf distance hypothesis.
3. Is the star eligible for the bounded blue main-sequence experiment? It
   needs the conservative dwarf label, `4200≤T_eff<5000 K`, and a continuous
   calibrated `[Fe/H]` with valid formal uncertainty. The later training core
   also requires clean low-dust DR11 `grz`, usable parallax and no unresolved
   independent CMD/photometry conflict.
4. Does Redrock give QSO/GALAXY? The MWS VAC provides `RR_SPECTYPE` but no
   Redrock `ZWARN` or `DELTACHI2`. A nonstellar type is retained as raw
   `nonstellar_unassessed` evidence with an explicit quality state. RVS
   stellar-fit success is not used to hide that type or claim it is a
   quality-confirmed nonstar.

The Gaia white-dwarf CMD candidate is stored independently from the RVS
class. The resulting v2 same-star inventories count zero direct CMD/RVS
class conflicts. The broad-temperature RVS rule finds 609 qualified giants
in the original seven fields and 3,085 in the four DESI-targeted fields.
Among these, 383 and 2,104 respectively have clean low-dust DR11 photometry.
The targeted set includes 27 with `T_eff<4200 K`, 230 at 4200–5000 K,
2,828 at 5000–7000 K, and 82 with `logg<1.8`.

The versioned 395,629-row candidate export preserves all 20 old columns
outside the five replaced class-evidence columns, verified row by row on
allocated compute. Its broad class labels are mostly unknown because few
photometric targets have qualified DESI spectra. Twenty-four candidates
carry raw Redrock QSO/GALAXY type without Redrock-specific quality metrics;
none has a qualified RVS giant conflict in these three test fields.

## Gaia source 3601173023628559616

The object matches the DR11 sweep `REF_ID` exactly. No other DR11 source is
within 10 arcsec. DR11 has four, three and three observations in `g,r,z`,
respectively. Compared with one PS1 source 0.14 arcsec away after proper
motion propagation and PS1-to-DECam color transformation, DR11 minus PS1 is
−0.011, +0.463 and +0.022 mag in `g,r,z`. The DR11 `r` fit has
`RCHISQ_R=2097`, while `g,z` have 27 and 22. This band-specific evidence
shows that its measured `r` is unsuitable for a precise color-luminosity
anchor even though the original `grz_clean` predicate passes.

The v3 joint evidence retains the entire row and adds
`photometry_audit_status=r_band_discordant`; it switches
`trusted_blue_core` to false for this source. The decision uses identity,
another survey's photometry and the DR11 fit diagnostic, not a fitted
distance residual. The regenerated targeted joint CMD extends to
`M_g=10.8` and shows the previously clipped `M_g≈10.19` source.

DR11 dereddened magnitudes are computed from measured flux and
`MW_TRANSMISSION` in each band, so these color and luminosity diagnostics
already include the project's extinction correction. The blue experiment
further requires `E(B−V)<0.03`.

## Artifacts

- [Joint seven-field counts](../manifests/joint_calibration_v1_evidence_v2.json)
- [Joint targeted-field counts](../manifests/joint_calibration_targeted_v1_evidence_v2.json)
- [Independent blue-source inspection](../manifests/blue_outlier_source_3601173023628559616.json)
- [Candidate v3 preservation check](../manifests/cold_v4_candidate_v3_validation.json)
- [Expanded targeted CMD](../figures/joint_calibration/03_targeted_full_cmd.png)
