# v2.3.1 — inference and evaluation correction of the frozen XD32 experiment

Authority: complete review in pro_review_569d6ff.md and user's instruction to
implement it. This is a correction, not a new blind experiment or model search.
Original v2_3_protocol.md remains immutable. Preserve 266774 training identities,
43914 unlabeled distance members, fitted32 components,179EM iterations and QRF.
All new samples require verified native DR11 TYPE=PSF. Figure text English only.

1. Correct effective T=Mg+epsilon_g for g_obs=Mg+mu+epsilon_g. The cross covariance
   with observed(gr,rz) is V_Mc+(sigma_g²,0). Independently test known SED weighted
   three-band distance and general non-diagonal Gaussian mixtures in band space,
   including posterior means, variances, weights, PDF masses and qnear.
2. Explicitly archive all old candidate PDFs, summaries and derived tables. Use
   inference-code/version stamps in every new candidate cache. Model byte hashes
   and all QRF output bytes remain unchanged; no fitting or threshold selection.
   Recompute all development/cal/reserved/application candidate-dependent outputs.
   Report paired old/new changes by optical measurement-error bins.
3. Replay exact unused-identity inventory from historical membership and source
   qualifications. Separate dynamic anchor-SNR distance admission from fixed
   covariates/other historical use. Report raw Gaia metrics, conditional metrics
   only in audited selection subsets, and forward draws through the same defined
   admission rule. Hold source quality/error/class/identity covariates fixed;
   do not claim this models spectroscopic targeting or total population selection.
   Never impose SNR<20 globally: optical-error/reference/class branches differ.
   Original raw-cal acceptance rules remain unchanged; new selection analysis
   explicitly supplemental. Already inspected reserved IDs are not a new holdout.
4. Promote SPDist new-ID near evidence alongside chosen, with unknown classes,
   shared-teacher and representative-population limits. Far full PDF mass within
   |mu-mu_ref|<1 and reference-error-convolved likelihood evaluated independently
   of auxiliary class tagging. QRF leaf class/luminosity and XD per-component
   constant frequencies are not equivalent conditional models.
5. Preserve poor-metal2055 original pairs/354unique controls. Report overall,
   per-side old masks and pair-conserving strata with both/one/neither retained
   counts. Match repetition is not independent data and not causal evidence.
   Report application evidence using dev-only and cal+dev counts side by side,
   under identical selection; evidence occupancy is not a quality gate.

End point: actual correction, source-backed comparisons and English figures;
recompute unchanged cal guards and judge this one candidate. Retain existing
deployment. No new candidate, monitoring, agents or whole-catalog expansion.
