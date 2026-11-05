# Expanded A apparent magnitude control

The GPT Pro review of `63cdbaa` authorizes one development contrast.
Use the frozen expanded A distance prediction as the control and fit one
histogram QRF with inputs `[g0-r0, r0-z0, r0]`. Keep the expanded A training
order, legacy and new identities, Legacy Mg target, unweighted fit and forest
parameters. The fixed max_features fraction yields one candidate feature per
split in two dimensions and two in three dimensions, as specified by sklearn.
Both arms use the same identity-keyed five-band Sobol128 stream. Construct
both colors and r0 from one perturbed grz triplet and retain the existing g0
shift in the distance PDF. Reuse frozen baseline predictions and refined Gaia
scores on the exact evaluation identities.

Primary development checks are evolved and metal-poor dwarf distance
residuals, all other class risks, r=18–20 without W1, and all eligible signed
Gaia parallaxes. Report source by magnitude and sky strata, 68/95% agreement
with the reference estimates, and same-star rescue/harm with whole-cell
bootstrap. Show original RVS and corrected-domain cuts side by side. Old
cal/test are development evidence after the diagnostic review. Adoption
requires independent validation. r=22–25 remains sparsely tested.

Correct the coverage figure with separate actual class and distance evaluation
identities, keeping chosen distances and parallax-only evidence distinct.
Use the Pal 5 New Step 5 palette, font scale and figure proportions for revised
and new science panels. Frozen v1.4 outputs remain reproducible by commit.
