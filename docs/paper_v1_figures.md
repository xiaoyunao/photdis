# Photdis v1 candidate article figures

Six plates are new evidence or revised layouts with English labels. Two
plates reproduce the supplied v1 source figures from their existing data.
The archived v3 plots retain their Pal 5 settings. The new plates use the
same type family and method colors, with sizes chosen for a 7-inch
double-column image.

| PDF | Scientific role | Caption details for the author |
|---|---|---|
| [Catalogue surface density](../figures/paper_v1_evidence/f21_catalogue_density.pdf) | Compact replacement for F21 | Owned parent, full PDF and core, left to right. Galactic Aitoff. Source count divided by random-point effective unmasked imaging area. True zero is dark blue and cells with <10% effective area are white. Shared scale. |
| [Observable coverage](../figures/paper_v1_evidence/observable_support.pdf) | Which apparent color–magnitude regions are populated and intersect the performance sample? | Visual window spans the shown intrinsic g−r and observed r limits. The reference count panel uses the SpecDis F15 pairs. The rightmost panel uses cells supported by either benchmark and full multivariate cell membership, projected to the plot. Native sky and cuts are retained. The two systems share each column’s scale. The first three colorbars start at the second percentile of nonzero cell counts or one, whichever is larger. |
| [Same-star distance comparison](../figures/paper_v1_evidence/f15_same_star_distances.pdf) | Direct evidence for the conditional scatter advantage | Same stars in the first two columns, identical count color scale within a reference, third column shows normalized residual distributions. Preserves estimates outside the reference 2–5 kpc slice. The table records display windows and any excluded extremes. |
| [Colour × native-region differences](../figures/paper_v1_evidence/paired_colour_nmad.pdf) | Where does the paired result persist? | Reused v1 plot using the frozen 2,000-draw differences. No new bootstrap. |
| [Spectral and signed-Gaia PDF diagnostics](../figures/paper_v1_evidence/pdf_calibration.pdf) | How does the full distribution behave by stellar class? | Dashed horizontal reference is uniform per 0.1 CDF bin. Top rows use all qualifying r=18–20 spectral-reference distances. Lower row uses signed Gaia and separates measured S/N windows. Convolution assumes Gaussian reported errors. |
| [Native optical luminosity residuals](../figures/paper_v1_evidence/native_luminosity_residuals.pdf) | Where do luminosity-branch degeneracies appear? | Paired projection of the same r=18–20 SpecDis parent, split by system. Pixels with fewer than 10 references are blank. Solid black, dotted navy and dashed green trace populated loci of ordinary dwarfs, metal-poor dwarfs and intermediate gravity where present. Both references enter the matched-condition table. Shared reference distance in CMD ordinate and residual introduces covariance. |
| [Stellar-population distance migration](../figures/paper_v1_evidence/distance_migration_matrix.pdf) | How do luminosity branches cross distance cuts? | Reused v1 plot from exact SpecDis counts, preselection with valid QRF. Row-normalized by the reference-distance shell. Core retention belongs in the associated table. |
| [Heliocentric projections](../figures/paper_v1_evidence/f16_heliocentric_projections.pdf) | Optional spatial illustration | Raw projected counts based on medians <10 kpc. Three projections use the same census per row. Top full PDF, bottom core with their own scales. Depth, footprint and criteria affect the shape. |

The new display and scientific tables are in
[manifests/paper_v1_evidence](../manifests/paper_v1_evidence).
Full numerical plot inputs and the exact row-level benchmark bundle stay
on NERSC. The original large F21 is too tall for the manuscript's width
setting. The compact replacement retains its three denominators. Plot
aliases and modified insertion captions can be handled by GPT when
updating the manuscript.
