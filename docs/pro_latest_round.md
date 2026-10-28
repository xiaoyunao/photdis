# Cold-v4 diagnostic round after the Pro review

## Scope and frozen method

This round followed the latest review of the `dc51856` application decision.
The cold-v4 forests, optical color box, 2/3/4-dimensional support domains,
strict Gaia cold-anchor definition, and `grz_clean` plus WISE S/N≥20
C→B→A route remained fixed. The earlier calibration and final-field results
were preserved. Already inspected RA35–45 and RA170–180 templates became
development diagnostics; newly registered fields supplied a one-time
native-transfer test.

The current output is a **cold-dwarf-hypothesis median distance** on the
tested high-quality population. The prior deep-distance intervals remain
unreleased as nominal 68%/95% products. Current color selection also lets
some DESI-confirmed giants through, especially in optical-only tier A.

## Main findings

| Question | Result | Evidence |
|---|---|---|
| Was foreground extinction corrected? | Yes. DR11 `MW_TRANSMISSION` converts nanomaggy `g,r,z,W1,W2` fluxes into dereddened AB magnitudes. Gaia labels and predicted distance modulus both use `g0`. Cross-field noise simulation transports flux with the source/target transmission ratio. | [Extinction audit](extinction_audit.md) |
| Can a uniform reduction of the catalog dust correction remove the native field contrast? | No. On the same three fields, reducing the subtracted extinction to 75% changes the north-minus-south residual contrasts by only 0.003–0.004 mag; the contrast remains even at zero correction. This is a bounded sensitivity test, not a 3D dust estimate. | [Dust sensitivity](cold_v4_dust_sensitivity.md) |
| Was the weak-parallax PIT affected by selection? | Yes. The revised diagnostic uses the forest-weighted absolute-magnitude distribution, convolves with the Gaia parallax error, and conditions on the observed corrected-parallax/S/N<10 cut. Negative parallaxes remain. Selected-PIT low tails are about 5.4%, 5.9%, 7.2% in the three diagnostic fields. | [Weak-parallax check](cold_v4_weak_parallax_v2.md) |
| Does the deep discrepancy follow stellar templates or the noise environment? | The paired 2×2 crossover found native template medians −0.009/+0.050 mag. Swapping noise fields on the same source changes medians by ≤0.011 mag, while degrading to `r=21–22` raises NMAD by about 0.09–0.10 mag and lowers acceptance by 10–13 percentage points. | [Template/noise crossover](cold_v4_crossover.md) |
| Do measured template properties explain the native offset? | Matching the old templates on Legacy colors, `A_g`, Gaia parallax, BP−RP and G left a positive center difference. The newly registered final fields reproduce a positive northern versus negative southern center; a six-feature outcome-blind match leaves +0.046/+0.076 mag north-minus-south. | [Population census](cold_v4_native_population.md), [registered transfer](cold_v4_native_transfer_final.md) |
| Are the systems and type checks conclusive for absolute distance? | Under WISE20, the same 456 wide-binary pairs retain a fainter-minus-brighter median +0.044 mag. Blanco 1 tests internal scatter, not its absolute zero point. DESI finds 12/32 cold giants selected in RA205–215 (11 tier A) and 14/14 metal-poor cold dwarfs excluded by the optical box in RA220–230. | [System validation](cold_v4_system_validation_v3.md), [type boundary](cold_v4_type_boundary.md) |

### Native-transfer field results

For `Δμ=μ̂−μ_Gaia`, the new southern final fields RA330–340 and RA20–30
have medians −0.0178 and −0.0520 mag (5,776 and 7,097 selected strict
cold anchors), while the preregistered northern RA130–140 field has
+0.0687 mag (2,941 anchors). Their nside64 sky-block median intervals do
not overlap. In the corrected-parallax ≥2 mas stratum the respective
medians are −0.022, −0.066 and +0.112 mag; the 1–2 mas medians are all
near zero and negative. The directional hypothesis was written before
the three final residuals were scored. Results, uncertainty and marginal
color/r/dust strata are in the [registered transfer report](cold_v4_native_transfer_final.md).

Gaia RUWE≤1.1 leaves the field contrast in place. The applied Gaia
zero-point medians differ by only ~0.003 mas across these fields; this
measured term does not account for a ~0.1 mag contrast at 2–3 mas. Usable,
unambiguous DESI dwarf metallicities number only 0, 13 and 35, so the
current spectroscopy does not characterize the full anchor populations.
The ≥2 mas contrast also persists among stars with tangential speed
<40 km/s and 40–80 km/s; the northern field is most positive among its
nearby-height stars. These Gaia-motion diagnostics do not identify a
metallicity or age distribution.

## Scientific decision

The native field-center variation is now an observed property of the
frozen model on these registered high-quality Gaia cold-dwarf samples.
The degradation experiment separately identifies noise and tier changes
as major contributors to deep scatter and selection loss. The field-center
cause remains to be separated among stellar population, photometric system,
extinction-map and Gaia-label effects. A single shift or interval multiplier
would mix those effects. The current route, forest and application scope
therefore stay fixed while the cause is mapped.

Six further disjoint 10° fields, spanning center Galactic latitudes
−87° to +79°, were registered in
`configs/cold_v4_native_transfer_sky_census.json` before any source rows
were read. Three informative positive-b fields have native residual
medians +0.039, +0.050 and +0.078 mag; the informative negative-b RA10
field has −0.009 mag. One negative-b field has 216 selected anchors and
a broad interval; another has zero under the fixed E(B−V)<0.03 cut.
The [sky census](cold_v4_sky_census.md) is explanatory evidence. Four
further fields were registered before those six scores were read, with
opposite signs of Galactic and ecliptic latitude. Two of those fields
have zero strict anchors under E(B−V)<0.03; one has 16 selected, and the
last has 680 with a near-zero median. This diagnostic cannot distinguish
the two sky-coordinate associations. The
registered native transfer fields above remain the one-time independent
test of the directional hypothesis. A future fitted correction requires
fresh calibration and final fields.

The [dust-sensitivity check](cold_v4_dust_sensitivity.md) kept Gaia labels,
models and route fixed while changing the fraction of catalog foreground
extinction subtracted in all five bands. A 25% reduction moves all three
field medians negative but barely changes their contrast. An additional
PS1-to-DECam optical comparison on these already inspected fields is
registered as an explanatory diagnostic; it does not create a new final
test or independently establish an optical zero point.

## Reproducibility and location

All full scans, joins, prediction and simulations ran on allocated NERSC
Perlmutter CPUs. Source catalogs, per-star tables and trained models remain
in `/pscratch/sd/y/yunao/photdis_pipeline/data/`. Programs, configs,
aggregate manifests, reports and selected figures are tracked in the
public GitHub repository and synchronized with this local checkout and
the NERSC checkout. `WORKLOG.md` records the Slurm job IDs, checks and
milestone commits.
