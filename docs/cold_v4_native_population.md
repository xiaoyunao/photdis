# Gaia-template population check for the cold-v4 crossover

The exact 500 source identities per field from the frozen crossover were
rejoined to their Gaia and Legacy diagnostics on a NERSC CPU node. The
identity hashes match the crossover. The detailed small result is in
`manifests/cold_v4_native_population.json`; the binned diagnostic is in
`figures/cold_v4/07_native_population.png`. No forest, route, interval or
field role was changed.

| Property, median | RA35–45 | RA170–180 |
|---|---:|---:|
| Native Δμ (mag) | −0.008 | +0.050 |
| Corrected Gaia parallax (mas) | 2.542 | 2.634 |
| Corrected parallax S/N | 29.65 | 29.32 |
| Applied Gaia zero-point term (mas) | −0.0319 | −0.0373 |
| Gaia RUWE | 1.041 | 1.017 |
| Gaia G (mag) | 17.26 | 16.92 |
| Gaia BP−RP (mag) | 2.304 | 2.188 |
| Legacy r₀ (mag) | 17.43 | 17.01 |
| Legacy source E(B−V) | 0.0181 | 0.0222 |

The medians of corrected parallax and parallax S/N are similar, while the
RA170–180 templates are somewhat brighter and have different color and
extinction distributions. In the 1–2 mas parallax bin, native residual
medians are −0.008 and −0.010 mag. At parallax ≥2 mas, the medians are
−0.009 and +0.075 mag. The latter bin contains 342 and 351 sources. The
shift is therefore concentrated among these closer anchors in this
500-source diagnostic.

At Gaia BP−RP 1.5–2.3, native medians are −0.003 and +0.058 mag; at
2.3–3.0, they are −0.043 and +0.051 mag. The same residual trend appears
across two well populated color ranges. The 8/14 objects with BP−RP≥3.0
are too few to define the red tail.

The applied zero-point terms and RUWE distributions describe the Gaia
labels; they do not isolate a remaining parallax systematic. The next
outcome-blind match adds corrected parallax, Gaia color and Gaia G to the
Legacy color/A_g matching. Its exact methods and small results are in
`manifests/cold_v4_native_population_match.json`. The field difference
remains +0.054 mag after color/A_g/parallax matching (460 pairs, bootstrap
95% +0.019 to +0.089), +0.062 mag after color/A_g/parallax/BP−RP matching
(458 pairs, +0.024 to +0.101), and +0.063 mag after color/A_g/G matching
(465 pairs, +0.028 to +0.096). These comparisons rule out a simple shift
in the measured distributions used for matching. They do not distinguish
unmeasured stellar-population structure from spatial Gaia or Legacy
systematics. Four separately registered 10° fields will test native transfer
across more sky before any center correction is considered.
