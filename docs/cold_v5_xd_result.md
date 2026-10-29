# Cold-dwarf XD comparison: registered development decision

The [method and decision gate](cold_v5_xd_protocol.md) were registered before
the candidate comparisons. All three full-covariance XD fits converged on
the exact frozen v4 A/B/C training identities. Equal-field held-out
conditional negative log density selected 32 components: K=16/32/64 scores
were 0.02515/0.00626/0.03742. The single K=32 same-star comparison then
ran on allocated NERSC CPU job 60482510. The machine-readable selection,
comparison and decision are in `manifests/cold_v5_xd_*.json`; the plotted
comparison is `figures/cold_v5/02_xd_same_star_comparison.png`.

The two registered informative development fields used the unchanged
`grz_clean` plus WISE20 route, cold-dwarf box and support test. Their
accepted source identities are identical between models. Residual is
predicted minus Gaia-reference distance modulus. Intervals below are raw
model quantiles, without a new interval calibration.

| Field | Same stars | v4 center / NMAD | XD center / NMAD | v4 / XD \|Δμ\|>0.5 | v4 / XD raw 95% coverage |
|---|---:|---:|---:|---:|---:|
| RA205–215 | 1,257 | +0.0177 / 0.2241 | +0.0123 / 0.1978 | 11.61% / 12.25% | 94.59% / 89.58% |
| RA60–70 | 6,594 | −0.0059 / 0.2056 | −0.0136 / 0.1802 | 11.77% / 11.37% | 94.34% / 91.64% |

XD reduced residual NMAD by 0.0263 and 0.0255 mag respectively. Its raw
68% coverage was 59.43% and 64.15%, versus 68.10% and 71.81% for v4.
In RA205, XD's 95% lower/upper misses were 6.68%/3.74%, versus v4's
2.23%/3.18%. The small RA235
diagnostic field has only nine accepted objects and did not enter the
configuration choice or decision gate.

## Fixed-tier optical response and decision

The same +0.01 mag perturbation of one optical band is applied to the
measured input and final distance-modulus term, holding tier fixed. The
median absolute change in predicted distance modulus is:

| Field | Band | v4 | XD | XD / v4 |
|---|---|---:|---:|---:|
| RA205–215 | g | 0.0269 | 0.0610 | 2.27 |
| RA205–215 | r | 0.0388 | 0.0511 | 1.32 |
| RA60–70 | g | 0.0320 | 0.0591 | 1.85 |
| RA60–70 | r | 0.0418 | 0.0520 | 1.24 |

The predeclared gate required at least a 25% reduction for **both** bands
in **each** informative field, with center, NMAD and large-error tail
noninferior. Center, NMAD and tail checks passed; both optical response
checks failed in both fields. The mechanical decision is
**retain the frozen v4 route**. This closes the bounded K=16/32/64 XD
comparison. The XD posterior is a development diagnostic, not a released
distance or interval product. No candidate-triggered calibration or new
final-field scoring follows this failed gate.

The measured-error tertiles have equal source counts within each field.
Their raw 95% coverage (v4 / XD, ordered low to high measured error) is
RA205: 92.1/89.5%, 95.5/91.2%, 96.2/88.1%; RA60: 91.5/90.6%,
95.5/93.2%, 96.0/91.1%. XD's median 95% widths are shorter in all six
cells; the full widths, lower/upper misses and optical-color groups are in
the comparison manifest. These are conditional development diagnostics,
not calibrated probability performance.

## Color-dependent optical response

Allocated job 60483193 reran the fixed-tier +0.01 mag perturbations solely
to tabulate their color dependence. Its separate
`manifests/cold_v5_xd_color_response.json` records the frozen comparison
SHA256 and reproduces the same 1,257/6,594 accepted development stars.
The [color-response figure](../figures/cold_v5/03_xd_color_response.png)
shows the result. Values below are median absolute changes in distance
modulus, in mag.

| Field | Observed `(g−r)₀` | Stars | g: v4 / XD | r: v4 / XD |
|---|---|---:|---:|---:|
| RA205–215 | 0.9–1.2 | 217 | 0.0100 / 0.0192 | 0.0167 / 0.0395 |
| RA205–215 | 1.2–1.6 | 1,040 | 0.0397 / 0.0693 | 0.0445 / 0.0557 |
| RA60–70 | 0.9–1.2 | 1,072 | 0.0100 / 0.0198 | 0.0218 / 0.0409 |
| RA60–70 | 1.2–1.6 | 5,522 | 0.0439 / 0.0688 | 0.0473 / 0.0564 |

Both models respond more strongly in the redder occupied bin, and XD has
the larger g and r response in each populated field/bin. This reinforces
the registered rejection. No development stars occupied the 1.6–2.0 or
2.0–4.0 bins; the nine-star RA235 field remains diagnostic only. A new
cause study of the optical derivative and true color–luminosity relation
would need its own registered design.
