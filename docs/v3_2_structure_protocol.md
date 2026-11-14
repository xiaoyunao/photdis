# NGC288 two-dimensional quadrature

Carry forward the same 512 native DR11 PSF objects, 32 shared response
states, common distance prior, five background bases, spatial intensity
and calibration. Compare the full native likelihood with the prior-withdrawn
Gaussian moment likelihood using the same entry and nodes.

The odds-only audit holds distance order at 64 and compares odds orders
256, 512 and 1024 with the original criterion. For the distance extension
use distance orders 128 and 256. Evaluate the two native grid resolutions
from each authenticated likelihood product at the additional narrow-prior
means. Keep the 0.02-mag width. Compare at both native resolutions, retaining
the 0.02 relevant log-response and 0.003 class-probability thresholds.
Recomputed responses at the archived nodes must match to 1e-8 nat.

Independently integrate entry by summing the SED and binary mixture and
combining Gauss–Hermite narrow-prior integration with sequential Gaussian
survival integration. Use order 9 and 256 draws with seeds 328211 and
328212. Compare also with order 5 and 128 draws with seed 328211. Keep
the original absolute 0.0005 and relative 0.01 entry criteria and verify
closure on every previously used distance node with the archived entry.
First exercise the three original extinction/noise contexts for state 12.
Preserve every failed context and the frozen v3.1 products.

Once the added inputs are verified, compare each axis while holding the
other fixed and compare the successive joint grids. Keep the original
criteria (max membership probability 0.003, mean count 0.5, common
distance-modulus quantiles 0.03 mag and log evidence 0.05). Use all 32
states for the final checks. This is a seen-system numerical audit.

The final grid uses 256 distance nodes and 1024 odds nodes. Compare this
against 128 × 1024, 256 × 512 and 128 × 512 to isolate each axis and their
joint refinement. Tabulate 64 × 1024 to 128 × 1024 as an earlier distance
transition. Also compare baseline with the original coarse native grid,
the independent entry seed and the lower entry quadrature at 256 × 1024.
The preparation, axis, native and entry flags remain separately inspectable.
Report common-distance quantiles, response ESS, membership probability
and shared/conditional count covariance for both likelihood branches.

## Fixed-distance odds result

Job 61259650 completed on 512 PSFs and all 32 response states with 64
distance nodes. Increasing the odds order from 256 to 512 failed the
membership and mean-count checks in both branches. At 1024 versus 512,
both branches pass the registered fixed-distance comparison.

| Likelihood | Maximum membership change | Absolute mean-count change | Common modulus quantile change (mag) | Log evidence change (nat) |
|---|---:|---:|---:|---:|
| Full | 0.002786 | 0.0951 | 0 | −0.009020 |
| Gaussian | 0.002333 | 0.0763 | 0 | +0.009074 |

The member means at odds order 1024 are 121.335 and 127.817, respectively.
Distance-axis and joint certification still require the added-distance
responses and selection integrals. [Both transitions and receipt](../manifests/v3_2/structure_odds_audit/)
