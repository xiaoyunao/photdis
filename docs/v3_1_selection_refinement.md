# Numerical amendment: hard candidate-support quadrature

The pre-registered median error context failed: max per-kernel/basis change
128→512 was.03665 and independent512 seeds differed by.03162, versus the
unchanged.003 tolerance. The.1→.05mag grid change was.002894. Preserve all
original tables and flags; do not fit candidate-population counts from them.

Use the identical truncated-Gaussian Sobol integral, original brightness/error
thresholds and frozen QRF tenth-neighbor support. Replace only the exact
Euclidean neighbor-query backend after a real-model equivalence audit. Omit
support queries whose individual Gaussian survival weights are<=1e-14, storing
their summed upper bound. This bounds pruning error only; it does not excuse
QMC integration error. Mathematical comparison to the original implementation
must bracket every pruned estimate, and the real median context must reproduce
the original128-point curves to1e-13 absolute including the bound.

For each original minimum/median/maximum error context, use2048 then8192 Sobol
points, the same two seeds317101/317102, grid.05mag. Refine all kernels together
at each level; never select by scientific recovery scores. Reuse per-kernel
checkpoints with immutable input hashes. Stop at the first level where both
the previous-level maximum per-kernel/basis change and independent-seed change
are<.003. Audit that accepted level on.025mag with seed317101; require the same
.003 limit. A failure at the fixed bound remains an unresolved numerical flag,
not permission to relax tolerance, drop a component, or claim a matched fit.

Parallelism and per-kernel caching change resource use only. These three pilot
contexts do not by themselves certify interpolation across the whole candidate
sample; that production design and its actual-context checks remain required.
