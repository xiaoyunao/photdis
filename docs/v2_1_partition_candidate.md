# Prespecified single candidate after diagnostics

Decision made2026-11-10 before fitting this candidate. Broad optical probe does
not improve over controlled QRF partitions. The matched grzW1 probe gains
0.012374 nats, positive in7/9 development sky cells; paired cell bootstrap
interval0.008081–0.017134. This motivates exactly one W1 partition-loss test,
not a model-family search. Existing development results inform this choice;
its subsequent test remains development, never independent final validation.

Both arms train on the same68004 verified-PSF, reliable-class, distance-qualified
objects in information_train.npz. Keep g-r,r-z,z-W1, Mg targets, natural unit
leaf-member mass, histogram edges, Sobol128 and shared g error identical.
Both use128trees, leaf32, depth20, max_features0.75, bootstrapFalse, seed20261102.
Control uses existing regression MSE partitions. Candidate uses balanced
three-class Gini partitions; class balance influences splits only, not final
distance histogram weights. Keep all evaluation classes including unknown.
Evaluation is the identical PSF i+W1+W2-available cal/development distance-evidence
subset. The training-label restriction and band coverage limit generalization.
No true evaluation class, FeH, sky, distance or parallax enters prediction.

The single candidate is tested once. On cal, require full observed-parallax mean
log score>=control; all/evolved severe rates<=control; dwarf severe<=control+0.005;
qnear>0.95 far contamination<=control+0.005 and near completeness>=control-0.02.
No per-hindsight-class PIT/68% equality criterion. Report all scores and all
negative effects on unchanged development identities even if cal rejects.
No threshold adjustment, parameter tuning, additional candidate or deployment.
Full release still requires independent labels/system-member audit; five
provisional reserved identities are insufficient.
