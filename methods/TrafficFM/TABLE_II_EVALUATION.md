# Table II probabilistic evaluation

`table2_probabilistic_evaluator.py` is the table-generation entry point. It
uses the existing fixed STD-MAE 144-to-144 indices and never rebuilds a split.
For every test window it draws exactly 10 stochastic TrafficFM trajectories.
Every trajectory and the target are evaluated after inverse transformation to
the original speed scale.

For a valid horizon/node value, empirical CRPS is

`mean_s |x_s-y| - 0.5 * mean_{s,s'} |x_s-x_s'|`.

Validity is `isfinite(target) and abs(target) > 1e-5` on the original scale.
CRPS and MAE are averaged over valid test-window/node values. For 1-WD, the
ten trajectory values, test windows, and nodes are concatenated in that order
for the prediction distribution; target test-window/node values are similarly
concatenated. SciPy's exact one-dimensional Wasserstein distance is then
computed once per horizon. It is not an average of per-window distances.
`Average` is the arithmetic mean of the reported horizon values.

Deterministic baselines use the separately documented validation-calibrated
conversion: retain the point forecast as the location, estimate a nodewise
horizon-144 residual RMS on validation, and draw 10 seeded Gaussian samples
for CRPS. Their legacy 1-WD is the pooled W1 of point-forecast and target
values, as recorded in the deterministic scorer metadata. These values are
labelled converted uncertainty, not native model samples.

Example commands:

```bash
export TRAFFICFM_CHECKPOINT=/absolute/path/to/table2/metrla/checkpoint.pth
CUDA_VISIBLE_DEVICES=3 bash run_table2_metrla.sh

export TRAFFICFM_CHECKPOINT=/absolute/path/to/table2/pemsbay/checkpoint.pth
CUDA_VISIBLE_DEVICES=3 bash run_table2_pemsbay.sh
```

Results are written outside Git under
`/tmp/trafficfm_benchmark_runs/TrafficFM/TableII/`.
