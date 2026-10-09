# ReLD Cross-Scale Parallel Inference

This package keeps the ReLD PI implementation separate from the untouched
original model under `backbones/reld/CVRP`:

- `pi/data.py`: CVRPLIB loading, normalization, bucketing, and padding.
- `pi/model.py`: padding-aware encoder and decoder using the original checkpoint weights.
- `pi/env.py`: mixed-size CVRP rollout, augmentation, POMO masking, cost, and legality checks.
- `pi/backend.py`: shared inference used by both the adapter and command-line tool.
- `pi/tester.py`: bucket-level inference and augmentation/POMO reduction.
- `tools/run_reld_pi.py`: command-line entry point and CSV reporting.

Run from the repository root:

```bash
python -m tools.run_reld_pi \
  --data /path/to/CVRPLib-Set-X \
  --buckets 200:100:8 500:100:8 1000:100:8
```

Each bucket item is `upper_bound:pomo_size:aug_factor`. For example,
`1000:50:1` assigns POMO width 50 and no augmentation to instances with at most 1000
customers that were not assigned to a smaller bucket.

The dense tensor POMO width is `min(bucket pomo_size, maximum customer count in the
padded bucket)`. Each instance keeps only `min(bucket pomo_size, its real customer count)`
valid POMO lanes; extra lanes are excluded from the best-of-POMO reduction.

The CS-PIF adapter is exported from this package, while `pi/` owns the
padding-aware data, environment, model, and standalone tester. The PI path
supports the deterministic diverse-first-step setting used by the supplied
ReLD configuration (`forcing_first_step: false`).

The decoder computes only the current-node-to-candidate distances needed at each
step. It does not retain a full `[batch, nodes, nodes]` distance matrix.
