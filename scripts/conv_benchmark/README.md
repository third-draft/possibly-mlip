# Convolution benchmark: MACE (e3j / Gaunt-TP) vs eSEN

Compares the convolution / tensor-product step of MACE (e3j Clebsch-Gordan backend,
or Gaunt tensor-product backend) against eSEN's SO(2)-reduced convolution, on dummy
graphs of configurable size and connectivity.

Two things are measured for each backend:

- **`*_tp_only`**: the raw per-edge tensor product / convolution, with no
  gather/scatter (`e3j.core.TensorProduct` for `mace_e3j`, `GauntTensorProduct` for
  `mace_gaunt`, `SO2Convolution` for `esen`).
- **`*_full`**: the full message-passing convolution (gather -> rotate/TP -> scatter,
  with surrounding linears/MLPs).

Because eSEN's `Edgewise` normally applies *two* `SO2Convolution`s with a gate
activation in between (more work than MACE's single tensor product per layer),
`esen_single_conv_full` (`esen_single_conv.py`) provides a single-convolution variant
of `Edgewise` for an apples-to-apples comparison with `mace_e3j_full` /
`mace_gaunt_full`.

Parameters are randomly initialized independently for each layer -- there is no
attempt at numerical equivalence between architectures, only matched problem sizes
(`n_nodes`, `n_edges`, connectivity, channel count, `l_max`).

## Usage

```bash
python -m scripts.conv_benchmark.run \
    --n-nodes 1024 --n-edges 16384 \
    --num-channels 64 --l-max 3 \
    --connectivity k_regular \
    --mode both
```

Key arguments:

- `--n-nodes` / `--n-edges`: dummy graph size.
- `--connectivity {random,k_regular}`: edge topology (shared between MACE and eSEN
  graphs).
- `--num-channels`: shared channel count (MACE `num_channels`, eSEN
  `sphere_channels`/`hidden_channels`). On GPU, `e3j.core.TensorProduct` requires this
  to be a multiple of 32.
- `--l-max` / `--m-max`: max spherical harmonic degree (`m-max` is eSEN-only,
  defaults to `l_max`).
- `--num-rbf`, `--edge-channels`: radial/edge embedding sizes.
- `--models`: subset of `mace_e3j`, `mace_gaunt`, `esen`, `esen_single_conv` (default:
  all).
- `--mode {tp,full,both}`: which benchmarks to run.
- `--no-backward`: skip the forward+backward (`jax.grad`) timing.
- `--n-warmup`, `--n-iters`, `--seed`.

## Example result

`--n-nodes 1024 --n-edges 16384 --num-channels 64 --l-max 3 --connectivity k_regular --mode both`
on a single GPU:

```
name                     fwd (ms)   fwd+bwd (ms)    peak (MB)  in_use (MB)
--------------------------------------------------------------------------
mace_e3j_tp_only           0.9419         3.6148        551.7         68.3
mace_e3j_full              2.5658         8.5048       1438.0         16.6
mace_gaunt_tp_only         2.4105         4.3758       1438.0        149.9
mace_gaunt_full            3.5380         8.6714       1438.0         15.9
esen_so2_tp_only           0.7378         1.0638       1438.0        150.7
esen_full                  3.4761         7.5795       1438.0         53.4
esen_single_conv_full       2.8274         5.8426       1438.0         52.2
```

`peak (MB)` is a high-water mark over the whole process (it does not reset between
rows), so it is most useful for comparing across runs with different `--n-nodes` /
`--n-edges` / `--num-channels`, not between rows of the same run. `in_use (MB)` after
each call is more indicative of that call's own footprint.
