# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prototype: deterministic segment-sum via sort + bounded-window Pallas kernel.

Strategy:
  1. (plain JAX) Sort items by segment id -> sorted_data, and per-segment
     [start, count) offsets into the sorted array. Pad sorted_data with
     `max_neighbors` zero rows so dynamic windows never read out of bounds.
  2. (Pallas/Triton kernel) For each segment, dynamically load a window of
     `max_neighbors` rows starting at `start`, mask out rows >= count, and
     reduce with `jnp.sum` (fixed-order tree reduction, not atomics).
  3. Backward is a plain gather: grad_data[i] = grad_out[segment_ids[i]].
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _segment_sum_kernel(starts_ref, counts_ref, data_ref, out_ref, *, max_neighbors):
    seg = pl.program_id(0)
    start = starts_ref[seg]
    count = counts_ref[seg]

    rows = data_ref[pl.ds(start, max_neighbors), :]

    row_ids = jax.lax.broadcasted_iota(jnp.int32, rows.shape, 0)
    rows = jnp.where(row_ids < count, rows, jnp.zeros_like(rows))

    out_ref[...] = jnp.sum(rows, axis=0, keepdims=True)


def sorted_segment_sum_pallas(
    sorted_data: jax.Array,
    starts: jax.Array,
    counts: jax.Array,
    num_segments: int,
    max_neighbors: int,
    block_f: int = 128,
) -> jax.Array:
    """sorted_data: (E, F). starts/counts: (num_segments,) int32."""
    e, f = sorted_data.shape
    assert f % block_f == 0, f"F={f} must be a multiple of block_f={block_f}"

    padded = jnp.pad(sorted_data, ((0, max_neighbors), (0, 0)))

    grid_spec = pl.GridSpec(
        grid=(num_segments, f // block_f),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec((e + max_neighbors, block_f), lambda seg, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((1, block_f), lambda seg, j: (seg, j)),
    )

    out = pl.pallas_call(
        functools.partial(_segment_sum_kernel, max_neighbors=max_neighbors),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_segments, f), sorted_data.dtype),
        compiler_params=plgpu.CompilerParams(),
    )(starts, counts, padded)

    return out


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n_edges, n_nodes, f = 256, 16, 8
    data = jax.random.normal(key, (n_edges, f))
    seg_ids = jax.random.randint(key, (n_edges,), 0, n_nodes)

    order = jnp.argsort(seg_ids)
    sorted_ids = seg_ids[order]
    sorted_data = data[order]
    starts = jnp.searchsorted(sorted_ids, jnp.arange(n_nodes), side="left").astype(jnp.int32)
    ends = jnp.searchsorted(sorted_ids, jnp.arange(n_nodes), side="right").astype(jnp.int32)
    counts = ends - starts

    out = sorted_segment_sum_pallas(sorted_data, starts, counts, n_nodes, max_neighbors=32)
    ref = jax.ops.segment_sum(data, seg_ids, num_segments=n_nodes)

    print("max abs diff:", jnp.max(jnp.abs(out - ref)))
