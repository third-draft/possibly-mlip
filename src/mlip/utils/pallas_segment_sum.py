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

"""A deterministic, GPU-accelerated segment-sum based on a Pallas/Triton kernel.

`jax.ops.segment_sum` on GPU uses atomic-add based parallel reductions, which are
fast but not reproducible run-to-run (summation order varies). The pure-JAX
deterministic alternative (one-hot matmul, see `_deterministic_segment_sum` in
`jax_utils.py`) is reproducible but costs O(num_items * num_segments) FLOPs/memory.

This module computes the segment sum deterministically in O(num_items) by:
  1. Sorting items by segment id (plain JAX, `jnp.argsort` - stable, deterministic).
  2. For each segment, summing its (contiguous, after sorting) items with a Pallas
     kernel that loads a fixed-size window and reduces with `jnp.sum` (a fixed
     compile-time reduction tree, not atomics -> deterministic and reproducible).

The backward pass is a plain gather (`grad_out[segment_ids]`), implemented via
`jax.custom_vjp` rather than differentiating through the kernel.

Constraint: every segment must contain at most `max_neighbors` items. This is a
static kernel parameter (default 128) chosen to comfortably cover typical MLIP
neighbor-list degrees. Use `max_degree` to check this assumption against real data.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

DEFAULT_MAX_NEIGHBORS = 128


def pallas_segment_sum_available() -> bool:
    """Whether the Pallas/Triton deterministic segment-sum kernel can run here."""
    return jax.default_backend() == "gpu"


def max_degree(segment_ids: jax.Array, num_segments: int) -> jax.Array:
    """Returns the size of the largest segment (i.e. max neighbor count)."""
    counts = jax.ops.segment_sum(
        jnp.ones_like(segment_ids), segment_ids, num_segments=num_segments
    )
    return jnp.max(counts)


def _largest_pow2_divisor(value: int, cap: int = 128) -> int:
    divisor = 1
    while divisor * 2 <= cap and value % (divisor * 2) == 0:
        divisor *= 2
    return divisor


def _segment_sum_kernel(starts_ref, counts_ref, data_ref, out_ref, *, max_neighbors):
    seg = pl.program_id(0)
    start = starts_ref[seg]
    count = counts_ref[seg]

    rows = data_ref[pl.ds(start, max_neighbors), :]

    row_ids = jax.lax.broadcasted_iota(jnp.int32, rows.shape, 0)
    rows = jnp.where(row_ids < count, rows, jnp.zeros_like(rows))

    out_ref[...] = jnp.sum(rows, axis=0, keepdims=True)


def _sorted_segment_sum_pallas(
    sorted_data: jax.Array,
    starts: jax.Array,
    counts: jax.Array,
    num_segments: int,
    max_neighbors: int,
) -> jax.Array:
    """sorted_data: (E, F) with E rows sorted by segment. starts/counts: (num_segments,)."""
    e, f = sorted_data.shape
    block_f = _largest_pow2_divisor(f)

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

    return pl.pallas_call(
        functools.partial(_segment_sum_kernel, max_neighbors=max_neighbors),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_segments, f), sorted_data.dtype),
        compiler_params=plgpu.CompilerParams(),
    )(starts, counts, padded)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _deterministic_segment_sum_pallas_flat(
    flat_data: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
    max_neighbors: int,
) -> jax.Array:
    order = jnp.argsort(segment_ids)
    sorted_ids = segment_ids[order]
    sorted_data = flat_data[order]

    seg_arange = jnp.arange(num_segments)
    starts = jnp.searchsorted(sorted_ids, seg_arange, side="left").astype(jnp.int32)
    ends = jnp.searchsorted(sorted_ids, seg_arange, side="right").astype(jnp.int32)
    counts = ends - starts

    return _sorted_segment_sum_pallas(
        sorted_data, starts, counts, num_segments, max_neighbors
    )


def _fwd(flat_data, segment_ids, num_segments, max_neighbors):
    out = _deterministic_segment_sum_pallas_flat(
        flat_data, segment_ids, num_segments, max_neighbors
    )
    return out, segment_ids


def _bwd(num_segments, max_neighbors, segment_ids, g):
    del num_segments, max_neighbors
    grad_flat_data = jnp.take(g, segment_ids, axis=0)
    return (grad_flat_data, None)


_deterministic_segment_sum_pallas_flat.defvjp(_fwd, _bwd)


def deterministic_segment_sum_pallas(
    data: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
) -> jax.Array:
    """Deterministic segment sum via sort + Pallas windowed reduction.

    Args:
        data: Array of shape `(num_items, ...)`.
        segment_ids: Integer array of shape `(num_items,)` in `[0, num_segments)`.
        num_segments: Static number of output segments.
        max_neighbors: Static upper bound on the number of items per segment.
            Segments with more than `max_neighbors` items will silently drop the
            excess. Use `max_degree` to check this assumption against real data.

    Returns:
        Array of shape `(num_segments, ...)`.
    """
    input_shape = data.shape
    flat_data = data.reshape(input_shape[0], -1)
    flat_result = _deterministic_segment_sum_pallas_flat(
        flat_data, segment_ids, num_segments, max_neighbors
    )
    return flat_result.reshape((num_segments,) + input_shape[1:])
