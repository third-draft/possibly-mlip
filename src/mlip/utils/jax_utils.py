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


import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from mlip.utils.pallas_segment_sum import (
    DEFAULT_MAX_NEIGHBORS,
    deterministic_segment_sum_pallas,
    pallas_segment_sum_available,
)


class TupleLeaf(tuple):
    """A tuple that is considered a leaf in a JAX pytree."""


def _deterministic_segment_sum(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    """Compute segment sum using a deterministic implementation.

    This function provides a deterministic alternative to `jax.ops.segment_sum`
    that avoids non-determinism from parallel reductions on GPU.

    Creates a one-hot matrix encoding the segment indices, and uses dense matrix
    multiplication to compute the segment sum using deterministic operations.

    Args:
        data: Array of shape `(num_items, ...)` containing the data to sum.
        segment_ids: Integer array of shape `(num_items,)` specifying which
            segment each item belongs to. Values should be in `[0, num_segments)`.
        num_segments: The total number of segments.

    Returns:
        Array of shape `(num_segments, ...)` where each segment contains the
        sum of all items belonging to that segment.
    """
    input_shape = data.shape
    num_items = input_shape[0]

    # Flatten all feature dimensions into one dimension
    flat_data = data.reshape(num_items, -1)

    # Create one-hot matrix encoding the segment indices
    mask = jax.nn.one_hot(segment_ids, num_segments, dtype=data.dtype)
    # HIGHEST precision avoids large errors from reduced-precision matmul on GPU.
    flat_result = jnp.dot(mask.T, flat_data, precision=jax.lax.Precision.HIGHEST)

    # Reshape back to original structure: (Segments, F1, F2...)
    output_shape = (num_segments,) + input_shape[1:]
    return flat_result.reshape(output_shape)


def segment_sum(
    data: e3nn.IrrepsArray | jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    deterministic: bool = False,
    deterministic_backend: str = "dense",
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
) -> e3nn.IrrepsArray | jnp.ndarray:
    """Compute segment sum with optional deterministic mode.

    Provides a universal replacement for `jax.ops.segment_sum`, with the option to
    replace non-deterministic parallel reductions with a deterministic alternative.

    Args:
        data: The data to scatter. Can be a regular JAX array or an `e3nn.IrrepsArray`.
        segment_ids: Integer array of shape `(num_items,)` specifying which
                     segment each item belongs to.
        num_segments: The total number of segments. Must be a concrete Python int
                      so that the output shape is statically known under jax.jit.
        deterministic: If `True`, uses a deterministic reduction operation. If `False`,
                       uses `jax.ops.segment_sum`.
        deterministic_backend: Which deterministic implementation to use when
            `deterministic=True`. `"dense"` (default) uses a one-hot matmul, O(num_items
            * num_segments) but works on any backend. `"pallas"` uses a Pallas/Triton
            kernel that sorts items by segment and reduces in O(num_items); it requires
            a GPU and that no segment has more than `max_neighbors` items (see
            `mlip.utils.pallas_segment_sum.max_degree` to check this for your data).
        max_neighbors: Static upper bound on items per segment, only used when
            `deterministic_backend="pallas"`.

    Returns:
        Array of shape `(output_size, ...)` where each segment contains the sum of all
        items belonging to that segment. The return type matches the input `data` type.
    """
    is_irreps = isinstance(data, e3nn.IrrepsArray)
    data_array = data.array if is_irreps else data

    def _standard_path(operands):
        d, s = operands
        return jax.ops.segment_sum(d, s, num_segments=num_segments)

    def _deterministic_path(operands):
        d, s = operands
        if deterministic_backend == "pallas":
            if not pallas_segment_sum_available():
                raise RuntimeError(
                    "deterministic_backend='pallas' requires a GPU backend, but "
                    f"the current default backend is {jax.default_backend()!r}."
                )
            return deterministic_segment_sum_pallas(
                d, s, num_segments=num_segments, max_neighbors=max_neighbors
            )
        if deterministic_backend != "dense":
            raise ValueError(
                f"Unknown deterministic_backend={deterministic_backend!r}, "
                "expected 'dense' or 'pallas'."
            )
        return _deterministic_segment_sum(d, s, num_segments=num_segments)

    # Branch on static Python bool: comes from model.config.
    if deterministic:
        result_array = _deterministic_path((data_array, segment_ids))
    else:
        result_array = _standard_path((data_array, segment_ids))

    if is_irreps:
        return e3nn.IrrepsArray(data.irreps, result_array)

    return result_array


def scatter_sum(
    data: e3nn.IrrepsArray | jnp.ndarray,
    num_elements_per_segment: jnp.ndarray,
    deterministic: bool = False,
) -> e3nn.IrrepsArray | jnp.ndarray:
    """Compute scatter sum with optional deterministic mode.

    Provides a universal replacement for `e3nn.scatter_sum`, with the option to replace
    non-deterministic parallel reductions on GPU with dense matrix multiplication.

    Assumes that `num_elements_per_segment` (`nel`) is provided instead of `dst`.

    Args:
        data: The data to scatter. Can be a regular JAX array or an `e3nn.IrrepsArray`.
        num_elements_per_segment: `(num_segments,)` integer array specifying how many
            items belong to each segment. Equivalent to `nel` in `e3nn.scatter_sum`.
        deterministic: If `True`, uses a deterministic reduction operation. If `False`,
                       uses `jax.ops.segment_sum`.

    Returns:
        Array of shape `(output_size, ...)` where each segment contains the sum of all
        items belonging to that segment. The return type matches the input `data` type.
    """
    num_segments = num_elements_per_segment.shape[0]
    segment_indices = jnp.repeat(
        jnp.arange(num_segments),
        num_elements_per_segment,
        total_repeat_length=data.shape[0],
    )
    return segment_sum(data, segment_indices, num_segments, deterministic=deterministic)
