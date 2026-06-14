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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mlip.utils.jax_utils import segment_sum
from mlip.utils.pallas_segment_sum import (
    deterministic_segment_sum_pallas,
    max_degree,
    pallas_segment_sum_available,
)

pytestmark = pytest.mark.skipif(
    not pallas_segment_sum_available(), reason="requires a GPU backend"
)

N_EDGES = 256
N_NODES = 16
RES_SIZE = 9
CHANNELS = 8


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def inputs(key):
    k1, k2 = jax.random.split(key)
    data = jax.random.normal(k1, (N_EDGES, RES_SIZE, CHANNELS))
    segment_ids = jax.random.randint(k2, (N_EDGES,), 0, N_NODES)
    return data, segment_ids


def test_matches_segment_sum(inputs):
    data, segment_ids = inputs
    expected = jax.ops.segment_sum(data, segment_ids, num_segments=N_NODES)
    actual = deterministic_segment_sum_pallas(data, segment_ids, N_NODES)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_deterministic_across_calls(inputs):
    data, segment_ids = inputs
    fn = jax.jit(
        lambda d, s: deterministic_segment_sum_pallas(d, s, N_NODES)
    )
    out1 = fn(data, segment_ids)
    out2 = fn(data, segment_ids)
    assert jnp.array_equal(out1, out2)


def test_gradient_matches_segment_sum(inputs):
    data, segment_ids = inputs

    def loss_pallas(d):
        return jnp.sum(deterministic_segment_sum_pallas(d, segment_ids, N_NODES) ** 2)

    def loss_baseline(d):
        return jnp.sum(jax.ops.segment_sum(d, segment_ids, num_segments=N_NODES) ** 2)

    grad_pallas = jax.grad(loss_pallas)(data)
    grad_baseline = jax.grad(loss_baseline)(data)

    np.testing.assert_allclose(grad_pallas, grad_baseline, atol=1e-4, rtol=1e-4)


def test_max_degree(inputs):
    _, segment_ids = inputs
    counts = jax.ops.segment_sum(
        jnp.ones_like(segment_ids), segment_ids, num_segments=N_NODES
    )
    assert max_degree(segment_ids, N_NODES) == jnp.max(counts)


def test_segment_sum_pallas_backend(inputs):
    data, segment_ids = inputs
    expected = jax.ops.segment_sum(data, segment_ids, num_segments=N_NODES)
    actual = segment_sum(
        data,
        segment_ids,
        N_NODES,
        deterministic=True,
        deterministic_backend="pallas",
    )
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_segment_sum_unknown_backend_raises(inputs):
    data, segment_ids = inputs
    with pytest.raises(ValueError):
        segment_sum(
            data,
            segment_ids,
            N_NODES,
            deterministic=True,
            deterministic_backend="bogus",
        )


def test_segments_exceeding_max_neighbors_drop_excess(key):
    """Documents the max_neighbors contract: items beyond it are silently dropped."""
    n_edges = 8
    n_nodes = 1
    data = jnp.ones((n_edges, 4))
    segment_ids = jnp.zeros((n_edges,), dtype=jnp.int32)

    result = deterministic_segment_sum_pallas(
        data, segment_ids, n_nodes, max_neighbors=4
    )
    np.testing.assert_allclose(result, jnp.full((1, 4), 4.0))
