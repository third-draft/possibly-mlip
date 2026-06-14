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

"""Dummy graph connectivity generators.

Shared between the MACE and eSEN graph builders so that both convolution layers are
benchmarked on the exact same `(senders, receivers)` topology.
"""

from typing import Literal

import jax
import jax.numpy as jnp

ConnectivityMode = Literal["random", "k_regular"]


def make_connectivity(
    n_nodes: int,
    n_edges: int,
    mode: ConnectivityMode,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Builds `(senders, receivers)` index arrays of shape `[n_edges]`.

    Args:
        n_nodes: Number of nodes in the dummy graph.
        n_edges: Number of (directed) edges in the dummy graph.
        mode: `"random"` draws senders/receivers uniformly at random (with
            `senders != receivers`). `"k_regular"` assigns edges round-robin to
            senders `0, 1, ..., n_nodes-1, 0, 1, ...` and offsets receivers by a fixed
            stride, giving every node (close to) the same out-degree -- useful for
            reproducible average-degree scaling.
        key: PRNG key, only used for `"random"`.

    Returns:
        `(senders, receivers)`, each an `int32` array of shape `[n_edges]`.
    """
    if mode == "random":
        k1, k2 = jax.random.split(key)
        senders = jax.random.randint(k1, (n_edges,), 0, n_nodes, dtype=jnp.int32)
        offsets = jax.random.randint(k2, (n_edges,), 1, n_nodes, dtype=jnp.int32)
        receivers = jnp.mod(senders + offsets, n_nodes)
        return senders, receivers

    if mode == "k_regular":
        senders = jnp.mod(jnp.arange(n_edges, dtype=jnp.int32), n_nodes)
        stride = jnp.arange(n_edges, dtype=jnp.int32) // n_nodes + 1
        receivers = jnp.mod(senders + stride, n_nodes)
        return senders, receivers

    raise ValueError(f"Unknown connectivity mode: {mode!r}")
