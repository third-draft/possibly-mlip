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

"""Dummy graph builders for the MACE and eSEN convolution layers.

Both builders take the same `(n_nodes, n_edges, senders, receivers)` topology (see
`connectivity.py`) so that the two architectures are benchmarked on identical problem
sizes. Feature *values* are random and weights are randomly initialized -- only the
shapes/irreps matter for timing.
"""

from dataclasses import dataclass

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.esen.blocks import EsenEmbeddingBlock
from mlip.models.esen.coefficient_mapping import CoefficientMapping


@dataclass(frozen=True)
class BenchmarkShapes:
    """A "comparable" shape configuration shared between MACE and eSEN.

    The correspondence chosen here is:

    * MACE `num_channels` <-> eSEN `sphere_channels` == `hidden_channels`.
    * MACE `source_irreps = e3nn.Irreps.spherical_harmonics(l_max)` so that the
      per-channel feature dimension is `(l_max + 1)**2`, matching eSEN's node-feature
      layout `[n_nodes, (l_max + 1)**2, sphere_channels]`.
    * `l_max` is shared. `m_max` is eSEN-only (defaults to `l_max`, i.e. no
      m-truncation, for the closest comparison to MACE's full SO(3) tensor product).
    """

    num_channels: int
    l_max: int
    m_max: int
    num_rbf: int
    edge_channels: int

    @property
    def source_irreps(self) -> e3nn.Irreps:
        return e3nn.Irreps.spherical_harmonics(self.l_max)

    @property
    def interaction_irreps(self) -> e3nn.Irreps:
        return e3nn.Irreps.spherical_harmonics(self.l_max)

    @property
    def sph_size(self) -> int:
        return (self.l_max + 1) ** 2

    @property
    def edge_channels_list(self) -> list[int]:
        return [self.num_rbf + 2 * self.edge_channels, self.edge_channels, self.edge_channels]


def default_shapes(
    num_channels: int = 32,
    l_max: int = 2,
    m_max: int | None = None,
    num_rbf: int = 16,
    edge_channels: int | None = None,
) -> BenchmarkShapes:
    """Builds a `BenchmarkShapes` with sensible defaults.

    Args:
        num_channels: Shared channel count (MACE `num_channels`, eSEN
            `sphere_channels`/`hidden_channels`).
        l_max: Shared maximum spherical harmonic degree.
        m_max: eSEN m-truncation. Defaults to `l_max` (no truncation).
        num_rbf: Number of radial basis functions (shared).
        edge_channels: eSEN edge embedding channels. Defaults to `num_channels`.
    """
    return BenchmarkShapes(
        num_channels=num_channels,
        l_max=l_max,
        m_max=l_max if m_max is None else m_max,
        num_rbf=num_rbf,
        edge_channels=num_channels if edge_channels is None else edge_channels,
    )


def build_mace_graph(
    key: jax.Array,
    n_nodes: int,
    n_edges: int,
    senders: jax.Array,
    receivers: jax.Array,
    shapes: BenchmarkShapes,
) -> Graph:
    """Builds a dummy `Graph` with the features expected by MACE interaction blocks.

    Populates `nodes.features["latent"]`, `edges.features["spherical_embedding"]`,
    and `edges.features["radial_embedding"]`, as consumed by
    `O3MessagePassingBlock.__call__` / `GauntMessagePassingBlock.__call__`.
    """
    k_node, k_vec, k_rbf = jax.random.split(key, 3)

    node_irreps = shapes.num_channels * shapes.source_irreps
    node_feats = e3nn.IrrepsArray(
        node_irreps, jax.random.normal(k_node, (n_nodes, node_irreps.dim))
    )

    vectors = jax.random.normal(k_vec, (n_edges, 3))
    spherical_embedding = e3nn.spherical_harmonics(
        e3nn.Irreps.spherical_harmonics(shapes.l_max), vectors, normalize=True
    )

    radial_embedding = e3nn.IrrepsArray(
        f"{shapes.num_rbf}x0e",
        jax.random.normal(k_rbf, (n_edges, shapes.num_rbf)),
    )

    graph = Graph(
        nodes=GraphNodes(positions=jnp.zeros((n_nodes, 3)), features={}),
        edges=GraphEdges(features={}),
        globals=GraphGlobals(cell=None, weight=1),
        senders=senders,
        receivers=receivers,
        n_node=n_nodes,
        n_edge=n_edges,
    )
    graph = graph.update_node_features(latent=node_feats)
    graph = graph.update_edge_features(
        spherical_embedding=spherical_embedding,
        radial_embedding=radial_embedding,
    )
    return graph


def build_esen_graph(
    key: jax.Array,
    n_nodes: int,
    n_edges: int,
    senders: jax.Array,
    receivers: jax.Array,
    shapes: BenchmarkShapes,
) -> Graph:
    """Builds a dummy `Graph` with the features expected by eSEN's `Edgewise`.

    Adapted from `scripts/profile_esen_layer.py::build_graph`.
    """
    k_node, k_vec = jax.random.split(key, 2)

    mapping_reduced = CoefficientMapping(l_max=shapes.l_max, m_max=shapes.m_max)
    embedding_block = EsenEmbeddingBlock(
        graph_cutoff_angstrom=5.0,
        l_max=shapes.l_max,
        num_species=10,
        num_charges=None,
        sphere_channels=shapes.num_channels,
        radial_envelope="polynomial",
        radial_basis="gauss",
        num_rbf=shapes.num_rbf,
        basis_width_scalar=2.0,
        cosine_cutoff=False,
        trainable_rbf=False,
        edge_channels=shapes.edge_channels,
        m_max=shapes.m_max,
        mapping_reduced=mapping_reduced,
        edge_channels_list=shapes.edge_channels_list,
    )

    node_feats = jax.random.normal(k_node, (n_nodes, shapes.sph_size, shapes.num_channels))
    edge_vectors = jax.random.normal(k_vec, (n_edges, 3))

    wigner_and_m_mapping, wigner_and_m_mapping_inv = embedding_block._get_rotmat_and_wigner(
        edge_vectors
    )

    graph = Graph(
        nodes=GraphNodes(positions=None, features={}),
        edges=GraphEdges(features={}),
        globals=GraphGlobals(cell=None, weight=None),
        senders=senders,
        receivers=receivers,
        n_node=n_nodes,
        n_edge=n_edges,
    )
    graph = graph.update_node_features(latent=node_feats)
    graph = graph.update_edge_features(
        latent=jnp.ones((n_edges, shapes.edge_channels)),
        vectors=edge_vectors,
        envelope=jnp.ones((n_edges, 1, 1)),
        wigner_and_m_mapping=wigner_and_m_mapping,
        wigner_and_m_mapping_inv=wigner_and_m_mapping_inv,
    )
    return graph
