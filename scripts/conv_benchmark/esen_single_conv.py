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

"""A stripped-down eSEN edge-wise convolution for benchmarking.

`Edgewise` (`mlip.models.esen.layer.Edgewise`) applies *two* `SO2Convolution`s with a
gate activation in between. MACE's `O3MessagePassingBlock`, by contrast, performs a
*single* tensor product per message-passing step. `EdgewiseSingleConv` mirrors
`Edgewise` but with the activation and second `SO2Convolution` removed, so that the
"full message passing" benchmarks for MACE/e3j, MACE/Gaunt-TP and eSEN all cover the
same amount of work: gather -> rotate -> single tensor product/convolution -> rotate
back -> scatter.
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from mlip.graph import Graph
from mlip.models.esen.esen_helpers import NODE_OFFSET, SO2Convolution
from mlip.models.esen.moe import expand_graph_coeffs_to_edges, get_graph_moe_coefficients
from mlip.utils.jax_utils import segment_sum
from mlip.utils.pallas_segment_sum import DEFAULT_MAX_NEIGHBORS


class EdgewiseSingleConv(nn.Module):
    """Single-`SO2Convolution` edge-wise convolution (no gate / second conv).

    Attributes:
        sphere_channels: The number of channels for the node embedding.
        l_max: Highest harmonic order included in the Spherical Harmonics series.
        m_max: Maximum order of the spherical harmonics to include in the embedding.
        edge_channels_list: The list of channels for the edge embedding's radial MLP.
        mapping_reduced: The mapping of the spherical harmonics to the reduced set of
                        coefficients.
    """

    sphere_channels: int
    l_max: int
    m_max: int
    edge_channels_list: Sequence[int]
    mapping_reduced: object
    num_experts: int | None = None
    deterministic_scatter_ops: bool = False
    deterministic_scatter_backend: str = "dense"
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS

    def setup(self) -> None:
        self.so2_conv = SO2Convolution(
            sphere_channels=2 * self.sphere_channels,
            m_output_channels=self.sphere_channels,
            l_max=self.l_max,
            m_max=self.m_max,
            mapping_reduced=self.mapping_reduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=None,
            num_experts=self.num_experts,
        )

    def __call__(self, graph: Graph) -> Graph:
        """Single-convolution edge-wise message passing.

        Steps:
          1) gather source/target node features -> concat along channels
          2) rotate (align with edge)
          3) SO2 conv
          4) apply envelope
          5) rotate back
          6) scatter-add to destination nodes

        Updated features in this function:
        - node-wise features: graph.nodes.features["latent"]
        """
        node_feats = graph.nodes.features["latent"]
        edge_feats = graph.edges.features["latent"]
        senders, receivers = graph.senders, graph.receivers
        wigner_and_m_mapping = graph.edges.features["wigner_and_m_mapping"]
        wigner_and_m_mapping_inv = graph.edges.features["wigner_and_m_mapping_inv"]
        edge_envelope = graph.edges.features["envelope"]

        num_nodes = node_feats.shape[0]

        source_node_feats = node_feats[senders]
        target_node_feats = node_feats[receivers]

        # Concatenate along channels
        node_feats = jnp.concatenate([source_node_feats, target_node_feats], axis=-1)

        # Rotate to edge frame: bmm -> einsum
        node_feats = jnp.einsum("eij,ejc->eic", wigner_and_m_mapping, node_feats)

        edge_moe_coeffs = None
        if self.num_experts is not None:
            graph_moe_coeffs = get_graph_moe_coefficients(graph, self.num_experts)
            edge_moe_coeffs = expand_graph_coeffs_to_edges(
                graph_moe_coeffs, graph.n_edge, total_edges=graph.senders.shape[0]
            )

        # Single SO2 convolution (the eSEN analog of a tensor product)
        edge_messages = self.so2_conv(node_feats, edge_feats, edge_moe_coeffs)

        # Envelope per-edge
        edge_messages = edge_messages * edge_envelope

        # Rotate back
        edge_messages = jnp.einsum(
            "eij,ejc->eic", wigner_and_m_mapping_inv, edge_messages
        )

        # Scatter-add onto destination nodes
        dst_adj = (receivers - NODE_OFFSET).astype(jnp.int32)
        node_feats = segment_sum(
            edge_messages,
            dst_adj,
            num_segments=num_nodes,
            deterministic=self.deterministic_scatter_ops,
            deterministic_backend=self.deterministic_scatter_backend,
            max_neighbors=self.max_neighbors,
        )

        return graph.update_node_features(latent=node_feats)
