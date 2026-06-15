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

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import initializers

from mlip.graph import Graph
from mlip.models.options import parse_activation
from mlip.models.radial_embedding import cosine_cutoff
from mlip.models.visnet.self_interaction import VisnetSelfInteractionBlock
from mlip.models.visnet.visnet_helpers import (
    LAYER_NORM_EPSILON,
    VEC_LAYER_NORM_EPSILON,
    VecLayerNorm,
)
from mlip.utils.jax_utils import segment_sum


class VisnetLayer(nn.Module):
    """VisnetLayer module representing a single vector-scalar interactive
    self-attention layer used in ViSNet.

    This layer performs equivariant message passing with multiple attention heads,
    supporting both scalar and vector features and including various normalization
    and activation options as configured.

    Relative to ``mlip.models.visnet_old.layer.VisnetLayer``, this version:

    - Allows the equivariant vector-feature channel width (``vec_channels``) to
      differ from the scalar channel width (``num_channels``). The
      rotation-invariant scalars derived from the vector pathway (``vec_dot``,
      ``w_dot``, both of width ``vec_channels``) are projected to
      ``num_channels`` via dedicated ``Dense`` layers (``vec_dot_proj``,
      ``w_dot_proj``) before being combined with the scalar pathway.
    - Optionally fuses same-input ``Dense`` projections (q/k/v, dk/dv/f,
      w_src/w_trg) into single matmuls followed by a split
      (``fuse_projections``).

    Attributes:
        num_heads: Number of attention heads.
        num_channels: Number of channels in the scalar input and output features.
        vec_channels: Number of channels in the equivariant vector features.
        activation: Activation function.
        attn_activation: Activation function for the attention heads.
        graph_cutoff_angstrom: Cutoff radius.
        vecnorm_type: Type of vector normalization to apply.
        last_layer: Whether this is the last layer of the network.
        l_max: Highest harmonic order included in the Spherical Harmonics series.
        fuse_projections: Whether to fuse same-input Dense projections into single
            matmuls followed by a split.
        deterministic_scatter_ops: Whether to use deterministic scatter operations.
        correlation: Correlation order of the equivariant many-body
            self-interaction applied to the aggregated scalar and vector
            messages (``node_feats`` and ``vec_out``), see
            :class:`~mlip.models.visnet.self_interaction.VisnetSelfInteractionBlock`.
            If `0` (default), the self-interaction block is disabled.
        num_species: Number of atomic species. Only used (and required to be
            accurate) when ``correlation > 0``.
    """

    num_heads: int
    num_channels: int
    vec_channels: int
    activation: str
    attn_activation: str
    graph_cutoff_angstrom: float
    vecnorm_type: str
    last_layer: bool
    l_max: int  # Required for input shape assertions.
    fuse_projections: bool = True
    deterministic_scatter_ops: bool = False
    correlation: int = 0
    num_species: int = 1

    def setup(self) -> None:
        """Initializes the VisnetLayer module."""
        assert self.num_channels % self.num_heads == 0, (
            f"The number of hidden channels ({self.num_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({self.num_heads})"
        )
        self.head_dim = self.num_channels // self.num_heads

        self.layernorm = nn.LayerNorm(epsilon=LAYER_NORM_EPSILON)
        self.vec_layernorm = VecLayerNorm(
            num_channels=self.vec_channels,
            norm_type=self.vecnorm_type,
            eps=VEC_LAYER_NORM_EPSILON,
        )
        self.act = parse_activation(self.activation)
        self.attn_act = parse_activation(self.attn_activation)
        self.cutoff_fn = functools.partial(
            cosine_cutoff, graph_cutoff_angstrom=self.graph_cutoff_angstrom
        )

        self.vec_proj = nn.Dense(
            features=self.vec_channels * 3,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
        )

        if self.fuse_projections:
            self.qkv_proj = nn.Dense(
                features=self.num_channels * 3,
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
            self.dkv_proj = nn.Dense(
                features=self.num_channels * (2 if self.last_layer else 3),
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
        else:
            self.q_proj = nn.Dense(
                features=self.num_channels,
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
            self.k_proj = nn.Dense(
                features=self.num_channels,
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
            self.v_proj = nn.Dense(
                features=self.num_channels,
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
            self.dk_proj = nn.Dense(
                features=self.num_channels,
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
            self.dv_proj = nn.Dense(
                features=self.num_channels,
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
            if not self.last_layer:
                self.f_proj = nn.Dense(
                    features=self.num_channels,
                    kernel_init=initializers.xavier_uniform(),
                    bias_init=initializers.zeros_init(),
                )

        self.s_proj = nn.Dense(
            features=self.vec_channels * 2,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.o_proj = nn.Dense(
            features=self.vec_channels + self.num_channels * 2,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.vec_dot_proj = nn.Dense(
            features=self.num_channels,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
        )

        if not self.last_layer:
            if self.fuse_projections:
                self.w_src_trg_proj = nn.Dense(
                    features=self.vec_channels * 2,
                    use_bias=False,
                    kernel_init=initializers.xavier_uniform(),
                )
            else:
                self.w_src_proj = nn.Dense(
                    features=self.vec_channels,
                    use_bias=False,
                    kernel_init=initializers.xavier_uniform(),
                )
                self.w_trg_proj = nn.Dense(
                    features=self.vec_channels,
                    use_bias=False,
                    kernel_init=initializers.xavier_uniform(),
                )
            self.w_dot_proj = nn.Dense(
                features=self.num_channels,
                use_bias=False,
                kernel_init=initializers.xavier_uniform(),
            )

        if self.correlation > 0:
            self.self_interaction = VisnetSelfInteractionBlock(
                vec_channels=self.vec_channels,
                l_max=self.l_max,
                correlation=self.correlation,
                num_species=self.num_species,
            )
            self.self_interaction_node_proj = nn.Dense(
                features=self.vec_channels,
                kernel_init=initializers.xavier_uniform(),
            )
            self.self_interaction_scalar_proj = nn.Dense(
                features=self.num_channels,
                use_bias=False,
                kernel_init=initializers.xavier_uniform(),
            )

    def _input_shape_assertions(self, graph: Graph) -> None:
        edge_feats = graph.edges.features.get("latent")
        if edge_feats is None:
            edge_feats = graph.edges.features["embedding"]

        node_feats = graph.nodes.features.get("latent_scalars")
        if node_feats is None:
            node_feats = graph.nodes.features["embedding_scalars"]
            vector_feats = graph.nodes.features["embedding_vectors"]
        else:
            vector_feats = graph.nodes.features["latent_vectors"]

        assert node_feats.ndim == 2
        assert node_feats.shape[1] == self.num_channels
        assert edge_feats.ndim == 2
        assert edge_feats.shape[1] == self.num_channels
        assert vector_feats.ndim == 3
        irrep_dim = ((self.l_max + 1) ** 2) - 1
        assert vector_feats.shape[1] == irrep_dim
        assert vector_feats.shape[2] == self.vec_channels
        assert graph.edges.features["distances"].ndim == 1
        assert graph.edges.features["spherical_embedding"].ndim == 2
        assert graph.edges.features["spherical_embedding"].shape[1] == irrep_dim

    def _message_fn(
        self,
        q_i: jax.Array,
        k_j: jax.Array,
        v_j: jax.Array,
        vec_j: jax.Array,
        dk: jax.Array,
        dv: jax.Array,
        distances: jax.Array,
        spherical_feats: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        attn = (q_i * k_j * dk).sum(axis=-1)
        attn = self.attn_act(attn) * jnp.expand_dims(self.cutoff_fn(distances), 1)

        v_j = v_j * dv
        v_j = (v_j * jnp.expand_dims(attn, 2)).reshape(-1, self.num_channels)

        s1, s2 = jnp.split(self.act(self.s_proj(v_j)), [self.vec_channels], axis=1)
        vec_j = vec_j * jnp.expand_dims(s1, 1) + jnp.expand_dims(
            s2, 1
        ) * jnp.expand_dims(spherical_feats, 2)

        return v_j, vec_j

    def _edge_update_w_dot(
        self,
        vec_i_proj: jax.Array,
        vec_j_proj: jax.Array,
        d_ij: jax.Array,
    ) -> jax.Array:
        # w_dot = sum_l w1[l] * w2[l], where w1 = reject(vec_i_proj, d_ij) and
        # w2 = reject(vec_j_proj, -d_ij). Expanding the rejections and using
        # sum_l d_ij[l]^2 = |d_ij|^2, the cross terms collapse to:
        #   w_dot = <a, b> + <a, s> * <b, s> * (|s|^2 - 2)
        # where a = vec_i_proj, b = vec_j_proj, s = d_ij, and <., .> denotes a
        # dot product over the irrep axis. This avoids materializing the
        # [n_edges, irrep_dim, vec_channels] rejection vectors w1 and w2.
        a, b, s = vec_i_proj, vec_j_proj, d_ij
        s_expanded = jnp.expand_dims(s, 2)
        a_dot_b = (a * b).sum(axis=1)
        a_dot_s = (a * s_expanded).sum(axis=1)
        b_dot_s = (b * s_expanded).sum(axis=1)
        s_norm_sq = (s**2).sum(axis=1, keepdims=True)
        return a_dot_b + a_dot_s * b_dot_s * (s_norm_sq - 2.0)

    def __call__(self, graph: Graph) -> Graph:
        """Applies the VisnetLayer module to an input Graph and returns a Graph.

        Runs the forward pass of the VisnetLayer module on an input Graph and returns
        an updated Graph with node and edge features processed using vector-scalar
        attention, normalization, and projection mechanisms.
        Residual connection are applied to the updated features.

        Updated features in this function:
        - scalar node-wise features: graph.nodes.features["latent_scalars"]
        - vector node-wise features: graph.nodes.features["latent_vectors"]
        - edge-wise features: graph.edges.features["latent"]

        Args:
            graph: Input Graph object containing node features ("latent_scalars",
                   "latent_vectors"), edge features ("latent", "distances",
                   "spherical_embedding"), and topology ("senders", "receivers").
                   If this is the first layer in a model, the input features can also
                   be named "embedding_*" instead of "latent_*".

        Returns:
            Updated Graph object with new node and edge features after message
            passing and attention updates.
        """
        self._input_shape_assertions(graph)

        edge_feats = graph.edges.features.get("latent")
        if edge_feats is None:
            edge_feats = graph.edges.features["embedding"]
            graph = graph.update_edge_features(latent=edge_feats)

        node_feats = graph.nodes.features.get("latent_scalars")
        if node_feats is None:
            node_feats = graph.nodes.features["embedding_scalars"]
            vector_feats = graph.nodes.features["embedding_vectors"]
            graph = graph.update_node_features(
                latent_scalars=node_feats, latent_vectors=vector_feats
            )
        else:
            vector_feats = graph.nodes.features["latent_vectors"]

        # Normalization
        node_feats = self.layernorm(node_feats)
        vector_feats = self.vec_layernorm(vector_feats)

        if self.fuse_projections:
            qkv_split_indices = np.cumsum([self.num_channels] * 2)
            q_feats, k_feats, v_feats = jnp.split(
                self.qkv_proj(node_feats), qkv_split_indices, axis=-1
            )
            if self.last_layer:
                dk_feats, dv_feats = jnp.split(
                    self.dkv_proj(edge_feats), [self.num_channels], axis=-1
                )
                f_ij_feats = None
            else:
                dkv_split_indices = np.cumsum([self.num_channels] * 2)
                dk_feats, dv_feats, f_ij_feats = jnp.split(
                    self.dkv_proj(edge_feats), dkv_split_indices, axis=-1
                )
        else:
            q_feats = self.q_proj(node_feats)
            k_feats = self.k_proj(node_feats)
            v_feats = self.v_proj(node_feats)
            dk_feats = self.dk_proj(edge_feats)
            dv_feats = self.dv_proj(edge_feats)
            f_ij_feats = None if self.last_layer else self.f_proj(edge_feats)

        # Reshape the outputs to include the num_heads dimension
        q_feats = jnp.reshape(q_feats, (-1, self.num_heads, self.head_dim))
        k_feats = jnp.reshape(k_feats, (-1, self.num_heads, self.head_dim))
        v_feats = jnp.reshape(v_feats, (-1, self.num_heads, self.head_dim))
        dk_feats = jnp.reshape(self.act(dk_feats), (-1, self.num_heads, self.head_dim))
        dv_feats = jnp.reshape(self.act(dv_feats), (-1, self.num_heads, self.head_dim))

        projected_vec = self.vec_proj(vector_feats)
        vec_split_indices = np.cumsum([self.vec_channels] * 2)
        vec1, vec2, vec3 = jnp.split(projected_vec, vec_split_indices, axis=-1)
        vec_dot = jnp.sum(vec1 * vec2, axis=1)

        # Apply message function for each edge
        q_i = q_feats[graph.receivers, :, :]
        k_j = k_feats[graph.senders, :, :]
        v_j = v_feats[graph.senders, :, :]
        vec_j = vector_feats[graph.senders, :, :]

        node_msgs, vec_msgs = self._message_fn(
            q_i,
            k_j,
            v_j,
            vec_j,
            dk_feats,
            dv_feats,
            graph.edges.features["distances"],
            graph.edges.features["spherical_embedding"],
        )
        # Aggregate the messages
        node_feats = segment_sum(
            node_msgs,
            graph.receivers,
            num_segments=node_feats.shape[0],
            deterministic=self.deterministic_scatter_ops,
        )
        vec_out = segment_sum(
            vec_msgs,
            graph.receivers,
            num_segments=node_feats.shape[0],
            deterministic=self.deterministic_scatter_ops,
        )

        if self.correlation > 0:
            si_scalars, si_vectors = self.self_interaction(
                self.self_interaction_node_proj(node_feats),
                vec_out,
                graph.nodes.features["species"],
            )
            node_feats = node_feats + self.self_interaction_scalar_proj(si_scalars)
            vec_out = vec_out + si_vectors

        o_split_indices = np.cumsum([self.vec_channels, self.num_channels])
        o1, o2, o3 = jnp.split(self.o_proj(node_feats), o_split_indices, axis=1)

        dx = self.vec_dot_proj(vec_dot) * o2 + o3
        dvec = vec3 * jnp.expand_dims(o1, 1) + vec_out

        if not self.last_layer:
            # w_trg_proj/w_src_proj are linear maps over the channel axis, which
            # commutes with gathering node features onto edges. Apply them once
            # per node (and gather afterwards) instead of once per edge, since
            # num_edges is typically much larger than num_nodes.
            if self.fuse_projections:
                vector_feats_trg, vector_feats_src = jnp.split(
                    self.w_src_trg_proj(vector_feats), [self.vec_channels], axis=-1
                )
            else:
                vector_feats_trg = self.w_trg_proj(vector_feats)
                vector_feats_src = self.w_src_proj(vector_feats)

            w_dot = self._edge_update_w_dot(
                vector_feats_trg[graph.receivers, :],
                vector_feats_src[graph.senders, :],
                graph.edges.features["spherical_embedding"],
            )
            df_ij = self.act(f_ij_feats) * self.w_dot_proj(w_dot)
        else:
            df_ij = jnp.zeros_like(edge_feats)

        diff_node_feats = dx
        diff_edge_feats = df_ij
        diff_vector_feats = dvec

        out_node_feats = graph.nodes.features["latent_scalars"] + diff_node_feats
        out_edge_feats = graph.edges.features["latent"] + diff_edge_feats
        out_vector_feats = graph.nodes.features["latent_vectors"] + diff_vector_feats

        graph = graph.update_node_features(
            latent_scalars=out_node_feats, latent_vectors=out_vector_feats
        )
        graph = graph.update_edge_features(latent=out_edge_feats)

        return graph
