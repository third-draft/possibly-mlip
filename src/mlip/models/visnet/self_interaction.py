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


import flax.linen as nn
import jax.numpy as jnp
from e3j.arrays.array import O3Array
from e3j.utils.irreps import irrep_range
from e3j.utils.options import Layout

from mlip.models.mace.symmetric_contraction import SymmetricContraction


class VisnetSelfInteractionBlock(nn.Module):
    """Equivariant many-body self-interaction for ViSNet's vector features.

    Applies a MACE-style symmetric contraction (power expansion of the input
    features up to a configurable correlation order, contracted with
    species-dependent weights, see
    :class:`~mlip.models.mace.symmetric_contraction.SymmetricContraction`) to
    the combined scalar (l=0) and equivariant vector (l=1..l_max) features of
    a node. This produces a new rotation-invariant (l=0) contribution that can
    be merged into the scalar features, and new equivariant (l=1..l_max)
    contributions with the same shape and irreps as the input vector
    features, that can be merged back into the vector features.

    Attributes:
        vec_channels: Channel width of the scalar and vector features (both
            must share this width, see ``__call__``).
        l_max: Highest spherical-harmonic degree present in the vector
            features.
        correlation: Correlation order (the ``nu`` in MACE's symmetric
            contraction ``B = W @ (A + A^2 + ... + A^nu)``).
        num_species: Number of atomic species, used for the species-dependent
            mixing weights.
    """

    vec_channels: int
    l_max: int
    correlation: int
    num_species: int

    def setup(self) -> None:
        """Initializes the underlying symmetric contraction."""
        irreps = irrep_range(self.l_max)
        self.irreps = irreps
        self.symmetric_contraction = SymmetricContraction(
            source_irreps=str(irreps),
            correlation=self.correlation,
            keep_irrep_out=str(irreps),
            num_species=self.num_species,
            num_channels=self.vec_channels,
            layout=Layout.TRAILING_CHANNELS,
            l_max=self.l_max,
        )

    def __call__(
        self,
        scalar_feats: jnp.ndarray,
        vector_feats: jnp.ndarray,
        node_species: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Applies the equivariant self-interaction to per-node features.

        Args:
            scalar_feats: Rotation-invariant node features of shape
                ``[n_nodes, vec_channels]`` (l=0), e.g. the aggregated scalar
                messages in :class:`~mlip.models.visnet.layer.VisnetLayer`.
            vector_feats: Equivariant node features of shape
                ``[n_nodes, irrep_dim, vec_channels]``, spanning l=1..l_max
                with no l=0 component, e.g. ``vec_out`` in
                :class:`~mlip.models.visnet.layer.VisnetLayer`.
            node_species: Integer species indices of shape ``[n_nodes]``.

        Returns:
            Tuple ``(scalars, vectors)`` where ``scalars`` has shape
            ``[n_nodes, vec_channels]`` (rotation-invariant, l=0) and
            ``vectors`` has shape ``[n_nodes, irrep_dim, vec_channels]``
            (l=1..l_max, same irreps as ``vector_feats``).
        """
        x = jnp.concatenate([scalar_feats[:, None, :], vector_feats], axis=1)
        x = jnp.swapaxes(x, -1, -2)
        node_feats = O3Array(self.irreps, x, layout=Layout.E3NN)
        out = self.symmetric_contraction(node_feats, node_species)
        out = jnp.swapaxes(out.array, -1, -2)
        scalars = out[:, 0, :]
        vectors = out[:, 1:, :]
        return scalars, vectors
