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

"""Factories for the convolution layers / tensor products under comparison.

Each factory returns a `BenchTarget`: a `(name, init_fn, apply_fn, inputs)` bundle
that `bench.benchmark_target` can time uniformly, whether the underlying op is a
`flax.linen.Module` (with real parameters) or a weightless bilinear tensor product
(with an empty parameter dict).
"""

from dataclasses import dataclass
from typing import Any, Callable

import e3j
import e3nn_jax as e3nn
import jax
from e3j.utils.options import Layout

from mlip.graph import Graph
from mlip.models.esen.coefficient_mapping import CoefficientMapping
from mlip.models.esen.esen_helpers import SO2Convolution
from mlip.models.esen.layer import Edgewise
from mlip.models.gaunt_tensor_product import GauntMessagePassingBlock, GauntTensorProduct
from mlip.models.message_passing import O3MessagePassingBlock

from .esen_single_conv import EdgewiseSingleConv
from .graphs import BenchmarkShapes


@dataclass
class BenchTarget:
    """A named, uniformly-callable benchmark target."""

    name: str
    init_fn: Callable[..., Any]
    apply_fn: Callable[..., Any]
    inputs: tuple

    def init(self, key: jax.Array):
        return self.init_fn(key, *self.inputs)

    def apply(self, params):
        return self.apply_fn(params, *self.inputs)


def _no_params_init(key: jax.Array, *inputs):
    del key, inputs
    return {}


# --------------------------------------------------------------------------- #
# Full message-passing convolutions (gather -> TP -> scatter, with surrounding
# linears/MLPs).
# --------------------------------------------------------------------------- #


def mace_e3j_full(
    shapes: BenchmarkShapes,
    graph: Graph,
    avg_num_neighbors: float,
    radial_mlp_hidden: list[int] | None = None,
) -> BenchTarget:
    """Full MACE interaction block using the e3j (Clebsch-Gordan) tensor product."""
    module = O3MessagePassingBlock(
        source_irreps=shapes.num_channels * shapes.source_irreps,
        l_max=shapes.l_max,
        target_irreps=shapes.num_channels * shapes.interaction_irreps,
        num_rbf=shapes.num_rbf,
        radial_mlp_hidden=radial_mlp_hidden or [64, 64, 64],
        radial_mlp_activation="silu",
        avg_num_neighbors=avg_num_neighbors,
    )
    return BenchTarget("mace_e3j_full", module.init, module.apply, (graph,))


def mace_gaunt_full(
    shapes: BenchmarkShapes,
    graph: Graph,
    avg_num_neighbors: float,
    radial_mlp_hidden: list[int] | None = None,
) -> BenchTarget:
    """Full MACE interaction block using the Gaunt tensor product."""
    module = GauntMessagePassingBlock(
        source_irreps=shapes.num_channels * shapes.source_irreps,
        l_max=shapes.l_max,
        target_irreps=shapes.num_channels * shapes.interaction_irreps,
        num_rbf=shapes.num_rbf,
        radial_mlp_hidden=radial_mlp_hidden or [64, 64, 64],
        radial_mlp_activation="silu",
        avg_num_neighbors=avg_num_neighbors,
    )
    return BenchTarget("mace_gaunt_full", module.init, module.apply, (graph,))


def esen_full(shapes: BenchmarkShapes, graph: Graph) -> BenchTarget:
    """Full eSEN edge-wise convolution (`Edgewise`: two `SO2Convolution`s + gate)."""
    mapping_reduced = CoefficientMapping(l_max=shapes.l_max, m_max=shapes.m_max)
    module = Edgewise(
        sphere_channels=shapes.num_channels,
        hidden_channels=shapes.num_channels,
        l_max=shapes.l_max,
        m_max=shapes.m_max,
        edge_channels_list=shapes.edge_channels_list,
        mapping_reduced=mapping_reduced,
        graph_cutoff_angstrom=5.0,
        act_type="gate",
    )
    return BenchTarget("esen_full", module.init, module.apply, (graph,))


def esen_single_conv_full(shapes: BenchmarkShapes, graph: Graph) -> BenchTarget:
    """Single-`SO2Convolution` eSEN edge-wise convolution.

    Matches the workload shape of `mace_e3j_full`/`mace_gaunt_full`: gather -> rotate
    -> single tensor product/convolution -> rotate back -> scatter, with no gate
    activation or second convolution.
    """
    mapping_reduced = CoefficientMapping(l_max=shapes.l_max, m_max=shapes.m_max)
    module = EdgewiseSingleConv(
        sphere_channels=shapes.num_channels,
        l_max=shapes.l_max,
        m_max=shapes.m_max,
        edge_channels_list=shapes.edge_channels_list,
        mapping_reduced=mapping_reduced,
    )
    return BenchTarget("esen_single_conv_full", module.init, module.apply, (graph,))


# --------------------------------------------------------------------------- #
# Tensor-product-only benchmarks: per-edge ops, no gather/scatter.
# --------------------------------------------------------------------------- #


def mace_e3j_tp_only(shapes: BenchmarkShapes, n_edges: int, key: jax.Array) -> BenchTarget:
    """The raw Clebsch-Gordan tensor product used inside `SO3Convolution`.

    Weightless: combines per-edge sender features (`x_feats`) with spherical
    harmonics of the edge vector (`y_lm`) to form messages, in
    `Layout.TRAILING_CHANNELS` (matching `O3MessagePassingBlock`'s default layout).
    """
    source_irreps = shapes.source_irreps
    harmonics_irreps = e3nn.Irreps.spherical_harmonics(shapes.l_max)
    target_irreps = shapes.interaction_irreps

    tensor_product = e3j.core.TensorProduct(
        source=(str(source_irreps), str(harmonics_irreps)),
        target=target_irreps,
        layout=Layout.TRAILING_CHANNELS,
        normalization="SQRT_DIM_OUT",
    )

    k1, k2 = jax.random.split(key)
    x_feats = jax.random.normal(k1, (n_edges, source_irreps.dim, shapes.num_channels))
    y_lm = jax.random.normal(k2, (n_edges, harmonics_irreps.dim))

    def apply_fn(params, x, y):
        del params
        return tensor_product(x, y)

    return BenchTarget("mace_e3j_tp_only", _no_params_init, apply_fn, (x_feats, y_lm))


def mace_gaunt_tp_only(shapes: BenchmarkShapes, n_edges: int, key: jax.Array) -> BenchTarget:
    """The raw Gaunt tensor product used inside `GauntMessagePassingBlock`.

    Weightless: combines per-edge, per-channel sender features with per-channel
    spherical harmonics of the edge vector.
    """
    source_irreps = shapes.source_irreps
    harmonics_irreps = e3nn.Irreps.spherical_harmonics(shapes.l_max)
    target_irreps = e3nn.tensor_product(
        source_irreps, harmonics_irreps, filter_ir_out=shapes.interaction_irreps
    ).set_mul(1)

    gtp = GauntTensorProduct(sources=(source_irreps, harmonics_irreps), target=target_irreps)

    k1, k2 = jax.random.split(key)
    x1 = e3nn.IrrepsArray(
        source_irreps,
        jax.random.normal(k1, (n_edges, shapes.num_channels, source_irreps.dim)),
    )
    x2 = e3nn.IrrepsArray(
        harmonics_irreps,
        jax.random.normal(k2, (n_edges, shapes.num_channels, harmonics_irreps.dim)),
    )

    def apply_fn(params, a, b):
        del params
        return gtp(a, b)

    return BenchTarget("mace_gaunt_tp_only", _no_params_init, apply_fn, (x1, x2))


def esen_so2_tp_only(shapes: BenchmarkShapes, n_edges: int, key: jax.Array) -> BenchTarget:
    """The `SO2Convolution` used inside eSEN's `Edgewise`.

    Unlike the (weightless) Clebsch-Gordan/Gaunt tensor products above, eSEN's
    SO(2)-reduced convolution has learned weights baked into the per-`m` mixing -- it
    is eSEN's analog of "the tensor product step", but is not a separate bilinear
    operator. Uses `internal_weights=True` (as in `Edgewise.so2_conv_2`) so it can be
    called directly on edge-batched features with no auxiliary radial MLP.
    """
    mapping_reduced = CoefficientMapping(l_max=shapes.l_max, m_max=shapes.m_max)
    module = SO2Convolution(
        sphere_channels=shapes.num_channels,
        m_output_channels=shapes.num_channels,
        l_max=shapes.l_max,
        m_max=shapes.m_max,
        mapping_reduced=mapping_reduced,
        internal_weights=True,
        edge_channels_list=None,
        extra_m0_output_channels=None,
    )

    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (n_edges, shapes.sph_size, shapes.num_channels))
    x_edge = jax.random.normal(k2, (n_edges, shapes.sph_size, shapes.num_channels))

    return BenchTarget("esen_so2_tp_only", module.init, module.apply, (x, x_edge))
