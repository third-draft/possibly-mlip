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

"""Profiling harness for ``ESENLayer``.

Produces two things:

1. A wall-clock breakdown of a handful of standalone JAX ops that are
   suspected hotspots (segment_sum variants, wigner-rotation einsum, gather),
   at realistic edge/node counts.
2. A full ``jax.profiler`` trace of the forward+backward pass of a stack of
   ``ESENLayer``s, written to ``--trace-dir`` for inspection in TensorBoard
   (``tensorboard --logdir <trace-dir>``).

Usage::

    python scripts/profile_esen_layer.py --n-nodes 2048 --n-edges 65536 \\
        --sphere-channels 128 --hidden-channels 128 --edge-channels 128 \\
        --num-layers 2 --trace-dir /tmp/esen_trace
"""

import argparse
import time
from typing import Callable

import jax
import jax.numpy as jnp

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.esen.blocks import EsenEmbeddingBlock
from mlip.models.esen.coefficient_mapping import CoefficientMapping
from mlip.models.esen.layer import ESENLayer
from mlip.utils.jax_utils import _deterministic_segment_sum


def _bench(name: str, fn: Callable[[], jax.Array], n_warmup: int = 3, n_iters: int = 10) -> None:
    for _ in range(n_warmup):
        jax.block_until_ready(fn())

    start = time.perf_counter()
    for _ in range(n_iters):
        jax.block_until_ready(fn())
    elapsed = (time.perf_counter() - start) / n_iters

    print(f"{name:<45s} {elapsed * 1e3:10.4f} ms")


def build_graph(
    key: jax.Array,
    n_nodes: int,
    n_edges: int,
    sphere_channels: int,
    l_max: int,
    m_max: int,
    embedding_block: EsenEmbeddingBlock,
) -> Graph:
    k1, k2, k3 = jax.random.split(key, 3)

    sph_size = (l_max + 1) ** 2
    node_feats = jax.random.normal(k1, (n_nodes, sph_size, sphere_channels))
    edge_vectors = jax.random.normal(k2, (n_edges, 3))
    senders = jax.random.randint(k3, (n_edges,), 0, n_nodes)
    offsets = jax.random.randint(k3, (n_edges,), 1, n_nodes)
    receivers = jnp.mod(senders + offsets, n_nodes)

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
        latent=jnp.ones((n_edges, sphere_channels)),
        vectors=edge_vectors,
        envelope=jnp.ones((n_edges, 1, 1)),
        wigner_and_m_mapping=wigner_and_m_mapping,
        wigner_and_m_mapping_inv=wigner_and_m_mapping_inv,
    )
    return graph


def microbenchmarks(n_nodes: int, n_edges: int, res_size: int, channels: int, key: jax.Array) -> None:
    print("\n=== Standalone op microbenchmarks ===")
    k1, k2, k3 = jax.random.split(key, 3)

    edge_messages = jax.random.normal(k1, (n_edges, res_size, channels))
    dst = jax.random.randint(k2, (n_edges,), 0, n_nodes)
    wigner = jax.random.normal(k3, (n_edges, res_size, res_size))
    node_feats = jax.random.normal(k1, (n_edges, res_size, 2 * channels))

    segment_sum_jit = jax.jit(
        lambda d, s: jax.ops.segment_sum(d, s, num_segments=n_nodes)
    )
    det_segment_sum_jit = jax.jit(
        lambda d, s: _deterministic_segment_sum(d, s, num_segments=n_nodes)
    )
    rotate_jit = jax.jit(lambda w, x: jnp.einsum("eij,ejc->eic", w, x))
    gather_jit = jax.jit(lambda x, idx: x[idx])

    node_feats_full = jax.random.normal(k1, (n_nodes, res_size, channels))

    _bench("segment_sum (non-deterministic)", lambda: segment_sum_jit(edge_messages, dst))
    _bench("segment_sum (deterministic, one-hot matmul)", lambda: det_segment_sum_jit(edge_messages, dst))
    _bench("wigner rotation einsum (eij,ejc->eic)", lambda: rotate_jit(wigner, node_feats))
    _bench("gather node_feats[senders]", lambda: gather_jit(node_feats_full, dst))


def full_layer_trace(
    key: jax.Array,
    n_nodes: int,
    n_edges: int,
    sphere_channels: int,
    hidden_channels: int,
    edge_channels: int,
    l_max: int,
    m_max: int,
    num_rbf: int,
    num_layers: int,
    trace_dir: str | None,
) -> None:
    print("\n=== Full ESENLayer stack: forward + backward ===")
    mapping_reduced = CoefficientMapping(l_max=l_max, m_max=m_max)
    edge_channels_list = [num_rbf + 2 * edge_channels, edge_channels, edge_channels]

    embedding_block = EsenEmbeddingBlock(
        graph_cutoff_angstrom=5.0,
        l_max=l_max,
        num_species=10,
        num_charges=None,
        sphere_channels=sphere_channels,
        radial_envelope="polynomial",
        radial_basis="gauss",
        num_rbf=num_rbf,
        basis_width_scalar=2.0,
        cosine_cutoff=False,
        trainable_rbf=False,
        edge_channels=edge_channels,
        m_max=m_max,
        mapping_reduced=mapping_reduced,
        edge_channels_list=edge_channels_list,
    )

    graph = build_graph(
        key, n_nodes, n_edges, sphere_channels, l_max, m_max, embedding_block
    )

    layers = [
        ESENLayer(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            l_max=l_max,
            m_max=m_max,
            mapping_reduced=mapping_reduced,
            edge_channels_list=edge_channels_list,
            graph_cutoff_angstrom=5.0,
            norm_type="rms_norm_sh",
            act_type="gate",
        )
        for _ in range(num_layers)
    ]

    params = [layer.init(key, graph) for layer in layers]

    def forward(params_list, graph):
        for layer, p in zip(layers, params_list):
            graph = layer.apply(p, graph)
        return jnp.sum(graph.nodes.features["latent"] ** 2)

    forward_jit = jax.jit(forward)
    grad_jit = jax.jit(jax.grad(forward))

    _bench("ESENLayer stack forward", lambda: forward_jit(params, graph))
    _bench("ESENLayer stack forward+backward (grad)", lambda: grad_jit(params, graph))

    if trace_dir is not None:
        print(f"\nWriting profiler trace to {trace_dir} ...")
        jax.block_until_ready(grad_jit(params, graph))  # warmup / compile
        with jax.profiler.trace(trace_dir):
            for _ in range(5):
                jax.block_until_ready(grad_jit(params, graph))
        print(f"Trace written. Inspect with: tensorboard --logdir {trace_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=2048)
    parser.add_argument("--n-edges", type=int, default=65536)
    parser.add_argument("--sphere-channels", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--edge-channels", type=int, default=128)
    parser.add_argument("--num-rbf", type=int, default=32)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--m-max", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--trace-dir", type=str, default=None)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    key = jax.random.PRNGKey(0)
    res_size = (args.l_max + 1) ** 2

    microbenchmarks(
        n_nodes=args.n_nodes,
        n_edges=args.n_edges,
        res_size=res_size,
        channels=args.hidden_channels,
        key=key,
    )

    full_layer_trace(
        key=key,
        n_nodes=args.n_nodes,
        n_edges=args.n_edges,
        sphere_channels=args.sphere_channels,
        hidden_channels=args.hidden_channels,
        edge_channels=args.edge_channels,
        l_max=args.l_max,
        m_max=args.m_max,
        num_rbf=args.num_rbf,
        num_layers=args.num_layers,
        trace_dir=args.trace_dir,
    )


if __name__ == "__main__":
    main()
