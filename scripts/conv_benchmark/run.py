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

"""Benchmark MACE (e3j / Gaunt-TP) vs eSEN convolution layers.

Compares the tensor-product step alone, and the full message-passing convolution
(gather -> TP -> scatter, with surrounding linears/MLPs), for dummy graphs of
configurable size and connectivity. Parameters are randomly initialized for each
layer independently -- there is no attempt at numerical equivalence between
architectures, only equivalent problem sizes (`n_nodes`, `n_edges`, channel count,
`l_max`).

Usage::

    python -m scripts.conv_benchmark.run --n-nodes 256 --n-edges 4096 \\
        --num-channels 32 --l-max 2
"""

import argparse

import jax

from . import bench, layers
from .connectivity import make_connectivity
from .graphs import build_esen_graph, build_mace_graph, default_shapes

ALL_MODELS = ("mace_e3j", "mace_gaunt", "esen", "esen_single_conv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", type=int, default=256)
    parser.add_argument("--n-edges", type=int, default=4096)
    parser.add_argument("--connectivity", choices=["random", "k_regular"], default="random")
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--m-max", type=int, default=None)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--edge-channels", type=int, default=None)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--mode", choices=["tp", "full", "both"], default="both")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    shapes = default_shapes(
        num_channels=args.num_channels,
        l_max=args.l_max,
        m_max=args.m_max,
        num_rbf=args.num_rbf,
        edge_channels=args.edge_channels,
    )
    print(
        f"n_nodes={args.n_nodes} n_edges={args.n_edges} connectivity={args.connectivity} "
        f"num_channels={shapes.num_channels} l_max={shapes.l_max} m_max={shapes.m_max} "
        f"num_rbf={shapes.num_rbf} edge_channels={shapes.edge_channels}"
    )
    avg_num_neighbors = args.n_edges / args.n_nodes

    key = jax.random.PRNGKey(args.seed)
    key_conn, key_mace_graph, key_esen_graph, key_tp = jax.random.split(key, 4)

    senders, receivers = make_connectivity(args.n_nodes, args.n_edges, args.connectivity, key_conn)

    with_backward = not args.no_backward

    rows = []
    for model in args.models:
        if args.mode in ("tp", "both"):
            tp_key, key_tp = jax.random.split(key_tp)
            if model == "mace_e3j":
                target = layers.mace_e3j_tp_only(shapes, args.n_edges, tp_key)
            elif model == "mace_gaunt":
                target = layers.mace_gaunt_tp_only(shapes, args.n_edges, tp_key)
            elif model == "esen":
                target = layers.esen_so2_tp_only(shapes, args.n_edges, tp_key)
            else:
                # esen_single_conv shares its TP-only target with "esen".
                target = None
            if target is not None:
                rows.append(
                    bench.benchmark_target(target, tp_key, with_backward, args.n_warmup, args.n_iters)
                )

        if args.mode in ("full", "both"):
            if model == "mace_e3j":
                graph = build_mace_graph(
                    key_mace_graph, args.n_nodes, args.n_edges, senders, receivers, shapes
                )
                target = layers.mace_e3j_full(shapes, graph, avg_num_neighbors)
            elif model == "mace_gaunt":
                graph = build_mace_graph(
                    key_mace_graph, args.n_nodes, args.n_edges, senders, receivers, shapes
                )
                target = layers.mace_gaunt_full(shapes, graph, avg_num_neighbors)
            elif model == "esen":
                graph = build_esen_graph(
                    key_esen_graph, args.n_nodes, args.n_edges, senders, receivers, shapes
                )
                target = layers.esen_full(shapes, graph)
            else:
                graph = build_esen_graph(
                    key_esen_graph, args.n_nodes, args.n_edges, senders, receivers, shapes
                )
                target = layers.esen_single_conv_full(shapes, graph)
            rows.append(
                bench.benchmark_target(target, key, with_backward, args.n_warmup, args.n_iters)
            )

    bench.print_table(rows)


if __name__ == "__main__":
    main()
