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

"""Benchmark the pretrained ViSNet model for speed and memory.

Downloads the "visnet_organics_02" checkpoint from the
`InstaDeepAI/mlip_models_organics_v2` HuggingFace repo, builds a synthetic system by
tiling a small organic molecule, and benchmarks `ForceField` inference (energy +
forces, i.e. forward + backward through the network).

Reports wall-clock time per call and peak GPU memory.

Usage::

    python scripts/benchmark_visnet_pretrained.py --n-replicas 256
"""

import argparse
import time

import ase.build
import jax
from huggingface_hub import hf_hub_download

from mlip.data import ChemicalSystem
from mlip.graph import Graph
from mlip.models.force_field import ForceField
from mlip.models.model_io import load_model_from_zip
from mlip.models.visnet.network import Visnet
from mlip.typing.properties import Properties

REPO_ID = "InstaDeepAI/mlip_models_organics_v2"
CHECKPOINT_FILENAME = "visnet_organics_02.zip"


def build_graph(n_replicas: int, cutoff: float) -> Graph:
    """Builds a Graph from `n_replicas` non-interacting benzene molecules."""
    molecule = ase.build.molecule("C6H6")
    molecule.center(vacuum=10.0)
    cell = molecule.get_cell()

    side = 1
    while side**3 < n_replicas:
        side += 1

    atoms = molecule.copy()
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                if len(atoms) // len(molecule) >= n_replicas:
                    break
                shifted = molecule.copy()
                shifted.translate(cell[0] * ix + cell[1] * iy + cell[2] * iz)
                atoms += shifted

    atoms.info["charge"] = 0
    chemical_system = ChemicalSystem.from_ase_atoms(atoms, get_property_fields=False)
    return Graph.from_chemical_system(chemical_system, cutoff)


def memory_stats() -> tuple[float, float]:
    """Returns (peak_bytes_in_use, bytes_in_use) for the default device, in MB."""
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return float("nan"), float("nan")
    return stats["peak_bytes_in_use"] / 1e6, stats["bytes_in_use"] / 1e6


def bench(
    name: str,
    fn,
    graph: Graph,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> None:
    for _ in range(n_warmup):
        jax.block_until_ready(fn(graph))

    peak_mb, in_use_mb = memory_stats()

    start = time.perf_counter()
    for _ in range(n_iters):
        jax.block_until_ready(fn(graph))
    elapsed = (time.perf_counter() - start) / n_iters

    print(
        f"{name:<25s} {elapsed * 1e3:10.4f} ms   "
        f"peak={peak_mb:9.1f} MB   in_use={in_use_mb:9.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-replicas", type=int, default=256)
    parser.add_argument("--label", type=str, default="visnet")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILENAME)

    required_properties = Properties(stress=False, partial_charges=False)
    force_field = load_model_from_zip(
        Visnet, checkpoint_path, required_properties=required_properties
    )
    dataset_info = force_field.predictor.mlip_network.dataset_info

    graph = build_graph(args.n_replicas, dataset_info.graph_cutoff_angstrom)
    print(f"n_nodes={graph.n_node}, n_edges={graph.n_edge}")

    predict_jit = jax.jit(force_field)

    def fn(g):
        result = predict_jit(g)
        return result.energy, result.forces

    bench(args.label, fn, graph)


if __name__ == "__main__":
    main()
