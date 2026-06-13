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

"""Benchmark the pretrained ViSNet model on a realistic MD workload.

Builds a ~1000-atom system by tiling the real aspirin geometry from
`tests/sample_data/small_aspirin_test.xyz`, and runs a 1000-step NVT-Langevin
MD simulation with `JaxMDSimulationEngine`, using the pretrained
`visnet_organics_02` checkpoint.

Reports total wall-clock time for the run and peak GPU memory.

Usage::

    python scripts/benchmark_visnet_md_pretrained.py --n-replicas 48 --num-steps 1000
"""

import argparse
import time
from pathlib import Path

import ase.io
import jax
from huggingface_hub import hf_hub_download

from mlip.models.model_io import load_model_from_zip
from mlip.models.visnet.network import Visnet
from mlip.simulation.enums import MDIntegrator, SimulationType
from mlip.simulation.jax_md.jax_md_simulation_engine import JaxMDSimulationEngine
from mlip.typing.properties import Properties

REPO_ID = "InstaDeepAI/mlip_models_organics_v2"
CHECKPOINT_FILENAME = "visnet_organics_02.zip"
ASPIRIN_XYZ = (
    Path(__file__).resolve().parents[1] / "tests" / "sample_data" / "small_aspirin_test.xyz"
)


def build_atoms(n_replicas: int) -> ase.Atoms:
    """Builds an `ase.Atoms` of `n_replicas` non-interacting aspirin molecules."""
    molecule = ase.io.read(ASPIRIN_XYZ, index=0)
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

    atoms.set_cell(None)
    atoms.set_pbc(False)
    atoms.info["charge"] = 0
    return atoms


def memory_stats() -> tuple[float, float]:
    """Returns (peak_bytes_in_use, bytes_in_use) for the default device, in MB."""
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return float("nan"), float("nan")
    return stats["peak_bytes_in_use"] / 1e6, stats["bytes_in_use"] / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-replicas", type=int, default=48)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--label", type=str, default="visnet")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILENAME)

    required_properties = Properties(stress=False, partial_charges=False)
    force_field = load_model_from_zip(
        Visnet, checkpoint_path, required_properties=required_properties
    )

    atoms = build_atoms(args.n_replicas)
    print(f"n_atoms={len(atoms)}")

    config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=args.num_steps,
        snapshot_interval=args.num_steps,
        num_episodes=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=None,
        edge_capacity_multiplier=1.25,
    )

    engine = JaxMDSimulationEngine(atoms, force_field, config)

    start = time.perf_counter()
    engine.run()
    jax.block_until_ready(engine.state.positions)
    elapsed = time.perf_counter() - start

    peak_mb, in_use_mb = memory_stats()

    print(
        f"{args.label:<25s} n_atoms={len(atoms):5d}  num_steps={args.num_steps:5d}  "
        f"total={elapsed:8.3f} s  peak={peak_mb:9.1f} MB   in_use={in_use_mb:9.1f} MB"
    )


if __name__ == "__main__":
    main()
