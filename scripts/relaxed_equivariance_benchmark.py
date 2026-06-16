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

"""Benchmarks ViSNet with relaxed equivariance against Config A.

Config A: num_channels=128, num_layers=4, l_max=2, MSELoss(e=1, f=1, s=0).

Relaxed equivariance config: same hyperparams + relaxed_equivariance=True,
same MSELoss.  A per-layer irrep-mixing block (64 params/layer, zero-init)
is applied to the vector update dvec with weight beta injected via
graph.globals.features["relaxed_equiv_weight"]:

  epochs  0-14 : beta = 0.05  (constant)
  epochs 15-19 : beta linearly from 0.05 -> 0.0  (epoch 19 = fully equivariant)

Usage::

    python scripts/relaxed_equivariance_benchmark.py
"""

import json
import time
from pathlib import Path
from typing import Callable

import jax
import optax

from mlip.data.chemical_systems_readers.hdf5_reader import Hdf5Reader
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.graph_dataset_builder import BuilderMode, GraphDatasetBuilder
from mlip.models import ForceField, Visnet
from mlip.models.loss import MSELoss
from mlip.models.visnet.config import VisnetConfig
from mlip.training import get_default_mlip_optimizer
from mlip.training.optimizer_config import OptimizerConfig
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.training.training_loop import TrainingLoop

DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "SPICE2_curated_v2_5pct"
)
TRAIN_FILE = DATA_DIR / "spice2_with_smiles_train.hdf5"
VAL_FILE = DATA_DIR / "spice2_with_smiles_val.hdf5"

NUM_EPOCHS = 20
BATCH_SIZE = 16
GRAPH_CUTOFF_ANGSTROM = 5.0
LEARNING_RATE = 1e-3

CONFIG_A_KWARGS = dict(num_channels=128, num_layers=4, l_max=2)
RELAXED_EQUIV_KWARGS = dict(num_channels=128, num_layers=4, l_max=2, relaxed_equivariance=True)

RESULTS_FILE = Path(__file__).resolve().parent / "relaxed_equivariance_results.json"


class DivergedError(Exception):
    pass


def build_datasets():
    print("\n=== Building train/val graph datasets ===")
    readers = {
        "train": Hdf5Reader(filepaths=TRAIN_FILE),
        "val": Hdf5Reader(filepaths=VAL_FILE),
    }
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=GRAPH_CUTOFF_ANGSTROM,
        batch_size=BATCH_SIZE,
    )
    start = time.perf_counter()
    builder = GraphDatasetBuilder(readers, builder_config, mode=BuilderMode.TRAINING)
    datasets = builder.get_datasets(prefetch=False)
    print(f"Dataset build time: {time.perf_counter() - start:.1f}s")
    print(
        f"Train batches: {len(datasets['train'])}, Val batches: {len(datasets['val'])}"
    )
    return datasets["train"], datasets["val"], builder.dataset_info


def run_config(
    name: str,
    model_kwargs: dict,
    loss,
    train_set,
    val_set,
    dataset_info,
    relaxed_equiv_schedule: Callable[[int], float] | None = None,
) -> dict:
    print(f"\n=== {name}: {model_kwargs}, lr={LEARNING_RATE} ===")
    model_config = VisnetConfig(**model_kwargs)
    model = Visnet(model_config, dataset_info)
    force_field = ForceField.from_mlip_network(model)
    n_params = sum(x.size for x in jax.tree.leaves(force_field.params))
    print(f"Parameter count: {n_params:,}")

    opt_config = OptimizerConfig(
        init_learning_rate=LEARNING_RATE,
        peak_learning_rate=LEARNING_RATE,
        final_learning_rate=LEARNING_RATE,
    )
    optimizer = get_default_mlip_optimizer(opt_config)

    training_config = TrainingLoop.Config(
        num_epochs=NUM_EPOCHS,
        run_eval_at_start=True,
    )

    history = {"train_loss": [], "eval": [], "epoch_time_seconds": []}

    def _logger(log_category, to_log, epoch_num):
        if log_category == LogCategory.TRAIN_METRICS:
            loss_val = float(to_log["loss"])
            history["train_loss"].append((epoch_num, loss_val))
            print(f"  epoch {epoch_num} train_loss={loss_val}")
            if loss_val != loss_val:  # NaN check
                raise DivergedError(f"NaN train loss at epoch {epoch_num}")
        elif log_category == LogCategory.EVAL_METRICS:
            history["eval"].append((epoch_num, {k: float(v) for k, v in to_log.items()}))
            print(f"  epoch {epoch_num} eval={to_log}")
        elif log_category == LogCategory.SYSTEM_METRICS:
            history["epoch_time_seconds"].append(float(to_log["runtime_in_seconds"]))

    io_handler = TrainingIOHandler(TrainingIOHandler.Config(checkpoint_dir=None))
    io_handler.attach_logger(_logger)

    training_loop = TrainingLoop(
        train_dataset=train_set,
        validation_dataset=val_set,
        force_field=force_field,
        loss=loss,
        optimizer=optimizer,
        config=training_config,
        io_handler=io_handler,
        relaxed_equiv_schedule=relaxed_equiv_schedule,
    )

    start = time.perf_counter()
    try:
        training_loop.run()
        diverged = False
    except DivergedError as e:
        print(f"  DIVERGED: {e}")
        diverged = True
    total_time = time.perf_counter() - start
    print(f"  total time: {total_time:.1f}s")

    return {
        "model_kwargs": model_kwargs,
        "n_params": n_params,
        "learning_rate": LEARNING_RATE,
        "diverged": diverged,
        "total_time_seconds": total_time,
        "history": history,
        "final_eval": history["eval"][-1][1] if history["eval"] else None,
    }


def main() -> None:
    print(f"JAX devices: {jax.devices()}")
    train_set, val_set, dataset_info = build_datasets()

    base_loss = MSELoss(lambda x: 1.0, lambda x: 1.0, lambda x: 0)

    # beta = 0.05 for epochs 0-14, then linearly -> 0.0 by epoch 19
    beta_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(0.05),
            optax.linear_schedule(init_value=0.05, end_value=0.0, transition_steps=4),
        ],
        boundaries=[15],
    )

    results = {}
    results["config_a"] = run_config(
        "Config A", CONFIG_A_KWARGS, base_loss, train_set, val_set, dataset_info
    )
    results["relaxed_equivariance"] = run_config(
        "Relaxed equivariance",
        RELAXED_EQUIV_KWARGS,
        base_loss,
        train_set,
        val_set,
        dataset_info,
        relaxed_equiv_schedule=beta_schedule,
    )

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    for name, result in results.items():
        print(
            f"\n=== {name}: n_params={result['n_params']:,}, "
            f"diverged={result['diverged']}, final_eval={result['final_eval']} ==="
        )


if __name__ == "__main__":
    main()
