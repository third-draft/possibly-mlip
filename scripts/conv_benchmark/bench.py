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

"""Timing / memory harness shared by all benchmark targets.

Adapted from `scripts/profile_esen_layer.py::_bench` and
`scripts/benchmark_esen_pretrained.py::memory_stats`.
"""

import time
from typing import Any, Callable

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from mlip.graph import Graph

from .layers import BenchTarget


def memory_stats() -> tuple[float, float]:
    """Returns `(peak_bytes_in_use, bytes_in_use)` for the default device, in MB.

    Returns `(nan, nan)` if the backend does not report memory stats (e.g. CPU).
    """
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return float("nan"), float("nan")
    return stats["peak_bytes_in_use"] / 1e6, stats["bytes_in_use"] / 1e6


def time_call(
    fn: Callable[[], Any], n_warmup: int = 3, n_iters: int = 10
) -> float:
    """Returns the mean wall-clock time of `fn()` in milliseconds."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn())

    start = time.perf_counter()
    for _ in range(n_iters):
        jax.block_until_ready(fn())
    elapsed = (time.perf_counter() - start) / n_iters
    return elapsed * 1e3


def _to_loss(output: Any) -> jax.Array:
    """Reduces a benchmark target's output to a scalar for `jax.grad`."""
    if isinstance(output, Graph):
        output = output.nodes.features["latent"]
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, e3nn.IrrepsArray):
        output = output.array
    return jnp.sum(output.astype(jnp.float32) ** 2)


def benchmark_target(
    target: BenchTarget,
    key: jax.Array,
    with_backward: bool = True,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> dict[str, Any]:
    """Times forward (and optionally forward+backward) passes of `target`.

    Returns a row dict with keys `name`, `fwd_ms`, `bwd_ms` (or `None`),
    `peak_mb`, `in_use_mb`.
    """
    params = target.init(key)

    forward_jit = jax.jit(target.apply_fn)
    fwd_ms = time_call(
        lambda: forward_jit(params, *target.inputs), n_warmup, n_iters
    )

    bwd_ms = None
    if with_backward:

        def loss(params, *inputs):
            return _to_loss(target.apply_fn(params, *inputs))

        # Weightless tensor products (`params == {}`) have no leaves to
        # differentiate w.r.t.; differentiate w.r.t. the first input instead so the
        # forward computation isn't dead-code-eliminated.
        has_params = len(jax.tree_util.tree_leaves(params)) > 0
        argnum = 0 if has_params else 1

        grad_jit = jax.jit(jax.grad(loss, argnums=argnum))
        bwd_ms = time_call(
            lambda: grad_jit(params, *target.inputs), n_warmup, n_iters
        )

    peak_mb, in_use_mb = memory_stats()

    return {
        "name": target.name,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "peak_mb": peak_mb,
        "in_use_mb": in_use_mb,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'name':<20s} {'fwd (ms)':>12s} {'fwd+bwd (ms)':>14s} {'peak (MB)':>12s} {'in_use (MB)':>12s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        bwd = f"{row['bwd_ms']:.4f}" if row["bwd_ms"] is not None else "-"
        print(
            f"{row['name']:<20s} {row['fwd_ms']:>12.4f} {bwd:>14s} "
            f"{row['peak_mb']:>12.1f} {row['in_use_mb']:>12.1f}"
        )
