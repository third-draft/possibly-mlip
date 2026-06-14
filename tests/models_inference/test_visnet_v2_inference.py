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

import jax
import jax.numpy as jnp


def test_visnet_v2_outputs_finite_energy_and_forces(setup_system, visnet_v2_force_field):
    _, graph = setup_system

    result = jax.jit(visnet_v2_force_field)(graph)

    assert jnp.all(jnp.isfinite(result.energy))
    assert jnp.all(jnp.isfinite(result.forces))
    assert result.stress is not None and jnp.all(jnp.isfinite(result.stress))


def test_visnet_v2_grad_params(setup_system, visnet_v2_force_field, pad_graph):
    _, graph = setup_system
    graph = pad_graph(graph, 4, 34, 92)
    _apply = jax.jit(visnet_v2_force_field.predictor.apply)
    params = visnet_v2_force_field.params

    energy_loss = lambda p, g: jnp.sum(_apply(p, g).globals.energy)  # noqa: E731
    forces_loss = lambda p, g: jnp.sum(_apply(p, g).nodes.forces ** 2)  # noqa: E731
    stress_loss = lambda p, g: jnp.sum(_apply(p, g).globals.stress ** 2)  # noqa: E731

    for loss_fn in [energy_loss, forces_loss, stress_loss]:
        backprop = jax.grad(loss_fn)
        params_grad = backprop(params, graph)
        leaves_grad = jax.tree.leaves(params_grad)

        for p in leaves_grad:
            assert not jnp.any(jnp.isnan(p)), f"NaN for {loss_fn.__name__}"
            assert not jnp.any(jnp.isinf(p)), f"Inf for {loss_fn.__name__}"
