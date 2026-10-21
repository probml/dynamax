"""Check optimizers that need the current parameters in their update rule."""

import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import optax
import pytest

from dynamax.utils.optimize import run_gradient_descent, run_sgd


@pytest.mark.parametrize("loop", ["sgd", "gradient_descent"])
@pytest.mark.parametrize("make_optimizer", [
    pytest.param(lambda: optax.adamw(0.1, weight_decay=0.2), id="adamw"),
    pytest.param(lambda: optax.chain(optax.add_decayed_weights(0.2), optax.sgd(0.1)),
                 id="decayed-sgd"),
    pytest.param(lambda: optax.sgd(0.1), id="sgd-control"),
])
def test_optimization_loops_supply_current_parameters(loop, make_optimizer):
    """Compare three parameter-dependent steps with direct Optax updates."""
    initial = {"weight": jnp.array([1.0, -1.0]), "bias": jnp.array(0.5)}
    target = {"weight": jnp.array([2.0, 1.0]), "bias": jnp.array(-1.0)}

    def objective(parameter):
        """Quadratic objective with an analytic PyTree gradient."""
        return 0.5 * sum(jnp.sum((parameter[k] - target[k])**2) for k in target)

    if loop == "gradient_descent":
        actual, _, losses = run_gradient_descent(
            objective, initial, optimizer=make_optimizer(), num_mstep_iters=3
        )
    else:
        actual, losses = run_sgd(
            lambda parameter, batch: objective(parameter),
            initial,
            dataset=jnp.zeros((1, 1)),
            optimizer=make_optimizer(),
            num_epochs=3,
        )

    optimizer = make_optimizer()
    expected = initial
    state = optimizer.init(expected)
    expected_losses = []
    for _ in range(3):
        expected_losses.append(objective(expected))
        gradient = tree_map(lambda value, desired: value - desired, expected, target)
        updates, state = optimizer.update(gradient, state, params=expected)
        expected = optax.apply_updates(expected, updates)

    tree_map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5), actual, expected)
    np.testing.assert_allclose(losses, jnp.stack(expected_losses), rtol=1e-5)
