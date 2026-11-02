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


@pytest.mark.parametrize("loop", ["sgd", "gradient_descent"])
def test_masked_adamw_preserves_zero_gradient_frozen_parameters(loop):
    """Mask the complete transform to exclude frozen leaves from weight decay."""
    initial = {"frozen": jnp.array([1.5, -2.0]), "weight": jnp.array(1.0)}
    optimizer = optax.masked(optax.adamw(0.01, weight_decay=0.2),
                             {"frozen": False, "weight": True})

    def objective(params):
        return params["weight"] ** 2

    if loop == "sgd":
        result, _ = run_sgd(lambda params, batch: objective(params), initial,
                            jnp.zeros((1, 1)), optimizer=optimizer, num_epochs=10)
    else:
        result, state, _ = run_gradient_descent(
            objective, initial, optimizer=optimizer, num_mstep_iters=3)
        result, _, _ = run_gradient_descent(
            objective, result, optimizer=optimizer, optimizer_state=state,
            num_mstep_iters=7)
    np.testing.assert_array_equal(result["frozen"], initial["frozen"])
    assert result["weight"] < initial["weight"]


def test_fit_sgd_adamw_uses_caller_trainable_mask():
    """The documented property mask preserves an actual frozen model weight."""
    from dynamax.linear_gaussian_ssm import LinearGaussianSSM
    from dynamax.parameters import ParameterProperties

    model = LinearGaussianSSM(1, 1)
    params, props = model.initialize(dynamics_weights=jnp.array([[0.85]]))
    props.dynamics.weights.trainable = False
    mask = tree_map(lambda prop: prop.trainable, props,
                    is_leaf=lambda prop: isinstance(prop, ParameterProperties))
    optimizer = optax.masked(optax.adamw(0.1, weight_decay=0.2), mask)
    fitted, losses = model.fit_sgd(params, props, jnp.zeros((16, 1)),
                                    num_epochs=1, optimizer=optimizer)
    np.testing.assert_array_equal(fitted.dynamics.weights, params.dynamics.weights)
    assert jnp.all(jnp.isfinite(losses))
