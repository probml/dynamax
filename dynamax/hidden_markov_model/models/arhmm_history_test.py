"""Check that ARHMM sampling and likelihood inputs use the same lag order."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dynamax.hidden_markov_model import LinearAutoregressiveHMM


@pytest.mark.parametrize("num_lags", [1, 2, 3])
@pytest.mark.parametrize("emission_dim", [1, 2])
@pytest.mark.parametrize("use_jit", [False, True])
def test_arhmm_sample_matches_chronological_history(num_lags, emission_dim, use_jit):
    """Check each sampled emission against its conditional distribution."""
    model = LinearAutoregressiveHMM(2, emission_dim, num_lags=num_lags)
    weights = jnp.arange(1, 2 * emission_dim**2 * num_lags + 1, dtype=float)
    weights = weights.reshape(2, emission_dim, num_lags * emission_dim) / 10
    params, _ = model.initialize(
        emission_weights=weights,
        emission_biases=jnp.zeros((2, emission_dim)),
        emission_covariances=jnp.tile(jnp.eye(emission_dim)[None], (2, 1, 1)),
    )
    previous = jnp.arange(1, num_lags * emission_dim + 1, dtype=float)
    previous = previous.reshape(num_lags, emission_dim)
    key = jr.PRNGKey(42)
    sample = partial(model.sample, num_timesteps=5)
    if use_jit:
        sample = jax.jit(sample)
    states, emissions = sample(params, key, prev_emissions=previous)

    # History is chronological; the weight blocks multiply newest-to-oldest lags.
    history = jnp.concatenate((previous, emissions))
    inputs = jnp.stack([history[t:t + num_lags][::-1].ravel() for t in range(5)])
    np.testing.assert_array_equal(model.compute_inputs(emissions, previous), inputs)

    _, initial_emission_key, remaining_key = jr.split(key, 3)
    remaining_keys = jr.split(remaining_key, 4)
    emission_keys = jnp.concatenate((
        initial_emission_key[None],
        jax.vmap(lambda key: jr.split(key)[0])(remaining_keys),
    ))
    expected = jax.vmap(
        lambda state, inputs, key: model.emission_distribution(
            params, state, inputs=inputs
        ).sample(seed=key)
    )(states, inputs, emission_keys)
    np.testing.assert_allclose(emissions, expected, rtol=1e-5, atol=1e-5)


def test_arhmm_default_history_is_zero():
    """Explicit zero history and omitted history give the same samples."""
    model = LinearAutoregressiveHMM(2, 2, num_lags=3)
    params, _ = model.initialize(jr.PRNGKey(0))
    key = jr.PRNGKey(1)
    implicit = model.sample(params, key, 5)
    explicit = model.sample(params, key, 5, prev_emissions=jnp.zeros((3, 2)))
    for actual, expected in zip(implicit, explicit):
        np.testing.assert_array_equal(actual, expected)
