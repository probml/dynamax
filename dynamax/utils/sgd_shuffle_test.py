"""Regression tests for shuffling across multiple SGD minibatches."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from dynamax.utils.optimize import run_sgd


@pytest.mark.parametrize(("num_samples", "batch_size"), [(10, 5), (11, 5), (10, 1), (5, 5)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_run_sgd_respects_requested_shuffle(num_samples, batch_size, shuffle):
    """Compare ordered scalar SGD updates with an explicit permutation reference."""
    targets = np.arange(num_samples, dtype=float)
    key = jr.PRNGKey(7)
    learning_rate = 0.1
    epochs = 3

    def loss(parameter, batch):
        """Quadratic objective with a known derivative."""
        return 0.5 * jnp.mean((parameter - batch["targets"])**2)

    parameter, losses = run_sgd(
        loss,
        params=jnp.array(0.0),
        dataset={"targets": jnp.asarray(targets)},
        optimizer=optax.sgd(learning_rate),
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=shuffle,
        key=key,
    )

    expected = 0.0
    expected_losses = []
    for epoch_key in jr.split(key, epochs):
        indices = (np.asarray(jr.permutation(epoch_key, num_samples))
                   if shuffle else np.arange(num_samples))
        batch_losses = []
        for start in range(0, num_samples, batch_size):
            batch = targets[indices[start:start + batch_size]]
            batch_losses.append(0.5 * np.mean((expected - batch)**2))
            expected -= learning_rate * (expected - np.mean(batch))
        expected_losses.append(np.mean(batch_losses))

    np.testing.assert_allclose(parameter, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(losses, expected_losses, rtol=1e-5, atol=1e-6)
