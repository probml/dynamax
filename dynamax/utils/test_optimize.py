from dynamax.utils.optimize import run_sgd
import jax.numpy as jnp
from optax import adam
from numpy.testing import assert_allclose


def test_run_sgd():
    """Test that run_sgd solves an exactly solvable problem."""

    def _loss(a, x):
        return  jnp.sum((x - a)**2 / 2)

    # Average `mini_batch_1`: 0.0
    mini_batch_1 = jnp.array([ 0.5333575 ,  1.5523977 , -0.34479547, -0.80614984, -0.93481004])
    # Average `mini_batch_2`: 1.0
    mini_batch_2 = jnp.array([0.52032334, 1.6625587 , 1.1381058 , 1.2635592 , 0.41545272])
    # Average `X_train`: 0.5
    X_train = jnp.concatenate([mini_batch_1, mini_batch_2]).reshape(-1, 1)

    param_init = jnp.array(0.5)
    settings = {
        'params': param_init, 'num_epochs': 10_000, 'optimizer': adam(1e-3), 'batch_size': 5,
    }
    # Train on mini_batch_1 with batch size five (=full dataset).
    solution_mini_batch_1, _ = run_sgd(_loss, dataset=mini_batch_1.reshape(-1, 1), **settings)
    assert_allclose(solution_mini_batch_1, 0.0, atol=1e-3, rtol=1e-3)

    # Train on X_train with mini batch size five (=half the dataset).
    solution, losses = run_sgd(_loss, dataset=X_train, **settings)
    assert_allclose(solution, 0.5, atol=1e-3)
    num_batches = len(X_train) / len(mini_batch_1)
    assert_allclose(losses[-1], _loss(0.5, X_train) / num_batches, atol=1e-3, rtol=1e-3)
