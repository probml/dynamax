from typing import NamedTuple

from jax.flatten_util import ravel_pytree
from jax import jit
import jax.numpy as jnp
from optax._src import base

from dynamax.rebayes.diagonal_inference import (
    _full_covariance_condition_on,
    _fully_decoupled_ekf_condition_on,
    _variational_diagonal_ekf_condition_on,
)


class EKFState(NamedTuple):
    """
    Lightweight container for EKF parameters.
    """
    mean: base.Updates
    cov: base.Updates


def make_ekf_optimizer(
    pred_mean_fn,
    pred_cov_fn,
    init_var = 1.0,
    ekf_type = 'fcekf',
    num_iter = 1
    ) -> base.GradientTransformation:
    """Generate optax optimizer object for EKF.

    Args:
        pred_mean_fn (Callable): Emission mean function for EKF.
        pred_cov_fn (Callable): Emission covariance function for EKF.
        init_var (float, optional): Initial covariance factor. Defaults to 1.0.
        ekf_type (str, optional): One of ['fcekf', 'fdekf', 'vdekf']. Defaults to 'fcekf'.
        num_iter (int, optional): Number of posterior linearizations to perform. Defaults to 1.
    Returns:
        base.GradientTransformation: Optax optimizer object for EKF.
    """    
    if ekf_type not in ['fcekf', 'fdekf', 'vdekf']:
        raise ValueError(f"'ekf_type' must be one of ['fcekf, 'fdekf', 'vdekf']")
    
    def init_fn(params):
        flat_params, _ = ravel_pytree(params)
        if ekf_type == 'fcekf':
            cov = init_var * jnp.eye(flat_params.shape[0])
        else:
            cov = init_var * jnp.ones_like(flat_params)
        return EKFState(mean=params, cov=cov)
    
    @jit
    def update_fn(updates, state, params=None):
        # Updates are new set of data points
        x, y = updates
        flat_mean, unflatten_fn = ravel_pytree(state.mean)
        if ekf_type == 'fcekf':
            mean, cov = _full_covariance_condition_on(
                flat_mean, state.cov, pred_mean_fn, pred_cov_fn, x, y, num_iter
            )
        elif ekf_type == 'fdekf':
            mean, cov = _fully_decoupled_ekf_condition_on(
                flat_mean, state.cov, pred_mean_fn, pred_cov_fn, x, y, num_iter
            )
        else:
            mean, cov = _variational_diagonal_ekf_condition_on(
                flat_mean, state.cov, pred_mean_fn, pred_cov_fn, x, y, num_iter
            )
        updates = unflatten_fn(mean - flat_mean)
        return updates, EKFState(mean=unflatten_fn(mean), cov=cov)
    
    return base.GradientTransformation(init_fn, update_fn)