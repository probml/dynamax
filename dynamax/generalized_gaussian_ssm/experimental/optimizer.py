from functools import partial
import collections.abc
import warnings

import jax.numpy as jnp
from jax import jit
import chex

from dynamax.generalized_gaussian_ssm.inference import _condition_on, EKFIntegrals, UKFIntegrals, GHKFIntegrals
from dynamax.generalized_gaussian_ssm.experimental.diagonal_inference import _fully_decoupled_ekf_condition_on, _variational_diagonal_ekf_condition_on

@chex.dataclass
class CMGFOptimizerParams:
    """
    Lightweight wrapper for CMGFOptimizer parameters.
    """
    mean: jnp.ndarray
    cov: jnp.ndarray


def check_optimizer_initial_params(filter_type, num_iter, alpha, beta, kappa, order):
    """Check initializer parameters for CMGFOptimizer and raise ValueError for invalid
    parameters, warn about parameters that will be ignored, and return parameters
    filled with default values.

    Args:
        filter_type (str): Type of CMGF to use for optimizer. One of ['ekf', 'ukf', 'ghkf'].
        num_iter (int): Number of posterior linearization steps.
        alpha (float): alpha hyperparameter for ukf optimizer.
        beta (float): beta hyperparameter for ukf optimizer.
        kappa (float): kappa hyperparameter for ukf optimizer.
        order (int): order of Gauss-Hermite quadrature for ghkf optimizer.

    Returns:
        processed_params (list): Parameters where missing values are filled with default values.
    """    
    if not isinstance(num_iter, int) or num_iter < 1:
        raise ValueError(
            f"num_iter={num_iter} is not a valid num_iter value. Input a positive integer."
        )
    params = {'alpha': alpha, 'beta': beta, 'kappa': kappa, 'order': order}
    for param_key, param_val in params.items():
        if param_key == 'order':
            if (isinstance(param_val, int) and param_val < 1) or (not isinstance(param_val, int) and param_val):
                raise ValueError(
                    f"{param_key}={param_val} is not a valid {param_key} value. "
                    "Input a positive integer."
                )
        else:
            if (isinstance(param_val, float) and param_val < 0) or (not isinstance(param_val, int) and param_val):
                raise ValueError(
                    f"{param_key}={param_val} is not a valid {param_key} value. "
                    "Input a non-negative number."
                )

    warning_message = lambda param_key, param_val, filter_type: (f"Concrete value for {param_key}={param_val} was provided. "
                                                                 f"This will be ignored by the cmgf-{filter_type} optimizer.")
    if filter_type == 'ekf':
        for param_key, param_val in params.items():
            if param_val:
                warnings.warn(
                    warning_message(param_key, param_val, filter_type)
                )
        filter = EKFIntegrals()
    elif filter_type == 'ukf':
        if params['order']:
            warnings.warn(
                warning_message('order', params['order'], filter_type)
            )
        ukf_default_dict = {'alpha': jnp.sqrt(3), 'beta': 2, 'kappa': 1}
        for param_key, param_val in ukf_default_dict.items():
            if not params[param_key]:
                params[param_key] = ukf_default_dict[param_key]
        filter = UKFIntegrals(alpha=params['alpha'], beta=params['beta'], kappa=params['kappa'])
    elif filter_type == 'ghkf':
        for param_key, param_val in params.items():
            if param_key != 'order' and param_val:
                warnings.warn(
                    warning_message(param_key, param_val, filter_type)
                )
        if not params['order']:
            params['order'] = 5
        filter = GHKFIntegrals(order=params['order'])
    else:
        raise ValueError(
            f"{filter_type} is not a valid filter_type. Input one of 'ekf', 'ukf', or 'ghkf'."
        )
    processed_params = [filter, num_iter] + list(params.values())
    return processed_params


def check_optimizer_params(params, params_name):
    """Check if params have valid form and raise ValueError otherwise.
    If a sequence of length 2 is provided, wraps it into CMGFOptimizerParams
    and returns the object.

    Args:
        params (CMGFOptimizerParams): Parameters.
        params_name (str): Name of parameters, e.g. 'params' or 'opt_state'. Used for error message.

    Returns:
        params: CMGFOptimizerParams object for the params.
    """    
    if isinstance(params, CMGFOptimizerParams):
        return params
    if not isinstance(params, collections.abc.Sequence):
        raise ValueError(
            f"'{params_name}' object of type {type(params)} was provided. Provide a valid CMGFOptimizerParams object."
        )
    if len(params) != 2:
        raise ValueError(
            f"'{params_name}' object of length {len(params)} was provided. Input a valid object of length 2."
        )
    params = CMGFOptimizerParams(mean=params[0], cov=params[1])
    return params


class CMGFOptimizer:
    """
    Lightweight wrapper that closely follows the style of optax optimizers.
    """    
    def __init__(self, filter_type='ekf', num_iter=1, alpha=None, beta=None, kappa=None, order=None):
        self._filter, self._num_iter, self._alpha, self._beta, self._kappa, self._order = \
            check_optimizer_initial_params(filter_type, num_iter, alpha, beta, kappa, order)

    def init(self, params: CMGFOptimizerParams):
        return check_optimizer_params(params, 'params')

    @partial(jit, static_argnums=(0, 2, 3,))
    def update(self, x, pred_mean_fn, pred_cov_fn, true_y, opt_state):
        prior_mean, prior_cov = check_optimizer_params(opt_state, 'opt_state').values()
        mean, cov = _full_covariance_condition_on(prior_mean, prior_cov, pred_mean_fn, pred_cov_fn, x, true_y, self._num_iter)
        post_opt_state = CMGFOptimizerParams(mean=mean, cov=cov)
        updates = mean - prior_mean
        return updates, post_opt_state


class FDEKFOptimizer:
    """
    Lightweight wrapper that closely follows the style of optax optimizers.
    """    
    def __init__(self, filter_type='ekf', num_iter=1, alpha=None, beta=None, kappa=None, order=None):
        self._filter, self._num_iter, self._alpha, self._beta, self._kappa, self._order = \
            check_optimizer_initial_params(filter_type, num_iter, alpha, beta, kappa, order)

    def init(self, params: CMGFOptimizerParams):
        return check_optimizer_params(params, 'params')

    @partial(jit, static_argnums=(0, 2, 3,))
    def update(self, x, pred_mean_fn, pred_cov_fn, true_y, opt_state):
        prior_mean, prior_cov = check_optimizer_params(opt_state, 'opt_state').values()
        mean, cov = _fully_decoupled_ekf_condition_on(prior_mean, prior_cov, pred_mean_fn, pred_cov_fn, x, true_y, self._num_iter)
        post_opt_state = CMGFOptimizerParams(mean=mean, cov=cov)
        updates = mean - prior_mean
        return updates, post_opt_state


class VDEKFOptimizer:
    """
    Lightweight wrapper that closely follows the style of optax optimizers.
    """    
    def __init__(self, filter_type='ekf', num_iter=1, alpha=None, beta=None, kappa=None, order=None):
        self._filter, self._num_iter, self._alpha, self._beta, self._kappa, self._order = \
            check_optimizer_initial_params(filter_type, num_iter, alpha, beta, kappa, order)

    def init(self, params: CMGFOptimizerParams):
        return check_optimizer_params(params, 'params')

    @partial(jit, static_argnums=(0, 2, 3,))
    def update(self, x, pred_mean_fn, pred_cov_fn, true_y, opt_state):
        prior_mean, prior_cov = check_optimizer_params(opt_state, 'opt_state').values()
        mean, cov = _variational_diagonal_ekf_condition_on(prior_mean, prior_cov, pred_mean_fn, pred_cov_fn, x, true_y, self._num_iter)
        post_opt_state = CMGFOptimizerParams(mean=mean, cov=cov)
        updates = mean - prior_mean
        return updates, post_opt_state