from copy import deepcopy

import chex
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import tensorflow_probability.substrates.jax.bijectors as tfb


@chex.dataclass
class ParameterProperties:
    trainable: bool = True
    constrainer: tfb.Bijector = tfb.Identity()


def to_unconstrained(params, param_props):
    """Extract the trainable parameters and convert to unconstrained, then return
    unconstrained parameters and fixed parameters.

    Args:
        params (dict): (nested) dictionary whose leaf values are DeviceArrays
        param_props (dict): matching (nested) dictionary whose leaf values are ParameterProperties

    Returns:
        unc_params (dict): (nested) dictionary whose values are the
            unconstrainend parameter values, but only for the parameters that
            are marked trainable in `param_props`.
        fixed_params (dict): (nested) dictionary whose values are the
            existing value, but only for the parameters that are marked not
            trainable in `param_props`.
    """
    unc_params, fixed_params = dict(), dict()
    for k, v in params.items():
        if isinstance(v, dict):
            unc_params[k], fixed_params[k] = to_unconstrained(v, param_props[k])
        elif param_props[k].trainable:
            unc_params[k] = param_props[k].constrainer.inverse(v)
        else:
            fixed_params[k] = v
    return unc_params, fixed_params


def from_unconstrained(unc_params, fixed_params, param_props):
    """Convert the unconstrained parameters to constrained form and
    combine them with the fixed parameters.

    Args:
        unc_params (dict): (nested) dictionary whose leaf values are DeviceArrays
        fixed_params (dict): (nested) dictionary whose leaf values are DeviceArrays
        param_props (dict): matching (nested) dictionary whose leaf values are ParameterProperties

    Returns:
        params (dict): combined dictionary of unconstrained and fixed parameters
            in their natural (constrained) form.
    """
    params = deepcopy(fixed_params)
    for k, v in unc_params.items():
        if isinstance(v, dict):
            params[k] = from_unconstrained(unc_params[k], fixed_params[k], param_props[k])
        else:
            params[k] = param_props[k].constrainer(v)
    return params


def log_det_jac_constrain(unc_params, fixed_params, param_props):
    """Log determinant of the Jacobian matrix evaluated at the unconstrained parameters.
    """
    log_det_jac = 0
    for k, v in unc_params.items():
        if isinstance(v, dict):
            ldj_inc = log_det_jac_constrain(unc_params[k], fixed_params[k], param_props[k])
            log_det_jac += ldj_inc
        else:
            log_det_jac += param_props[k].constrainer.forward_log_det_jacobian(v)
    return log_det_jac


def flatten(params):
    """Flatten the (unconstrained) parameters to an 1-d numpy array.

    Returns:
        params_flat: flat parameters
        structure (dict): structure information of parameters, used to unflatten the parameters.
    """
    # Flatten the tree of parameters into leaves
    tree_flat, tree_structure = tree_flatten(params)
    # Flatten leaves of the tree
    array_shapes = [x.shape for x in tree_flat]
    params_flat = jnp.concatenate([x.flatten() for x in tree_flat])

    return params_flat, {'array_shapes': array_shapes, 'tree_structure': tree_structure}


def unflatten(structure, params_flat):
    """Unflatten the (unconstrained) parameters.

    Args:
        params_flat: flat parameters
        structure (dict): structure information of parameters
    Returns:
        params: (unconstrained) parameters
    """
    array_shapes = structure['array_shapes']
    tree_structure = structure['tree_structure']
    # Restore leaves of the parameter tree, each leave is a numpy array
    _sizes = jnp.cumsum(jnp.array([jnp.array(x).prod() for x in array_shapes]))
    cum_sizes = jnp.concatenate([jnp.zeros(1), _sizes]).astype(int)
    arrays = [params_flat[cum_sizes[i]:cum_sizes[i+1]].reshape(array_shapes[i])
              for i in range(len(cum_sizes)-1)]
    # Restore the tree of parameters
    params = tree_unflatten(tree_structure, arrays)

    return params
