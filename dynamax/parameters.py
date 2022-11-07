from copy import deepcopy

from dataclasses import is_dataclass

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
        params (dataclass): (nested) dataclass whose leaf values are DeviceArrays containing 
                              parameter values.
        param_props (dict): matching (nested) dictionary whose leaf values are ParameterProperties

    Returns:
        unc_params (dict): (nested) dictionary whose values are the unconstrainend parameter 
                            values, but only for the parameters that are marked trainable in
                            `param_props`.
        params (dataclass): the original `params` input.
    """
    unc_params = dict()
    for k, v in params.items():
        if is_dataclass(v):
            unc_params[k], _ = to_unconstrained(v, param_props[k])
        elif param_props[k].trainable:
            unc_params[k] = param_props[k].constrainer.inverse(v)
    return unc_params, params


def from_unconstrained(unc_params, orig_params, param_props):
    """Convert the unconstrained parameters to constrained form and
    combine them with the original parameters.

    Args:
        unc_params (dict): (nested) dictionary whose leaf values are DeviceArrays
        orig_params (dataclass): (nested) dataclasses whose leaf values are DeviceArrays containing
                             parameter values
        param_props (dict): matching (nested) dictionary whose leaf values are ParameterProperties

    Returns:
        params (dataclass): copy of `orig_params` where values have been replaced by the corresponding 
                             leaf in `unc_params`, if present.
    """
    params = deepcopy(orig_params)
    for k, v in unc_params.items():
        if isinstance(v, dict):
            params = params.replace(
                **{k: from_unconstrained(unc_params[k], orig_params[k], param_props[k])}
            )
        else:
            params = params.replace(**{k:param_props[k].constrainer(v)})
    return params


def log_det_jac_constrain(unc_params, param_props):
    """Log determinant of the Jacobian matrix evaluated at the unconstrained parameters.
    """
    log_det_jac = 0
    for k, v in unc_params.items():
        if isinstance(v, dict):
            ldj_inc = log_det_jac_constrain(unc_params[k], param_props[k])
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
