from copy import deepcopy

import chex
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
