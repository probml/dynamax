"""
Helpers for managing parameters and their properties as PyTrees.
"""
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_reduce, tree_map, register_pytree_node_class
import tensorflow_probability.substrates.jax.bijectors as tfb
from typing import Optional, runtime_checkable
from typing_extensions import Protocol

from dynamax.types import Scalar

@runtime_checkable
class ParameterSet(Protocol):
    """A :class:`NamedTuple` with parameters stored as :class:`jax.DeviceArray` in the leaf nodes.

    """
    pass

@runtime_checkable
class PropertySet(Protocol):
    """A matching :class:`NamedTuple` with :class:`ParameterProperties` stored in the leaf nodes.

    """
    pass


@register_pytree_node_class
class ParameterProperties:
    """A PyTree containing parameter metadata (properties).

    Note: the properties are stored in the aux_data of this PyTree so that
    changes will trigger recompilation of functions that rely on them.

    Args:
        trainable (bool): flag specifying whether or not to fit this parameter is adjustable.
        constrainer (Optional tfb.Bijector): bijector mapping to constrained form.

    """
    def __init__(self,
                 trainable: bool = True,
                 constrainer: Optional[tfb.Bijector] = None) -> None:
        self.trainable = trainable
        self.constrainer = constrainer

    def tree_flatten(self):
        """Flatten the PyTree into a tuple of aux_data and children."""
        return (), (self.trainable, self.constrainer)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the PyTree from the tuple of aux_data and children."""
        return cls(*aux_data)

    def __repr__(self):
        """Return a string representation of the PyTree."""
        return f"ParameterProperties(trainable={self.trainable}, constrainer={self.constrainer})"


def to_unconstrained(params: ParameterSet, props: PropertySet) -> ParameterSet:
    """Convert the constrained parameters to unconstrained form.

    Args:
        params: (nested) named tuple whose leaf values are DeviceArrays containing
                              parameter values.
        props: matching named tuple whose leaf values are ParameterProperties,
                containing an optional bijector that converts to unconstrained form,
                and a boolean flag specifying if the parameter is trainable or not.

    Returns:
        unc_params: named tuple containing parameters in unconstrained form.

    """
    to_unc = lambda value, prop: prop.constrainer.inverse(value) \
        if prop.constrainer is not None else value
    is_leaf = lambda node: isinstance(node, (ParameterProperties,))
    return tree_map(to_unc, params, props, is_leaf=is_leaf)


def from_unconstrained(unc_params: ParameterSet, props: PropertySet) -> ParameterSet:
    """Convert the unconstrained parameters to constrained form.

    Args:
        unc_params: (nested) named tuple whose leaf values are DeviceArrays containing
                              unconstrained parameter values.
        props: matching named tuple whose leaf values are ParameterProperties,
                containing an optional bijector that converts to unconstrained form,
                and a boolean flag specifying if the parameter is trainable or not.

    Returns:
        params: named tuple containing parameters in constrained form.
                 If a parameter is marked with trainable=False (frozen) in the properties structure,
                 it will be tagged with a "stop gradient". Thus the gradient of any loss function computed
                 using these frozen constrained parameters will be zero.

    """
    def from_unc(unc_value, prop):
        """Convert the unconstrained value to constrained form."""
        value = prop.constrainer(unc_value) if prop.constrainer is not None else unc_value
        value = lax.stop_gradient(value) if not prop.trainable else value
        return value

    is_leaf = lambda node: isinstance(node, (ParameterProperties,))
    return tree_map(from_unc, unc_params, props, is_leaf=is_leaf)


def log_det_jac_constrain(params: ParameterSet, props: PropertySet) -> Scalar:
    """Log determinant of the Jacobian matrix evaluated at the unconstrained parameters.

    Let x be the unconstrained parameter and f(x) be the constrained parameter, so
    that in code, `props.constrainer` is the Bijector f. To perform Hamiltonian
    Monte Carlo (HMC) on the unconstrained parameters we need the log determinant of
    the forward Jacobian, |df(x) / dx|. In math, this falls out as follows:

    ..math:
        log p(x) = log p(f(x)) + log |df(x) / dx|

    Args:
        params: PyTree whose leaf values are DeviceArrays
        props: matching PyTree whose leaf values are ParameterProperties

    Returns:
        logdet: the log determinant of the forward Jacobian.
    """
    unc_params = to_unconstrained(params, props)
    def _compute_logdet(unc_value, prop):
        """Compute the log determinant of the Jacobian matrix."""
        logdet = prop.constrainer.forward_log_det_jacobian(unc_value).sum() \
            if prop.constrainer is not None else 0.0
        return logdet if prop.trainable else 0.0

    is_leaf = lambda node: isinstance(node, (ParameterProperties,))
    logdets = tree_map(_compute_logdet, unc_params, props, is_leaf=is_leaf)
    return tree_reduce(jnp.add, logdets, 0.0)
