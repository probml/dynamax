from abc import ABC

from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax.bijectors as tfb


@register_pytree_node_class
class Parameter:

    def __init__(self, value, is_frozen=False, bijector=None):
        self.value = value
        self.is_frozen = is_frozen
        self.bijector = bijector if bijector is not None else tfb.Identity()

    def __repr__(self):
        return f"Parameter(value={self.value}, " \
               f"is_frozen={self.is_frozen}, " \
               f"bijector={self.bijector})"

    @property
    def unconstrained_value(self):
        return self.bijector(self.value)

    def tree_flatten(self):
        children = (self.value,)
        aux_data = self.is_frozen, self.bijector
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)


class Module(ABC):

    @property
    def unconstrained_params(self):
        # Find all parameters and convert to unconstrained
        items = sorted(self.__dict__.items())
        params = [prm.unconstrained_value for key, prm in items if isinstance(prm, Parameter) and not prm.is_frozen]

        return params

    @unconstrained_params.setter
    def unconstrained_params(self, values):
        items = sorted(self.__dict__.items())
        params = [val for key, val in items if isinstance(val, Parameter) and not val.is_frozen]
        assert len(params) == len(values)
        for param, value in zip(params, values):
            param.value = param.bijector.inverse(value)

    @property
    def hyperparams(self):
        """Helper property to get a PyTree of model hyperparameters."""
        items = sorted(self.__dict__.items())
        hyper_values = [val for key, val in items if not isinstance(val, Parameter)]
        return hyper_values

    @hyperparams.setter
    def hyperparams(self, values):
        items = sorted(self.__dict__.items())
        params = [val for key, val in items if not isinstance(val, Parameter)]
        assert len(params) == len(values)
        for param, value in zip(params, values):
            param.value = value

    # Generic implementation of tree_flatten and unflatten. This assumes that
    # the Parameters are all valid JAX PyTree nodes.
    def tree_flatten(self):
        items = sorted(self.__dict__.items())
        param_values = [val for key, val in items if isinstance(val, Parameter)]
        param_names = [key for key, val in items if isinstance(val, Parameter)]
        hyper_values = [val for key, val in items if not isinstance(val, Parameter)]
        hyper_names = [key for key, val in items if not isinstance(val, Parameter)]
        return param_values, (param_names, hyper_names, hyper_values)

    @classmethod
    def tree_unflatten(cls, aux_data, param_values):
        param_names, hyper_names, hyper_values = aux_data
        obj = object.__new__(cls)
        for name, value in zip(param_names, param_values):
            setattr(obj, name, value)
        for name, value in zip(hyper_names, hyper_values):
            setattr(obj, name, value)
        return obj
