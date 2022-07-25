from abc import ABC

from ssm_jax.hmm.models.parameter import Parameter


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
