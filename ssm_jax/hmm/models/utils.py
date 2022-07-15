from dataclasses import dataclass
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import chex


def _check_training_params(params):
    is_valid = 0 < len(params) <= 3
    allowed = set("ite")
    is_valid = is_valid and set(params.lower()) <= allowed
    return is_valid


@dataclass
class ParameterTransformation:
    initial_dist_params: chex.Array
    transition_dist_params: chex.Array
    emission_dist_params: Tuple[chex.Array]
    hyperparameters: Union[Tuple, List]
    flatten_fn: Callable
    unflatten_fn: Callable

    def init(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class InitialParametrization(ParameterTransformation):

    def init(self):
        return tuple([self.flatten_fn[0]])

    def update(self, params):
        complete_params = (*params, self.transition_dist_params, *self.emission_dist_params)
        return self.unflatten_fn(complete_params, self.hyperparameters)


class EmissionParametrization(ParameterTransformation):

    def init(self):
        return tuple(self.flatten_fn[2:])

    def update(self, params):
        complete_params = (self.initial_dist_params, self.transition_dist_params, *params)
        return self.unflatten_fn(complete_params, self.hyperparameters)


class TransitionParametrization(ParameterTransformation):

    def init(self):
        return tuple([self.flatten_fn[1]])

    def update(self, params):
        complete_params = (self.initial_dist_params, *params, *self.emission_dist_params)
        return self.unflatten_fn(complete_params, self.hyperparameters)


class InitialAndTransitionParametrization(ParameterTransformation):

    def init(self):
        return tuple(self.flatten_fn[:2])

    def update(self, params):
        complete_params = (*params, *self.emission_dist_params)
        return self.unflatten_fn(complete_params, self.hyperparameters)


class TransitionAndEmissionParametrization(ParameterTransformation):

    def init(self):
        return tuple(self.flatten_fn[1:])

    def update(self, params):
        trans_dist_params, *emission_dist_params = params
        complete_params = (self.initial_dist_params, trans_dist_params, *emission_dist_params)
        return self.unflatten_fn(complete_params, self.hyperparameters)


class InitialAndEmissionParametrization(ParameterTransformation):

    def init(self):
        return tuple(self.flatten_fn[0:1] + self.flatten_fn[2:])

    def update(self, params):
        initial_dist_params, *emission_dist_params = params
        complete_params = (initial_dist_params, self.transition_dist_params, *emission_dist_params)
        return self.unflatten_fn(complete_params, self.hyperparameters)


class DefaultParametrization(ParameterTransformation):

    def init(self):
        return self.flatten_fn

    def update(self, params):
        return self.unflatten_fn(params, self.hyperparameters)


def get_training_parametrization(initial_dist_params: chex.Array,
                                 transition_dist_params: chex.Array,
                                 emission_dist_params: Tuple[chex.Array],
                                 hyperparameters: Union[Tuple, List],
                                 flatten_fn: Callable,
                                 unflatten_fn: Callable,
                                 params_names: str = "ite"):
    if _check_training_params(params_names):
        sorted_params_names = sorted(params_names)
        if sorted_params_names == "e":
            parametrization = EmissionParametrization
        elif sorted_params_names == "i":
            parametrization = InitialParametrization
        elif sorted_params_names == "t":
            parametrization = TransitionParametrization
        elif sorted_params_names == "ei":
            parametrization = InitialAndEmissionParametrization
        elif sorted_params_names == "et":
            parametrization = TransitionAndEmissionParametrization
        elif sorted_params_names == "it":
            parametrization = InitialAndTransitionParametrization
        else:
            parametrization = DefaultParametrization
        return parametrization(initial_dist_params, transition_dist_params, emission_dist_params, hyperparameters,
                               flatten_fn, unflatten_fn)
    else:
        raise ValueError("params must be shorter than 4 characters and must contain only 'i', 't', 'e' standing for " +
                         "initial, transition and emission probabilities, respectively.")
