from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr
from jax import lax
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.tree_util import register_pytree_node_class, tree_map


@register_pytree_node_class
class Parameter:
    """A lightweight wrapper for parameters of a model. It combines the `value`
    (a JAX PyTree) with a flag `is_frozen` (bool) to specify whether or not
    the parameter should be updated during model learning, as well as a `bijector`
    (tensorflow_probability.bijectors.Bijector) to map the parameter to/from an
    unconstrained space.
    """

    def __init__(self, value, is_frozen=False, bijector=None, prior=None):
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

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def tree_flatten(self):
        children = (self.value,)
        aux_data = self.is_frozen, self.bijector
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)


class SSM(ABC):
    """A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form. It also uses
    these parameters to implement the tree_flatten and tree_unflatten methods necessary
    to register a model as a JAX PyTree.
    """
    @abstractmethod
    def initial_distribution(self, **covariates):
        """Return an initial distribution over latent states.

        Returns:
            dist (tfd.Distribution): distribution over initial latent state.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(self, state, **covariates):
        """Return a distribution over next latent state given current state.

        Args:
            state (PyTree): current latent state

        Returns:
            dist (tfd.Distribution): conditional distribution of next latent state.
        """
        raise NotImplementedError

    @abstractmethod
    def emission_distribution(self, state, **covariates):
        """Return a distribution over emissions given current state.

        Args:
            state (PyTree): current latent state.

        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

    def sample(self, key, num_timesteps, **covariates):
        """Sample a sequence of latent states and emissions.

        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """

        def _step(prev_state, args):
            key, covariate = args
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(prev_state, **covariate).sample(seed=key2)
            emission = self.emission_distribution(state, **covariate).sample(seed=key1)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_covariate = tree_map(lambda x: x[0], covariates)
        initial_state = self.initial_distribution(**initial_covariate).sample(seed=key1)
        initial_emission = self.emission_distribution(initial_state, **initial_covariate).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_covariates = tree_map(lambda x: x[1:], covariates)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_covariates))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def log_prob(self, states, emissions, **covariates):
        """Compute the log joint probability of the states and observations"""
        def _step(carry, args):
            lp, prev_state = carry
            state, emission, covariate = args
            lp += self.transition_distribution(prev_state, **covariate).log_prob(state)
            lp += self.emission_distribution(state, **covariate).log_prob(emission)
            return (lp, state), None

        # Compute log prob of initial time step
        initial_state = tree_map(lambda x: x[0], states)
        initial_emission = tree_map(lambda x: x[0], emissions)
        initial_covariate = tree_map(lambda x: x[0], covariates)
        lp = self.initial_distribution(**initial_covariate).log_prob(initial_state)
        lp += self.emission_distribution(initial_state, **initial_covariate).log_prob(initial_emission)

        # Scan over remaining time steps
        next_states = tree_map(lambda x: x[1:], states)
        next_emissions = tree_map(lambda x: x[1:], emissions)
        next_covariates = tree_map(lambda x: x[1:], covariates)
        (lp, _), _ = lax.scan(_step, (lp, initial_state), (next_states, next_emissions, next_covariates))
        return lp

    def log_prior(self):
        """Return the log prior probability of any model parameters.

        Returns:
            lp (Scalar): log prior probability.
        """
        return 0.0

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
        hyper_values = [val for key, val in items if (not isinstance(Parameter) or val.is_frozen)]
        return hyper_values

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
