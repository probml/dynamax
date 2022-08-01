from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr
from jax import lax
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.tree_util import register_pytree_node_class


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
        self.prior = prior

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

    def prior_log_prob(self):
        return jnp.sum(self.prior.log_prob(self.value)) if self.prior is not None else 0

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
    def initial_distribution(self):
        """Return an initial distribution over latent states.

        Returns:
            dist (tfd.Distribution): distribution over initial latent state.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(self, state):
        """Return a distribution over next latent state given current state.

        Args:
            state (PyTree): current latent state

        Returns:
            dist (tfd.Distribution): conditional distribution of next latent state.
        """
        raise NotImplementedError

    @abstractmethod
    def emission_distribution(self, state):
        """Return a distribution over emissions given current state.

        Args:
            state (PyTree): current latent state.

        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

    def sample(self, key, num_timesteps):
        """Sample a sequence of latent states and emissions.

        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """

        def _step(state, key):
            key1, key2 = jr.split(key, 2)
            emission = self.emission_distribution(state).sample(seed=key1)
            next_state = self.transition_distribution(state).sample(seed=key2)
            return next_state, (state, emission)

        # Sample the initial state
        key1, key = jr.split(key, 2)
        initial_state = self.initial_distribution().sample(seed=key1)

        # Sample the remaining emissions and states
        keys = jr.split(key, num_timesteps)
        _, (states, emissions) = lax.scan(_step, initial_state, keys)
        return states, emissions

    def log_prob(self, states, emissions):
        """Compute the log joint probability of the states and observations"""
        def _step(carry, args):
            lp, prev_state = carry
            state, emission = args
            lp += self.transition_distribution(prev_state).log_prob(state)
            lp += self.emission_distribution(state).log_prob(emission)
            return (lp, state), None

        # Compute log prob of initial time step
        lp = self.initial_distribution().log_prob(states[0])
        lp += self.emission_distribution(states[0]).log_prob(emissions[0])

        # Scan over remaining time steps
        (lp, _), _ = lax.scan(_step, (lp, states[0]), (states[1:], emissions[1:]))
        return lp

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

    def prior_log_prob(self):
        items = sorted(self.__dict__.items())
        prior_log_probs = [
            val.prior_log_prob()
            for _, val in items
            if isinstance(val, Parameter) and not val.is_frozen and val.prior is not None
        ]
        return sum(prior_log_probs)

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
