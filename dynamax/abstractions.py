from abc import ABC
from abc import abstractmethod
from fastprogress.fastprogress import progress_bar
from functools import partial
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit, lax, vmap
from jax.tree_util import tree_map
from warnings import warn

from dynamax.optimize import run_sgd
from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.utils import pytree_stack, ensure_array_has_batch_dim


class SSM(ABC):
    """A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form. It also uses
    these parameters to implement the tree_flatten and tree_unflatten methods necessary
    to register a model as a JAX PyTree.
    """

    @abstractmethod
    def initial_distribution(self, params, inputs=None):
        """Return an initial distribution over latent states.

        Returns:
            dist (tfd.Distribution): distribution over initial latent state.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(self, params, state, inputs=None):
        """Return a distribution over next latent state given current state.

        Args:
            state (PyTree): current latent state
        Returns:
            dist (tfd.Distribution): conditional distribution of next latent state.
        """
        raise NotImplementedError

    @abstractmethod
    def emission_distribution(self, params, state, inputs=None):
        """Return a distribution over emissions given current state.

        Args:
            state (PyTree): current latent state.
        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def emission_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's emissions.
        For example, a Gaussian HMM with D dimensional emissions would return (D,).
        """
        raise NotImplementedError

    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.
        """
        return None

    def sample(self, params, key, num_timesteps, inputs=None):
        """Sample a sequence of latent states and emissions.

        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """

        def _step(prev_state, args):
            key, inpt = args
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state, inpt).sample(seed=key2)
            emission = self.emission_distribution(params, state, inpt).sample(seed=key1)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, initial_input).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def log_prob(self, params, states, emissions, inputs=None):
        """Compute the log joint probability of the states and observations"""

        def _step(carry, args):
            lp, prev_state = carry
            state, emission, inpt = args
            lp += self.transition_distribution(params, prev_state, inpt).log_prob(state)
            lp += self.emission_distribution(params, state, inpt).log_prob(emission)
            return (lp, state), None

        # Compute log prob of initial time step
        initial_state = tree_map(lambda x: x[0], states)
        initial_emission = tree_map(lambda x: x[0], emissions)
        initial_input = tree_map(lambda x: x[0], inputs)
        lp = self.initial_distribution(params, initial_input).log_prob(initial_state)
        lp += self.emission_distribution(params, initial_state, initial_input).log_prob(initial_emission)

        # Scan over remaining time steps
        next_states = tree_map(lambda x: x[1:], states)
        next_emissions = tree_map(lambda x: x[1:], emissions)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        (lp, _), _ = lax.scan(_step, (lp, initial_state), (next_states, next_emissions, next_inputs))
        return lp

    def log_prior(self, params):
        """Return the log prior probability of any model parameters.

        Returns:
            lp (Scalar): log prior probability.
        """
        return 0.0

    def fit_em(self, params, props, emissions, inputs=None, num_iters=50, verbose=True):
        """Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM)."""
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        @jit
        def em_step(params, m_step_state):
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            return params, m_step_state, lp

        log_probs = []
        params = params
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        for _ in pbar:
            params, m_step_state, marginal_loglik = em_step(params, m_step_state)
            log_probs.append(marginal_loglik)
        return params, jnp.array(log_probs)

    def fit_sgd(self,
                curr_params,
                param_props,
                emissions,
                inputs=None,
                optimizer=optax.adam(1e-3),
                batch_size=1,
                num_epochs=50,
                shuffle=False,
                key=jr.PRNGKey(0)):
        """Compute parameter MLE/ MAP estimate using Stochastic gradient descent (SGD).

        Fit this HMM by running SGD on the marginal log likelihood.
        Note that batch_emissions is initially of shape (N,T)
        where N is the number of independent sequences and
        T is the length of a sequence. Then, a random susbet with shape (B, T)
        of entire sequence, not time steps, is sampled at each step where B is
        batch size.

        Args:
            batch_emissions (chex.Array): Independent sequences.
            optmizer (optax.Optimizer): Optimizer.
            batch_size (int): Number of sequences used at each update step.
            num_epochs (int): Iterations made through entire dataset.
            shuffle (bool): Indicates whether to shuffle minibatches.
            key (chex.PRNGKey): RNG key to shuffle minibatches.
        Returns:
            losses: Output of loss_fn stored at each step.
        """
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        curr_unc_params, fixed_params = to_unconstrained(curr_params, param_props)

        def _loss_fn(unc_params, minibatch):
            """Default objective function."""
            params = from_unconstrained(unc_params, fixed_params, param_props)
            minibatch_emissions, minibatch_inputs = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            minibatch_lls = vmap(partial(self.marginal_log_prob, params))(minibatch_emissions, minibatch_inputs)
            lp = self.log_prior(params) + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

        dataset = (batch_emissions, batch_inputs)
        unc_params, losses = run_sgd(_loss_fn,
                                     curr_unc_params,
                                     dataset,
                                     optimizer=optimizer,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     shuffle=shuffle,
                                     key=key)

        params = from_unconstrained(unc_params, fixed_params, param_props)
        return params, losses
