from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from ssm_jax.parameters import ParameterProperties
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM


@chex.dataclass
class MultinomialHMMSuffStats:
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    sum_x: chex.Array


class MultinomialHMM(StandardHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 num_classes,
                 num_trials,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.num_trials = num_trials
        self.emission_prior_concentration = emission_prior_concentration * jnp.ones(num_classes)

    def random_initialization(self, key):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_probs = jr.dirichlet(key3, jnp.ones(self.num_classes), (self.num_states, self.emission_dim))
        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(probs=emission_probs))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered())))
        return  params, param_props

    def emission_distribution(self, params, state):
        return tfd.Independent(tfd.Multinomial(self.num_trials, probs=params['emissions']['probs'][state]),
                               reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(
            params['transitions']['transition_matrix']).sum()
        lp += tfd.Dirichlet(self.emission_prior_concentration).log_prob(
            params['emissions']['probs']).sum()
        return lp

    def e_step(self, params, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """
        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(params),
                                     self._compute_transition_matrices(params),
                                     self._compute_conditional_logliks(params, emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(params['transitions']['transition_matrix'], posterior)

            # Compute the expected sufficient statistics
            sum_x = jnp.einsum("tk, tdi->kdi", posterior.smoothed_probs, emissions)

            # Pack into a dataclass
            stats = MultinomialHMMSuffStats(
                marginal_loglik=posterior.marginal_loglik,
                initial_probs=initial_probs,
                trans_probs=trans_probs,
                sum_x=sum_x,
            )
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, params, param_props, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        # Then maximize the expected log probability as a fn of model parameters
        params['emissions']['probs'] = tfd.Dirichlet(self.emission_prior_concentration + stats.sum_x).mode()
        return params