import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_map

from .core import hmm_filter, hmm_smoother

from distrax import (
    MultivariateNormalFullCovariance as MVN,
    Categorical,
    Dirichlet)


class GaussianHMM:

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_covariance_matrices,
                 initial_probabilities_prior_concentration=1.0001,
                 transition_matrix_prior_concentration=1.0001):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        self.num_states, self.emission_dim = emission_means.shape

        # Check shapes
        assert initial_probabilities.shape == (self.num_states,)
        assert transition_matrix.shape == (self.num_states, self.num_states)
        assert emission_covariance_matrices.shape == \
            (self.num_states, self.emission_dim, self.emission_dim)

        # Construct the model from distrax distributions
        self.initial_distribution = Categorical(probs=initial_probabilities)
        self.transition_distribution = Categorical(probs=transition_matrix)
        self.emission_distribution = MVN(emission_means, emission_covariance_matrices)

        # Construct the priors
        self.initial_distribution_prior = Dirichlet(
            concentration=initial_probabilities_prior_concentration \
                * jnp.ones(self.num_states))
        self.transition_distribution_prior = Dirichlet(
            concentration=transition_matrix_prior_concentration \
                * jnp.ones((self.num_states, self.num_states)))

    # Properties to get various parameters of the model
    @property
    def initial_probabilities(self):
        return self.initial_distribution.probs

    @property
    def transition_matrix(self):
        return self.transition_distribution.probs

    @property
    def emission_means(self):
        return self.emission_distribution.loc

    @property
    def emission_covariance_matrices(self):
        return self.emission_distribution.covariance_matrix

    def e_step(self, emissions):
        log_lkhds = self.emission_distribution.log_prob(emissions[...,None,:])
        return hmm_smoother(self.initial_probabilities,
                            self.transition_matrix,
                            log_lkhds)

    def m_step(self, emissions, posterior):
        # Initial distribution
        self.initial_distribution = Categorical(
            probs=self.initial_distribution_prior.concentration
                  + posterior.smoothed_probs[0])

        # Transition distribution
        self.transition_distribution = Categorical(
            probs=self.transition_distribution_prior.concentration
                  + posterior.smoothed_transition_probs)

        # Gaussian emission distribution
        # TODO: Include NIW prior
        w_sum = jnp.sum(posterior.smoothed_probs, 0)
        x_sum = jnp.einsum('tk, ti->ki', posterior.smoothed_probs, emissions)
        xxT_sum = jnp.einsum('tk, ti, tj->kij', posterior.smoothed_probs, emissions, emissions)

        emission_means = x_sum / w_sum[:, None]
        emission_covs = xxT_sum / w_sum[:, None, None] \
            - jnp.einsum('ki,kj->kij', emission_means, emission_means) \
            + 1e-4 * jnp.eye(self.emission_dim)
        self.emission_distribution = MVN(emission_means, emission_covs)

    def fit(self,
            emissions,
            num_iters=100):
        """Fit a Hidden Markov Model with expectation-maximization (EM).

        Args:
            emissions (Array of shape (batch_size, num_timesteps, emission_dim)):
                The observed emissions. If no batch dimension is given, it will
                be created.
            num_iters (int, optional): Number of EM emissions. Defaults to 100.

        Returns:
            log_probs: marginal log probability of the data over EM iterations
            posteriors: HMMPosterior object with statistics for the batch of posteriors
        """
        # Add a batch dimension if not given
        assert emissions.shape[-1] == self.emission_dim and emissions.ndim in (2, 3)
        squeeze = False
        if emissions.ndim == 2:
            emissions = emissions[None, :, :]
            squeeze = True

        # Run the EM iterations
        log_probs = []
        for _ in range(num_iters):
            posteriors = vmap(lambda x: self.e_step(x))(emissions)
            log_probs.append(posteriors.marginal_log_lkhd.sum())
            self.m_step(emissions, posteriors)

        # Squeeze unwanted dimensions from the posteriors if batch size was 1.
        if squeeze:
            posteriors = tree_map(lambda x: x[0], posteriors)

        return jnp.array(log_probs), posteriors
