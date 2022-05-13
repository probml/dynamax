import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_map

from abc import ABC, abstractmethod

from .core import hmm_smoother

# from distrax import (
#     MultivariateNormalFullCovariance as MVN,
#     Categorical,
#     Dirichlet,
#     Gamma,
#     Poisson)
from tensorflow_probability.substrates.jax.distributions import (
    Bernoulli,
    Beta,
    Categorical,
    Dirichlet,
    Gamma,
    MultivariateNormalFullCovariance as MVN,
    Poisson)


class BaseHMM(ABC):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 initial_probs_concentration=1.0001,
                 transition_matrix_concentration=1.0001):
        """Abstract base class for Hidden Markov Models.

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            initial_probs_concentration (float, optional): _description_.
                Defaults to 1.0001.
            transition_matrix_concentration (float, optional): _description_.
                Defaults to 1.0001.
        """
        self.num_states = transition_matrix.shape[-1]

        # Check shapes
        assert initial_probabilities.shape == (self.num_states,)
        assert transition_matrix.shape == (self.num_states, self.num_states)

        # Construct the model from distrax distributions
        self.initial_distribution = Categorical(probs=initial_probabilities)
        self.transition_distribution = Categorical(probs=transition_matrix)

        # Construct the priors
        self.initial_distribution_prior = Dirichlet(
            concentration=initial_probs_concentration \
                * jnp.ones(self.num_states))
        self.transition_distribution_prior = Dirichlet(
            concentration=transition_matrix_concentration \
                * jnp.ones((self.num_states, self.num_states)))

    # Properties to get various parameters of the model
    @property
    def initial_probabilities(self):
        return self.initial_distribution.probs

    @property
    def transition_matrix(self):
        return self.transition_distribution.probs

    @abstractmethod
    def log_likelihoods(self, emissions):
        """Compute the log likelihood of the emissions under each discrete state.

        Args:
            emissions (Array, (..., num_timesteps, emission_dim)): observed emissions.

        Returns:
            log_likes (Array, (..., num_timesteps, num_states)): log likelihood of the
                emissions under each discrete latent state.
        """
        raise NotImplemented

    def e_step(self, emissions):
        return hmm_smoother(self.initial_probabilities,
                            self.transition_matrix,
                            self.log_likelihoods(emissions))

    @abstractmethod
    def m_step_emission_distribution(self, emissions, posterior):
        """Perform an M-step on the parameters of the emission distribution.

        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """
        raise NotImplemented

    def m_step(self, emissions, posterior):
        # Initial distribution
        initial_probs = Dirichlet(self.initial_distribution_prior.concentration
                                  + jnp.einsum('bk->k', posterior.smoothed_probs[:,0,:])).mode()
        self.initial_distribution = Categorical(probs=initial_probs)

        # Transition distribution
        transition_matrix = Dirichlet(self.transition_distribution_prior.concentration
                                      + jnp.einsum('bij->ij', posterior.smoothed_transition_probs)).mode()
        self.transition_distribution = Categorical(probs=transition_matrix)

        # Emission distribution
        self.m_step_emission_distribution(emissions, posterior)

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


class BernoulliHMM(BaseHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_probs,
                 initial_probs_concentration=1.0001,
                 transition_matrix_concentration=1.0001,
                 emission_probs_concentration1=1.0001,
                 emission_probs_concentration0=1.0001):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_probs.shape[-1]

        # Check shapes
        assert emission_probs.shape == \
            (self.num_states, self.emission_dim)

        # Construct the model from distrax distributions
        self.emission_distribution = Bernoulli(emission_probs)
        self.emission_prior = Beta(emission_probs_concentration1,
                                   emission_probs_concentration0)

    # Properties to get various parameters of the model
    @property
    def emission_probs(self):
        return self.emission_distribution.probs_parameter()

    def log_likelihoods(self, emissions):
        # Sum over the last dimension since we're assuming a batch of
        # conditionally independent Bernoullis
        return self.emission_distribution.log_prob(emissions[...,None,:]).sum(-1)

    def m_step_emission_distribution(self, emissions, posterior):
        x1_sum = jnp.einsum('btk, bti->ki', posterior.smoothed_probs, emissions)
        x0_sum = jnp.einsum('btk, bti->ki', posterior.smoothed_probs, 1 - emissions)

        emission_probs = Beta(self.emission_prior.concentration1 + x1_sum,
                              self.emission_prior.concentration0 + x0_sum).mode()
        self.emission_distribution = Bernoulli(emission_probs)


class CategoricalHMM(BaseHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_probs,
                 initial_probs_concentration=1.0001,
                 transition_matrix_concentration=1.0001,
                 emission_probs_concentration=1.0001):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_probs.shape[-1]

        # Check shapes
        assert emission_probs.shape == (self.num_states, self.emission_dim)

        # Construct the model from distrax distributions
        self.emission_distribution = Categorical(probs=emission_probs)
        self.emission_prior = Dirichlet(emission_probs_concentration * jnp.ones(self.emission_dim))

    # Properties to get various parameters of the model
    @property
    def emission_probs(self):
        return self.emission_distribution.probs_parameter()

    def log_likelihoods(self, emissions):
        return self.emission_distribution.log_prob(emissions[...,None,:])

    def m_step_emission_distribution(self, emissions, posterior):
        x_sum = jnp.einsum('btk, bti->ki', posterior.smoothed_probs, emissions)
        emission_probs = Dirichlet(self.emission_prior.concentration + x_sum).mode
        self.emission_distribution = Categorical(probs=emission_probs)


class GaussianHMM(BaseHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_covariance_matrices,
                 initial_probs_concentration=1.0001,
                 transition_matrix_concentration=1.0001):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_means.shape[-1]

        # Check shapes
        assert emission_means.shape == \
            (self.num_states, self.emission_dim)
        assert emission_covariance_matrices.shape == \
            (self.num_states, self.emission_dim, self.emission_dim)

        # Construct the model from distrax distributions
        self.emission_distribution = MVN(emission_means, emission_covariance_matrices)

    # Properties to get various parameters of the model
    @property
    def emission_means(self):
        return self.emission_distribution.loc

    @property
    def emission_covariance_matrices(self):
        return self.emission_distribution.covariance_matrix

    def log_likelihoods(self, emissions):
        return self.emission_distribution.log_prob(emissions[...,None,:])

    def m_step_emission_distribution(self, emissions, posterior):
        # Gaussian emission distribution
        # TODO: Include NIW prior
        w_sum = jnp.sum('btk->k', posterior.smoothed_probs)
        x_sum = jnp.einsum('btk, bti->ki', posterior.smoothed_probs, emissions)
        xxT_sum = jnp.einsum('btk, bti, btj->kij', posterior.smoothed_probs, emissions, emissions)

        emission_means = x_sum / w_sum[:, None]
        emission_covariance_matrices = xxT_sum / w_sum[:, None, None] \
            - jnp.einsum('ki,kj->kij', emission_means, emission_means) \
            + 1e-4 * jnp.eye(self.emission_dim)
        self.emission_distribution = MVN(emission_means, emission_covariance_matrices)


class PoissonHMM(BaseHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_rates,
                 initial_probs_concentration=1.0001,
                 transition_matrix_concentration=1.0001,
                 emission_rates_concentration=1.0001,
                 emission_rates_rate=0.0001):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_rates.shape[-1]

        # Check shapes
        assert emission_rates.shape == \
            (self.num_states, self.emission_dim)

        # Construct the model from distrax distributions
        self.emission_distribution = Poisson(emission_rates)
        self.emission_prior = Gamma(emission_rates_concentration,
                                    emission_rates_rate)

    # Properties to get various parameters of the model
    @property
    def emission_rates(self):
        return self.emission_distribution.rate

    def log_likelihoods(self, emissions):
        # Sum over the last dimension since we're assuming a batch of
        # conditionally independent Poissons
        return self.emission_distribution.log_prob(emissions[...,None,:]).sum(-1)

    def m_step_emission_distribution(self, emissions, posterior):
        # Gaussian emission distribution
        # TODO: Include NIW prior
        w_sum = jnp.sum('btk->k', posterior.smoothed_probs)
        x_sum = jnp.einsum('btk, bti->ki', posterior.smoothed_probs, emissions)

        emission_rates = Gamma(self.emission_prior.concentration + x_sum,
                               self.emission_prior.rate + w_sum[:, None]).mode()
        self.emission_distribution = Poisson(emission_rates)
