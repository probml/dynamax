from abc import abstractproperty
from jax import vmap, tree_map
import jax.numpy as jnp
import optax
from functools import partial
import tensorflow_probability.substrates.jax.distributions as tfd

from ssm_jax.hmm.learning import hmm_fit_sgd
from ssm_jax.hmm.models.base import BaseHMM
from ssm_jax.hmm.models.gaussian_hmm import GaussianHMM
from ssm_jax.utils_distributions import NormalInverseWishart as NIW, MatrixNormalInverseWishart as MNIW

class _BaseHMM(BaseHMM):
    
    def __init__(self, initial_probabilities, transition_matrix, 
                 prior_dirichlet_concentrations=None):
        super().__init__(initial_probabilities, transition_matrix)
        
        if prior_dirichlet_concentrations is None:
            self._init_dirichlet_concentration = 1e-4 * jnp.ones(self.num_states)
            self._transit_dirichlet_concentration = 1e-4 * jnp.ones(self.num_states)
        else:
            self._init_dirichlet_concentration = prior_dirichlet_concentrations[0]
            self._transit_dirichlet_concentration = prior_dirichlet_concentrations[1]

        self._initial_prior = tfd.Dirichlet(self._init_dirichlet_concentration)
        self._transition_prior = tfd.Dirichlet(self._transit_dirichlet_concentration)
        
        @property
        def initial_prior(self):
            return self._initial_prior
        
        @property
        def transition_prior(self):
            return self._transition_prior
        
        @property
        def emission_prior(self):
            return NotImplementedError
        
        @property
        def emission_params(self):
            # return the parameter of the emission distribution for all clusters
            return NotImplementedError
        
    def m_step(self, batch_emissions, batch_posteriors, batch_trans_probs, 
               optimizer=optax.adam(1e-2), num_iters=50):
        
        hypers = self.hyperparams
        
        def _single_expected_log_joint(hmm, emissions, posterior, trans_probs):
            log_likelihoods = vmap(hmm.emission_distribution.log_prob)(emissions)
            expected_states = posterior.smoothed_probs
            
            lp = jnp.sum(expected_states[0] * jnp.log(hmm.initial_probabilities))
            lp += jnp.sum(trans_probs * jnp.log(hmm.transition_matrix))
            lp += jnp.sum(expected_states * log_likelihoods)
            
            initial_params = hmm.initial_probabilities
            transition_params = hmm.transition_matrix
            emissions_params = hmm.emissions_params
            l_prior += hmm.initial_prior.log_prob(initial_params)
            l_prior += vmap(hmm.transition_prior.log_prob)(transition_params).sum()
            l_prior = vmap(hmm.emission_prior.log_prob)(emissions_params).sum()
            
            return lp + l_prior         
            
        def neg_expected_log_joint(params):
            hmm = self.from_unconstrained_params(params, hypers)
            f = vmap(partial(_single_expected_log_joint, hmm))
            lps = f(batch_emissions, batch_posteriors, batch_trans_probs)
            return -jnp.sum(lps / jnp.ones_like(batch_emissions).sum())
        
        hmm, losses = hmm_fit_sgd(self, batch_emissions, optimizer, num_iters, neg_expected_log_joint)
         
        return hmm
    
    
class _GaussianHMM(GaussianHMM):
    
    def __init__(self, initial_probabilities, transition_matrix, 
                 prior_dirichlet_concentration,
                 emission_means, emission_covariance_matrices,
                 emission_prior_hyperparams):
        
        super().__init__(initial_probabilities, transition_matrix, 
                         emission_means, emission_covariance_matrices)
        
        self.emission_dim = emission_means.shape[1]
        
        if emission_prior_hyperparams is None:
            niw_params = (jnp.zeros(self.emission_dim),
                          1.,
                          self.emission_dim,
                          1e4 * jnp.eye(self.emission_dim))
            emission_prior_hyperparams = niw_params
            
        self._initial_dirichlet_concent = initial_dirichlet_concentration
        self._transition_dirichlet_concent = transition_dirichlet_concentration
        self._emission_covariance_scale = emission_covariance_scale
        self._emission_covariance_df = emission_covariance_df
        self._emission_means_mean = emission_means_mean
        self._emission_means_cov = emission_means_cov
        self._emission_precision = emission_precision
        
        self._initial_prior = tfd.Dirichlet(initial_dirichlet_concentration)
        self._transition_prior = tfd.Dirichlet(transition_dirichlet_concentration)
        if emission_prior_type is 'Joint':
            self._emission_prior = NIW(emission_means_mean,
                                       emission_precision,
                                       emission_covariance_df,
                                       emission_covariance_scale)
        elif emission_prior_type is 'Mean':
            self._emission_prior = tfd.MultivariateNormalFullCovariance(emission_means_mean,
                                                                        emission_means_cov)
        elif emission_prior_type is 'Covariance':
            self._emission_prior = IW(emission_covariance_df,
                                      emission_covariance_scale)
        
        @property
        def initial_prior(self):
            # not needed if implemented in _BaseHMM 
            return self._initial_prior
        
        @property
        def transition_prior(self):
            return self._transition_prior
        
        @property
        def emission_prior(self):
            return self._emission_prior
        
        @property
        def emission_params(self):
            return zip(emission_means, emission_covariance_matrices)
    
    def m_step(self, batch_stats, MAP=True):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        
        if MAP:
            # Initial distribution
            initial_probs = tfd.Dirichlet(1. + self._initial_dirichlet_concent + stats.initial_probs).mode()
            # Transition distribution
            transition_matrix = tfd.Dirichlet(1. + self._transition_dirichlet_concent + stats.sum_trans_probs).mode()
            
            # Gaussian emission distribution
            if self._emission_prior_type is 'Joint':
                emission_means = (stats.sum_x + self._emission_precision*self._emission_means_mean) \
                                 / (stats.sum_w[:, None] + self._emission_precision)
                covs = self._emission_covariance_scale \
                    + stats.sum_xxT - jnp.einsum('ki,kj->kij', emission_means, emission_means)*stats.sum_w[:, None, None] \
                    + jnp.einsum('ki,kj->kij', emission_means-self._emission_means_mean, emission_means-self._emission_means_mean) \
                        *(self._emission_precision*stats.sum_w/(self._emission_precision+stats.sum_w))[:, None, None]
                emission_covs = covs / (self._emission_covariance_df + stats.sum_w + self.emission_dim + 2)[:, None, None]
                
            elif self._emission_prior_type is 'Mean':
                emission_covs = stats.sum_xxT / stats.sum_w[:, None, None] \
                            - jnp.einsum('ki,kj->kij', emission_means, emission_means)
                weighted_means = jnp.linalg.solve(self._emission_covariance_scale, self._emission_means_mean) \
                    + vmap(jnp.linalg.solve)(emission_covs, stats.sum_x/stats.sum_w[:, None])
                precisions = jnp.linalg.solve(self._emission_covariance_scale, jnp.eye(self.emission_dim)) \
                    + vmap(jnp.linalg.solve, in_axes=(0,None))(emission_covs, jnp.eye(self.emission_dim))
                emission_means = vmap(jnp.linalg.solve)(precisions, weighted_means)
                
            elif self._emission_prior_type is 'Covariance':
                emission_means = stats.sum_x / stats.sum_w[:, None]
                covs = self._emission_covariance_scale + \
                    stats.sum_xxT - jnp.einsum('ki,kj->kij', emission_means, emission_means)*stats.sum_w[:, None, None]
                emission_covs = covs /(self._emission_covariance_df + stats.sum_w + self.emission_dim + 1)
                
        else:
            # Initial distribution
            initial_probs = tfd.Dirichlet(1. + stats.initial_probs).mode()
            # Transition distribution
            transition_matrix = tfd.Dirichlet(1. + stats.sum_trans_probs).mode()
            
            # Gaussian emission distribution
            emission_means = stats.sum_x / stats.sum_w[:, None]
            emission_covs = stats.sum_xxT / stats.sum_w[:, None, None] \
                            - jnp.einsum('ki,kj->kij', emission_means, emission_means) 
        
        # Pack the results into a new GaussianHMM
        hmm =  _GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs,
                            self._initial_dirichlet_concent, self._transition_dirichlet_concent,
                            self._emission_prior_type,
                            self._emission_means_mean, self._emission_means_cov,
                            self._emission_covariance_df, self._emission_covariance_scale,
                            self._emission_precision)
        return hmm, 