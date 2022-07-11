from abc import abstractproperty
from jax import jit, lax, value_and_grad, vmap, tree_map
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm.auto import trange
from functools import partial
import tensorflow_probability.substrates.jax.distributions as tfd

from learning import hmm_fit_sgd, _sample_minibatches
from models import BaseHMM, GaussianHMM
from NIW import NormalInverseWishart


class _BaseHMM(BaseHMM):
    
    def __init__(self, initial_probabilities, transition_matrix):
        super().__init__(self, initial_probabilities, transition_matrix)
        
    @abstractproperty
    def emission_prior(self):
        raise NotImplementedError
    
    def marginal_log_evidence(self, emissions):
        """log marginal evidence (log marginal likelihood of observations + log prior of emission)"""
        ll = self.marginal_log_prob(emissions)
        l_prior = self.emission_prior.log_prob(emission_params)
        return ll + l_prior
    
    def map_step(self, batch_emissions, batch_posteriors, batch_trans_probs, optimizer=optax.adam(1e-2), num_iters=50):
        """_summary_

        Args:
            batch_emissions (_type_): _description_
            batch_posteriors (_type_): _description_
            batch_trans_probs (_type_): _description_
            optimizer (_type_, optional): _description_. Defaults to optax.adam(1e-2).
            num_iters (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """
        def _single_expected_log_joint_plus_prior(hmm, emissions, posterior, trans_probs):
            log_likelihoods = vmap(hmm.emission_distribution.log_prob)(emissions)
            expected_states = posterior.smoothed_probs
            
            lp = jnp.sum(expected_states[0] * jnp.log(hmm.initial_probabilities))
            lp += jnp.sum(trans_probs * jnp.log(hmm.transition_matrix))
            lp += jnp.sum(expected_states * log_likelihoods)
            
            l_prior = vmap(hmm.emission_prior.log_prob)(emissions_params)
            return lp + l_prior
            
        def neg_expected_log_joint_plus_prior(params):
            hmm = self.from_unconstrained_params(params, hypers)
            f = vmap(partial(_single_expected_log_joint_plus_prior, hmm))
            lps = f(batch_emissions, batch_posteriors, batch_trans_probs)
            return -jnp.sum(lps / jnp.ones_like(batch_emissions).sum())
        
        hmm, losses = hmm_fit_sgd(self, batch_emissions, optimizer, num_iters, neg_expected_log_joint_plus_prior) 
        return hmm, -losses
    
    
class _GaussianHMM(GaussianHMM):
    
    def __init__(self, initial_probabilities, transition_matrix, emission_means, emission_covariance_matrices,
               prior_params=None):
        super().__init__(initial_probabilities, transition_matrix, emission_means, emission_covariance_matrices)
        
        if prior_params != None:
            self._emission_prior = NormalInverseWishart(prior_params)
        else:
            _prior_params = {}
            self._emission_prior = NormalInverseWishart(_prior_params)
        
    @property
    def emission_prior(self):
        return self._emission_prior
    
    @classmethod
    def initialization_from_prior(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_params = emission_prior.sample(seed=key3, shape=(num_states))
        emission_means = emission_params['mu']
        emission_covs = emission_params['Sig']
        return cls(initial_probs, transition_matrix, emission_means, emission_covs)
    
    @classmethod
    def map_step(cls, batch_stats):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        
        # Initial distribution
        initial_probs = tfd.Dirichlet(1.0001 + stats.initial_probs).mode()
        
        # Transition distribution
        transition_matrix = tfd.Dirichlet(1.0001 + stats.sum_trans_probs).mode()
        
        # Gaussian emission distribution
        emission_dim = stats.sum_x.shape[-1]
        emission_means = None
        emission_covs = None
        emission_prior = None
        
        # Pack the results into a new GaussianHMM
        return cls(initial_probs, transition_matrix, emission_means, emission_covs, emission_prior)
    
    
def hmm_fit_emap(hmm, batch_emissions, optimizer=optax.adam(1e-2), num_iters=50):
    
    @jit 
    def emap_step(hmm):
        batch_posteriors, batch_trans_probs = hmm.e_step(batch_emissions)
        hmm, marginal_log_evs = hmm.map_step(batch_emissions, batch_posteriors, batch_trans_probs, optimizer)
        return hmm, marginal_log_evs
    
    log_evs = []
    for _ in trange(num_iters):
        hmm, marginal_log_evs = emap_step(hmm)
        log_evs.append(marginal_log_evs[-1])
        
    return hmm, log_evs


def hmm_fit_minibatch_gradient_descent(hmm, emissions, optimizer, loss_fn=None, batch_size=1, num_iters=50, key=jr.PRNGKey(0)):
    cls = hmm.__class__
    hypers = hmm.hyperparams
    
    params = hmm.unconstrained_params
    opt_state = optimizer.init(params)
    
    num_complete_batches, leftover = jnp.divmod(len(emissions), batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)
    
    def loss(params, batch_emissions):
        hmm = cls.from_unconstrained_params(params, hypers)
        f = lambda emissions: -hmm.marginal_log_prob(emissions) / len(emissions)
        return vmap(f)(batch_emissions).mean()
    
    loss_grad_fn = jit(value_and_grad(loss))
    
    def train_step(carry, key):
        perm = jr.permutation(key, len(emissions))
        _emissions = emissions[perm]
        sample_generator = _sample_minibatches(_emissions, batch_size)
        
        def opt_step(carry, i):
            params, opt_state = carry
            batch = next(sample_generator)
            val, grads = loss_grad_fn(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), val

        state, losses = lax.scan(opt_step, carry, jnp.arange(num_batches))
        return state, losses.mean()
    
    keys = jr.split(key, num_iters)
    (params, _), losses = lax.scan(train_step, (params, opt_state), keys)
    
    losses = losses.flatten()
    hmm = cls.from_unconstrained_params(params, hypers)
    
    return hmm, losses
        