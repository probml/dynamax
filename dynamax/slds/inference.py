import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap, debug, jit, pmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from functools import wraps, partial
import inspect

from jax.tree_util import tree_map
from jaxtyping import Array, Float, Int
from typing import NamedTuple, Optional, Union, Tuple
from dynamax.utils.utils import psd_solve
from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMMParameterSet
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, PosteriorGSSMFiltered, _predict

from dynamax.types import PRNGKey, Scalar

class DiscreteParamsSLDS(NamedTuple):
    initial: Float[Array, "num_states"]
    transition_matrix : Float[Array, "num_states num_states"]
    proposal_transition_matrix : Float[Array, "num_states num_states"]

class flatParamsSLDS(NamedTuple):
    r"""Parameters of a SLDS, which has a discrete component and a continuous component.
    The discrete component is a Markov model for the hidden discrete state $m_t$.
    The linear_gaussian component is a linear Gaussian state space model for the hidden continuous state $z_t$ and
    emission $y_t$.
    """
    dynamics_weights: Float[Array, "num_states state_dim state_dim"]
    dynamics_input_weights: Float[Array, "num_states num_states input_dim"]
    dynamics_bias: Float[Array, "num_states state_dim"]
    dynamics_covariance: Float[Array, "num_states state_dim state_dim"]
    emission_weights: Float[Array, "num_states emission_dim state_dim"]
    emission_input_weights: Float[Array, "num_states num_emissions input_dim"]
    emission_bias: Float[Array, "num_states emission_dim"]
    emission_covariance: Float[Array, "num_states emission_dim emission_dim"]

class ParamsSLDS(NamedTuple):
    r"""Parameters of a SLDS, which has a discrete component and a continuous component.
    The discrete component is a Markov model for the hidden discrete state $m_t$.
    The linear_gaussian component is a linear Gaussian state space model for the hidden continuous state $z_t$ and
    emission $y_t$.
    """
    discrete: DiscreteParamsSLDS
    linear_gaussian: ParamsLGSSM[Array, "num_states"]

    def flatten(self):
        num_states = self.discrete.transition_matrix.shape[0]
        flat_dynamics_weights = jnp.array([self.linear_gaussian[x].dynamics.weights for x in range(num_states)])
        flat_dynamics_input_weights = jnp.array([self.linear_gaussian[x].dynamics.input_weights for x in range(num_states)])
        flat_dynamics_biases = jnp.array([self.linear_gaussian[x].dynamics.bias for x in range(num_states)])
        flat_dynamics_covariances = jnp.array([self.linear_gaussian[x].dynamics.cov for x in range(num_states)])
        flat_emission_weights = jnp.array([self.linear_gaussian[x].emissions.weights for x in range(num_states)])
        flat_emission_input_weights = jnp.array([self.linear_gaussian[x].emissions.input_weights for x in range(num_states)])
        flat_emission_biases = jnp.array([self.linear_gaussian[x].emissions.bias for x in range(num_states)])
        flat_emission_covariances = jnp.array([self.linear_gaussian[x].emissions.cov for x in range(num_states)])
        flat_params = flatParamsSLDS(
            dynamics_weights = flat_dynamics_weights,
            dynamics_input_weights = flat_dynamics_input_weights,
            dynamics_bias = flat_dynamics_biases,
            dynamics_covariance = flat_dynamics_covariances,
            emission_weights = flat_emission_weights,
            emission_input_weights = flat_emission_input_weights,
            emission_bias = flat_emission_biases,
            emission_covariance = flat_emission_covariances
        )
        return flat_params

class RBPFiltered(NamedTuple):
    r"""RBPF posterior.
    :param weights: weights of the particles.
    :param means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    :param states: array of sampled discrete state sequences (particles) $$.
    """
    weights: Optional[Float[Array, "num_particles ntime"]] = None
    states: Optional[Int[Array, "num_particles ntime num_states"]] = None
    means: Optional[Float[Array, "num_particles ntime state_dim"]] = None
    covariances: Optional[Float[Array, "num_particles ntime state_dim state_dim"]] = None
   

def _resample(weights, new_states, means, covariances, key):                                                                  
    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    new_states = jnp.take(new_states, resampled_idx, axis=0)
    filtered_means = jnp.take(means, resampled_idx, axis=0)
    filtered_covs = jnp.take(covariances, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]
    return weights, new_states, filtered_means, filtered_covs, next_key    

def _kalman_step(mu, Sigma, params, u, y):
    r"""
    Perform a Kalman step, given a prior and a linear Gaussian observation model.
    """
    F = params.dynamics_weights
    b = params.dynamics_bias
    B = params.dynamics_input_weights
    Q = params.dynamics_covariance
    H = params.emission_weights
    d = params.emission_bias
    D = params.emission_input_weights
    R = params.emission_covariance

    # prediction
    mu_pred = F @ mu + B @ u + b
    Sigma_pred = F @ Sigma @ F.T + Q

    # update
    S = R + H @ Sigma_pred @ H.T
    K = psd_solve(S, H @ Sigma_pred).T
    mu_y = H @ mu_pred + D @ u + d
    ll = MVN(loc = mu_y, covariance_matrix = S).log_prob(y)
    mu_cond = mu_pred + K @ (y - mu_y)
    Sigma_cond = Sigma_pred - K @ S @ K.T
    return ll, mu_cond, Sigma_cond


def _conditional_kalman_step(state, mu, Sigma, params, u, y):
    """
    Perform a Kalman step, given a prior and a linear Gaussian observation model.
    """
    F = params.dynamics_weights[state]
    b = params.dynamics_bias[state]
    B = params.dynamics_input_weights[state]
    Q = params.dynamics_covariance[state]
    H = params.emission_weights[state]
    d = params.emission_bias[state]
    D = params.emission_input_weights[state]
    R = params.emission_covariance[state]

    # prediction
    mu_pred = F @ mu + B @ u + b
    Sigma_pred = F @ Sigma @ F.T + Q

    # update
    S = R + H @ Sigma_pred @ H.T
    K = psd_solve(S, H @ Sigma_pred).T
    mu_y = H @ mu_pred + D @ u + d
    ll = MVN(loc = mu_y, covariance_matrix = S).log_prob(y)
    mu_cond = mu_pred + K @ (y - mu_y)
    Sigma_cond = Sigma_pred - K @ S @ K.T
    return ll, mu_cond, Sigma_cond

def rbpfilter(
    num_particles: int,
    params: ParamsSLDS,
    emissions:  Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    key: PRNGKey = jr.PRNGKey(0),
    ess_threshold: float = 0.5
) -> RBPFiltered:
    '''
    Implementation of the Rao-Blackwellized particle filter, for approximating the 
    filtering distribution of a switching linear dynamical system. The filter at each iteration
    samples discrete states from a discrete proposal, and then runs a KF step conditional on the sampled
    value of the chain. At the end of the update it computes an effective sample size and decide whether
    resampling is necessary.
    '''

    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs
    flat_params = params.flatten()

    def _step(carry, t):
        r"""
        carry = (weights, prev_states, filtered_means, filtered_covs, key)

        where,
            prev_states : (num_particles, )
            weights : (num_particles, )
            pred_means : (num_particles, state_dim)
            pred_covs : (num_particles, state_dim, state_dim)
        """

        # Unpack carry
        weights, prev_states, filtered_means, filtered_covs, key = carry
        num_states = params.discrete.transition_matrix.shape[0]

        # Get emissions and inputs for time index t
        u = inputs[t]
        y = emissions[t]

        # Sample discrete states from the proposal
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        new_states = vmap(lambda key, x: jr.choice(key, jnp.arange(num_states), p=params.discrete.proposal_transition_matrix[x]))(keys[1:], prev_states)

        # Run KF step conditional on the sampled states
        lls, filtered_means, filtered_covs = vmap(_conditional_kalman_step, in_axes = (0, 0, 0, None, None, None))(new_states, filtered_means, filtered_covs, flat_params, u, y)

        # Compute weights
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        weights = jnp.multiply(loglik_weights.T, weights)
        weights /= jnp.sum(weights)

        # Resample if necessary
        resample_cond = 1.0 / jnp.sum(jnp.square(weights)) < ess_threshold * num_particles
        weights, new_states, filtered_means, filtered_covs, next_key = lax.cond(resample_cond, _resample, lambda *args: args,
                                                                                weights, new_states, filtered_means, filtered_covs, next_key)

        # Build carry and output states
        carry = (weights, prev_states, filtered_means, filtered_covs, next_key)
        outputs = {
            "weights": weights,
            "states": prev_states,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        return carry, outputs

    key1, key2, next_key = jr.split(key, 3)

    # Initialize carry
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    initial_states = jr.choice(key1, jnp.arange(params.discrete.initial.shape[0]), shape=(num_particles,), p = params.discrete.initial)
    initial_means = jnp.array([MVN(params.linear_gaussian[state].initial.mean, params.linear_gaussian[state].initial.cov).sample(seed=key2) for state in initial_states])
    initial_covs = jnp.array([params.linear_gaussian[state].initial.cov for state in initial_states])
    
    carry = (
        initial_weights,
        initial_states, 
        initial_means,
        initial_covs, 
        next_key)
    
    _, out = lax.scan(_step, carry, jnp.arange(num_timesteps))

    return out


def rbpfilter_optimal(
    num_particles: int,
    params: ParamsSLDS,
    emissions:  Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None,
    ess_threshold: float = 0.5,
    key: PRNGKey = jr.PRNGKey(0)
) -> RBPFiltered:
    '''
    Implementation of the Rao-Blackwellized particle filter, for approximating the 
    filtering distribution of a switching linear dynamical system. The filter at each iteration
    samples discrete states from a discrete proposal, and then runs a KF step conditional on the sampled
    value of the chain. At the end of the update it computes an effective sample size and decide whether
    resampling is necessary.
    '''

    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs
    flat_params = params.flatten()

    def _step(carry, t):
        # prev_states : (num_particles, )
        # weights : (num_particles, )
        # pred_means : (num_particles, state_dim)
        # pred_covs : (num_particles, state_dim, state_dim)

        # Unpack carry
        weights, prev_states, filtered_means, filtered_covs, key = carry
        num_states = params.discrete.transition_matrix.shape[0]

        # Get emissions and inputs for time index t
        u = inputs[t]
        y = emissions[t]

        # Sample discrete states from the proposal
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        new_states = vmap(lambda key, x: jr.choice(key, jnp.arange(num_states), p=params.discrete.proposal_transition_matrix[x]))(keys[1:], prev_states)

        # Run KF step conditional on the sampled states
        lls, filtered_means, filtered_covs = vmap(_conditional_kalman_step, in_axes = (0, 0, 0, None, None, None))(new_states, filtered_means, filtered_covs, flat_params, u, y)

        # _vec_kalman_step = lambda mu, Sigma: vmap(_kalman_step, in_axes=(None, None, 0, None, None))(mu, Sigma, flat_params, u, y)
        # lls, filtered_means, filtered_covs = vmap(_vec_kalman_step, in_axes=(0, 0))(filtered_means, filtered_covs)

        print('means shape', filtered_means.shape)
        print('covs shape', filtered_covs.shape)
        print('new_states shape', len(new_states))


        # Compute weights
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        weights = jnp.multiply(loglik_weights.T, weights)
        weights /= jnp.sum(weights)

        # Resample if necessary
        resample_pred = 1.0 / jnp.sum(jnp.square(weights)) < ess_threshold * num_particles
        weights, new_states, filtered_means, filtered_covs, next_key = lax.cond(resample_pred, 
                                                                                _resample, 
                                                                                lambda *args: args, 
                                                                                weights, new_states, filtered_means, filtered_covs, next_key)

        # Optimal resampling
#        weights, new_states, filtered_means, filtered_covs, next_key  = _resample(weights.reshape(num_states*num_particles,), filtered_means, filtered_covs, next_key)
                                                                                 
        # Build carry and output states
        carry = (weights, prev_states, filtered_means, filtered_covs, next_key)
        outputs = {
            "weights": weights,
            "states": prev_states,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        return carry, outputs

    key1, key2, next_key = jr.split(key, 3)

    # Initialize carry
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    initial_states = jr.choice(key1, jnp.arange(params.discrete.initial.shape[0]), shape=(num_particles,), p = params.discrete.initial)
    initial_means = jnp.array([MVN(params.linear_gaussian[state].initial.mean, params.linear_gaussian[state].initial.cov).sample(seed=key2) for state in initial_states])
    initial_covs = jnp.array([params.linear_gaussian[state].initial.cov for state in initial_states])
    
    carry = (
        initial_weights,
        initial_states, 
        initial_means,
        initial_covs, 
        next_key)
    
#    _, out = lax.scan(_step, carry, jnp.arange(num_timesteps))
    out = _step(carry, 0)[1]

    return out
