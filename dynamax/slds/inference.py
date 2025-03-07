"""
Rao-Blackwellized Particle Filter for inference in a 
Switching Linear Dynamical Systems (SLDS).
"""
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import lax, vmap, jit
from jaxtyping import Array, Float, Int
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import NamedTuple, Optional

from dynamax.utils.utils import psd_solve
from dynamax.types import PRNGKeyT

class DiscreteParamsSLDS(NamedTuple):
    """
    Parameters of a discrete state dynamics for an SLDS.
    """
    initial_distribution: Float[Array, " num_states"]
    transition_matrix : Float[Array, "num_states num_states"]
    proposal_transition_matrix : Float[Array, "num_states num_states"]

class LGParamsSLDS(NamedTuple):
    r"""Parameters of a SLDS, which has a discrete component and a continuous component.
    The discrete component is a Markov model for the hidden discrete state $m_t$.
    The linear_gaussian component is a linear Gaussian state space model for the hidden continuous state $z_t$ and
    emission $y_t$.
    """
    initial_mean: Float[Array, "num_states state_dim"]
    initial_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_weights: Float[Array, "num_states state_dim state_dim"]
    dynamics_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_bias: Float[Array, "num_states state_dim"]
    dynamics_input_weights: Float[Array, "num_states num_states input_dim"]
    emission_weights: Float[Array, "num_states emission_dim state_dim"]
    emission_cov: Float[Array, "num_states emission_dim emission_dim"]
    emission_bias: Float[Array, "num_states emission_dim"]
    emission_input_weights: Float[Array, "num_states num_emissions input_dim"]
    initialized: bool = False


class ParamsSLDS(NamedTuple):
    r"""Parameters of a SLDS, which has a discrete component and a continuous component.
    The discrete component is a Markov model for the hidden discrete state $m_t$.
    The linear_gaussian component is a linear Gaussian state space model for the hidden continuous state $z_t$ and
    emission $y_t$.
    """
    discrete: DiscreteParamsSLDS
    linear_gaussian: LGParamsSLDS

    def initialize(self, num_states, state_dim, emission_dim, input_dim=1):
        """
        Initialize the parameters of the SLDS. For each parameter that is the same over all models, the parameter
        is repeated to match the number of states. Also optional paramters are set to zero if not specified.
        """
        params = self.linear_gaussian
        initial_mean = params.initial_mean if params.initial_mean.shape == (num_states, state_dim) else jnp.array([params.initial_mean]*num_states)
        initial_cov = params.initial_cov if params.initial_cov.shape == (num_states, state_dim, state_dim) else jnp.array([params.initial_cov]*num_states)
        dynamics_weights = params.dynamics_weights if params.dynamics_weights.shape == (num_states, state_dim, state_dim) else jnp.array([params.dynamics_weights]*num_states)
        dynamics_cov = params.dynamics_cov if params.dynamics_cov.shape == (num_states, state_dim, state_dim) else jnp.array([params.dynamics_cov]*num_states)
        dynamics_bias = params.dynamics_bias if params.dynamics_bias is not None else jnp.zeros((num_states, state_dim))
        dynamics_bias = dynamics_bias if dynamics_bias.shape == (num_states, state_dim) else jnp.array([dynamics_bias]*num_states)
        dynamics_input_weights = params.dynamics_input_weights if params.dynamics_input_weights is not None else jnp.zeros((num_states, state_dim, input_dim))
        dynamics_input_weights = dynamics_input_weights if dynamics_input_weights.shape == (num_states, state_dim, input_dim) else jnp.array([dynamics_input_weights]*num_states)
        emission_weights = params.emission_weights if params.emission_weights.shape == (num_states, emission_dim, state_dim) else jnp.array([params.emission_weights]*num_states)
        emission_cov = params.emission_cov if params.emission_cov.shape == (num_states, emission_dim, emission_dim) else jnp.array([params.emission_cov]*num_states)
        emission_bias = params.emission_bias if params.emission_bias is not None else jnp.zeros((num_states, emission_dim))
        emission_bias = emission_bias if emission_bias.shape == (num_states, emission_dim) else jnp.array([emission_bias]*num_states)
        emission_input_weights = params.emission_input_weights if params.emission_input_weights is not None else jnp.zeros((num_states, emission_dim, input_dim))
        emission_input_weights = emission_input_weights if emission_input_weights.shape == (num_states, emission_dim, input_dim) else jnp.array([emission_input_weights]*num_states)
        
        return ParamsSLDS(
            discrete=self.discrete,
            linear_gaussian=LGParamsSLDS(
                initial_mean=initial_mean,
                initial_cov=initial_cov,
                dynamics_weights=dynamics_weights,
                dynamics_cov=dynamics_cov,
                dynamics_bias=dynamics_bias,
                dynamics_input_weights=dynamics_input_weights,
                emission_weights=emission_weights,
                emission_cov=emission_cov,
                emission_bias=emission_bias,
                emission_input_weights=emission_input_weights,
                initialized=True
            )
        )


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
   

def resampling(weights, states, means, covariances, key):
    """Resample particles using the multinomial resampling algorithm."""
    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    new_states = jnp.take(states, resampled_idx, axis=0)
    filtered_means = jnp.take(means, resampled_idx, axis=0)
    filtered_covs = jnp.take(covariances, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]
    return weights, new_states, filtered_means, filtered_covs, next_key 

@partial(jit, static_argnums=(1,))
def optimal_resampling(weights, N, key):
    """Find the threshold for resampling particles using the optimal resampling algorithm of Fearnhead and Clifford (2003).
    Returns inidices of resampled particles and their weights.
    """
    # sort weights
    M = weights.shape[0]
    sorted_weights = jnp.sort(weights)
    sorted_idx = jnp.argsort(weights)

    # compute threshold p and value L of particles to retain
    lower_diag = jnp.triu(jnp.ones((M, M)), k=0).T
    ps = lax.dynamic_slice(lower_diag, (M-N, 0), (N-1, M)) @ sorted_weights / jnp.arange(1,N)
    ps = jnp.flip(ps)
    bounds = vmap(lambda ind: (sorted_weights[M-ind-1], sorted_weights[M-ind]))(jnp.arange(1, N))
    preds = vmap(lambda y,x,z : jnp.logical_and(y < x, x < z))(bounds[0], ps, bounds[1])
    L = jnp.where(preds, jnp.arange(1, N), 0).sum()
    p = jnp.where(L==0, 1/N, ps[L-1])
    
    # resample
    res_weights = jnp.where(sorted_weights < p, sorted_weights, 0.0)
    res_weights = res_weights / res_weights.sum()
    res_idx = jr.choice(key, M, shape=(M,), replace=True, p=res_weights)
    unsort_res_idx = sorted_idx[res_idx]

    final_idx = jnp.where(sorted_weights < p, unsort_res_idx, sorted_idx)
    final_weights = jnp.where(sorted_weights < p, p, sorted_weights)

    return final_idx[M-N:], final_weights[M-N:] / final_weights[M-N:].sum() 

def _conditional_kalman_step(state, mu, Sigma, params, u, y):
    """
    Perform a Kalman step, given a prior and a linear Gaussian observation model.
    """
    F = params.dynamics_weights[state]
    b = params.dynamics_bias[state]
    B = params.dynamics_input_weights[state]
    Q = params.dynamics_cov[state]
    H = params.emission_weights[state]
    d = params.emission_bias[state]
    D = params.emission_input_weights[state]
    R = params.emission_cov[state]

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
    key: PRNGKeyT = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ess_threshold: float = 0.5
    ):
    """
    Implementation of the Rao-Blackwellized particle filter, for approximating the 
    filtering distribution of a switching linear dynamical system. The filter at each iteration
    samples discrete states from a discrete proposal, and then runs a KF step conditional on the sampled
    value of the chain. At the end of the update it computes an effective sample size and decide whether
    resampling is necessary.
    """

    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 1)) if inputs is None else inputs
    if not params.linear_gaussian.initialized: raise ValueError("ParamsSLDS must be initialized")

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
        lls, filtered_means, filtered_covs = vmap(_conditional_kalman_step, in_axes = (0, 0, 0, None, None, None))(new_states, filtered_means, filtered_covs, params.linear_gaussian, u, y)

        # Compute weights
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        weights = jnp.multiply(loglik_weights.T, weights)
        weights /= jnp.sum(weights)

        # Resample if necessary
        resample_cond = 1.0 / jnp.sum(jnp.square(weights)) < ess_threshold * num_particles
        weights, new_states, filtered_means, filtered_covs, next_key = lax.cond(resample_cond, resampling, lambda *args: args,
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

    keys = jr.split(key, num_particles+2)
    next_key = keys[-1]

    # Initialize carry
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    initial_states = jr.choice(keys[0], jnp.arange(params.discrete.initial_distribution.shape[0]), shape=(num_particles,), p = params.discrete.initial_distribution)
    initial_means = jnp.array([MVN(params.linear_gaussian.initial_mean[initial_states[i]], params.linear_gaussian.initial_cov[initial_states[i]]).sample(seed=keys[i+1]) for i in range(num_particles)])
    initial_covs = jnp.array([params.linear_gaussian.initial_cov[state] for state in initial_states])
    
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
    key: PRNGKeyT = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ):
    """
    Implementation of the Rao-Blackwellized particle filter with optimal resampling
    """

    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 1)) if inputs is None else inputs
    state_dim = params.linear_gaussian.initial_mean.shape[1]

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

        key, next_key = jr.split(key, 2)

        # Get emissions and inputs for time index t
        u = inputs[t]
        y = emissions[t]

        # Run KF step conditional on all possible states
        _vec_kalman_step = lambda mu, Sigma: vmap(_conditional_kalman_step, in_axes=(0, None, None, None, None, None))(jnp.arange(num_states), mu, Sigma ,params.linear_gaussian, u, y)
        lls, filtered_means, filtered_covs = vmap(_vec_kalman_step, in_axes=(0, 0))(filtered_means, filtered_covs)

        # Compute weights
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        trans_weights = params.discrete.transition_matrix[prev_states, :]
        weights = jnp.multiply(jnp.einsum('i,ij->ij', weights, trans_weights), loglik_weights)
        weights /= jnp.sum(weights)

        # Reshape
        weights = weights.reshape((num_particles * num_states,))
        states = jnp.tile(jnp.arange(num_states), num_particles)
        filtered_means = filtered_means.reshape((num_particles * num_states, state_dim))
        filtered_covs = filtered_covs.reshape((num_particles * num_states, state_dim, state_dim))

        # Optimal resampling
        res_idx, res_weights = optimal_resampling(weights, num_particles, key)
        new_states = jnp.take(states, res_idx, axis=0)
        filtered_means = jnp.take(filtered_means, res_idx, axis=0)
        filtered_covs = jnp.take(filtered_covs, res_idx, axis=0)
                                                                                 
        # Build carry and output states
        carry = (res_weights, new_states, filtered_means, filtered_covs, next_key)
        outputs = {
            "weights": res_weights,
            "states": new_states,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        return carry, outputs

    keys = jr.split(key, num_particles+2)
    next_key = keys[-1]

    # Initialize carry
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    initial_states = jr.choice(keys[0], jnp.arange(params.discrete.initial_distribution.shape[0]), shape=(num_particles,), p = params.discrete.initial_distribution)
    initial_means = jnp.array([MVN(params.linear_gaussian.initial_mean[initial_states[i]], params.linear_gaussian.initial_cov[initial_states[i]]).sample(seed=keys[i+1]) for i in range(num_particles)])
    initial_covs = jnp.array([params.linear_gaussian.initial_cov[state] for state in initial_states])
    
    carry = (
        initial_weights,
        initial_states, 
        initial_means,
        initial_covs, 
        next_key)
    
    _, out = lax.scan(_step, carry, jnp.arange(num_timesteps))

    return out
