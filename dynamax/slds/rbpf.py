r"""
Rao-Blackwellized particle filters for switching linear dynamical systems (SLDS).

The model has discrete states $z_t \in \{1, \ldots, K\}$ and continuous states $x_t$:

$$z_0 \sim \mathrm{Cat}(\pi), \qquad x_0 \mid z_0 \sim \mathcal{N}(\mu_{z_0}, \Sigma_{z_0})$$
$$z_t \mid z_{t-1} \sim \mathrm{Cat}(P_{z_{t-1}}), \qquad x_t \mid x_{t-1}, z_t \sim \mathcal{N}(F_{z_t} x_{t-1} + B_{z_t} u_t + b_{z_t}, Q_{z_t})$$
$$y_t \mid x_t, z_t \sim \mathcal{N}(H_{z_t} x_t + D_{z_t} u_t + d_{z_t}, R_{z_t})$$

for $t = 1, \ldots, T-1$, with $y_0$ emitted from $x_0$. As in `LinearGaussianSSM`, the initial state is at
time 0 (there is no transition before $y_0$) and the pair $t-1 \to t$ uses the input $u_t$.
"""
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import lax, vmap, jit
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import Array, Float, Int
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import NamedTuple, Optional

from dynamax.utils.utils import psd_solve
from dynamax.types import PRNGKeyT

class DiscreteParamsRBPF(NamedTuple):
    """
    Parameters of the discrete state dynamics of an SLDS, plus the proposal used by `rbpfilter`.
    """
    initial_distribution: Float[Array, " num_states"]
    transition_matrix : Float[Array, "num_states num_states"]
    proposal_transition_matrix : Float[Array, "num_states num_states"]

class LGParamsRBPF(NamedTuple):
    r"""Linear Gaussian parameters of an SLDS, one set per discrete state $z_t$: the initial distribution of
    the continuous state $x_0$, its dynamics and the emissions $y_t$.
    """
    initial_mean: Float[Array, "num_states state_dim"]
    initial_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_weights: Float[Array, "num_states state_dim state_dim"]
    dynamics_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_bias: Float[Array, "num_states state_dim"]
    dynamics_input_weights: Float[Array, "num_states state_dim input_dim"]
    emission_weights: Float[Array, "num_states emission_dim state_dim"]
    emission_cov: Float[Array, "num_states emission_dim emission_dim"]
    emission_bias: Float[Array, "num_states emission_dim"]
    emission_input_weights: Float[Array, "num_states emission_dim input_dim"]


class ParamsRBPF(NamedTuple):
    r"""Parameters of a SLDS, which has a discrete component and a continuous component.
    The discrete component is a Markov model for the hidden discrete state $z_t$.
    The linear_gaussian component is a linear Gaussian state space model for the hidden continuous state $x_t$ and
    emission $y_t$.
    """
    discrete: DiscreteParamsRBPF
    linear_gaussian: LGParamsRBPF

    def initialize(self, num_states, state_dim, emission_dim, input_dim=0):
        """
        Initialize the parameters of the SLDS. For each parameter that is the same over all models, the parameter
        is repeated to match the number of states. Also optional paramters are set to zero if not specified.
        Raises a ValueError if a parameter has neither the shared nor the per-state shape.
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
        
        initialized = ParamsRBPF(
            discrete=self.discrete,
            linear_gaussian=LGParamsRBPF(
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
            )
        )
        _check_params(initialized)
        return initialized


def _check_params(params: ParamsRBPF) -> None:
    """Raise a ValueError unless `params` has the per-state layout that `ParamsRBPF.initialize` produces.

    Only shapes are read, so the check is static under `jax.jit`.
    """
    fields = {**params.discrete._asdict(), **params.linear_gaussian._asdict()}
    if any(value is None for value in fields.values()):
        raise ValueError("ParamsRBPF has fields set to None; call ParamsRBPF.initialize first.")
    K = jnp.shape(fields["initial_distribution"])[-1]
    D = jnp.shape(fields["initial_mean"])[-1]
    N = jnp.shape(fields["emission_cov"])[-1]
    M = jnp.shape(fields["dynamics_input_weights"])[-1]
    expected = dict(
        initial_distribution=(K,), transition_matrix=(K, K), proposal_transition_matrix=(K, K),
        initial_mean=(K, D), initial_cov=(K, D, D),
        dynamics_weights=(K, D, D), dynamics_cov=(K, D, D), dynamics_bias=(K, D), dynamics_input_weights=(K, D, M),
        emission_weights=(K, N, D), emission_cov=(K, N, N), emission_bias=(K, N), emission_input_weights=(K, N, M))
    for name, shape in expected.items():
        if jnp.shape(fields[name]) != shape:
            raise ValueError(f"ParamsRBPF field {name} has shape {jnp.shape(fields[name])}, expected {shape}; "
                             "call ParamsRBPF.initialize first.")


class RBPFiltered(NamedTuple):
    r"""RBPF posterior.
    :param weights: weights of the particles.
    :param means: array of filtered means $\mathbb{E}[x_t \mid y_{1:t}, u_{1:t}]$
    :param covariances: array of filtered covariances $\mathrm{Cov}[x_t \mid y_{1:t}, u_{1:t}]$
    :param states: array of sampled discrete state sequences (particles) $$.
    """
    weights: Optional[Float[Array, "num_particles ntime"]] = None
    states: Optional[Int[Array, "num_particles ntime num_states"]] = None
    means: Optional[Float[Array, "num_particles ntime state_dim"]] = None
    covariances: Optional[Float[Array, "num_particles ntime state_dim state_dim"]] = None
   

def resampling(weights, states, means, covariances, key):
    """Resample particles using the multinomial resampling algorithm."""
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(key, jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    new_states = jnp.take(states, resampled_idx, axis=0)
    filtered_means = jnp.take(means, resampled_idx, axis=0)
    filtered_covs = jnp.take(covariances, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    return weights, new_states, filtered_means, filtered_covs

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

def _predict(state, mu, Sigma, params, u):
    """
    Push the Gaussian N(mu, Sigma) over x_{t-1} through the dynamics of discrete state `state` (the pair t-1 -> t uses u_t).
    """
    F = params.dynamics_weights[state]
    b = params.dynamics_bias[state]
    B = params.dynamics_input_weights[state]
    Q = params.dynamics_cov[state]
    mu_pred = F @ mu + B @ u + b
    Sigma_pred = F @ Sigma @ F.T + Q
    return mu_pred, Sigma_pred

def _condition_on(state, mu_pred, Sigma_pred, params, u, y):
    """
    Condition the Gaussian N(mu_pred, Sigma_pred) over x_t on y_t, under the emission model of discrete state `state`.
    Returns log p(y_t | z_t = state, y_{0:t-1}) and the conditioned mean and covariance.
    """
    H = params.emission_weights[state]
    d = params.emission_bias[state]
    D = params.emission_input_weights[state]
    R = params.emission_cov[state]
    S = R + H @ Sigma_pred @ H.T
    K = psd_solve(S, H @ Sigma_pred).T
    mu_y = H @ mu_pred + D @ u + d
    ll = MVN(loc = mu_y, covariance_matrix = S).log_prob(y)
    mu_cond = mu_pred + K @ (y - mu_y)
    Sigma_cond = Sigma_pred - K @ S @ K.T
    return ll, mu_cond, Sigma_cond

def _stack_outputs(initial, rest):
    """Prepend the t=0 particles to the scanned t=1..T-1 particles and name the outputs."""
    weights, states, means, covs = tree_map(lambda x0, x: jnp.concatenate((x0[None], x)), initial, rest)
    return {"weights": weights, "states": states, "means": means, "covariances": covs}

def rbpfilter(
    num_particles: int,
    params: ParamsRBPF,
    emissions:  Float[Array, "ntime emission_dim"],
    key: PRNGKeyT = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ess_threshold: float = 0.5
    ):
    r"""
    Implementation of the Rao-Blackwellized particle filter, for approximating the
    filtering distribution of a switching linear dynamical system. At $t=0$ the filter samples $z_0$ from the
    initial distribution and conditions $x_0 \sim \mathcal{N}(\mu_{z_0}, \Sigma_{z_0})$ on $y_0$. At each later
    step it samples $z_t$ from the proposal `proposal_transition_matrix[z_{t-1}]`, runs a KF step conditional on
    the sampled value of the chain, and multiplies each weight by the emission likelihood times the importance
    ratio `transition_matrix[z_{t-1}, z_t] / proposal_transition_matrix[z_{t-1}, z_t]`. After each step it
    computes an effective sample size and resamples when it falls below `ess_threshold * num_particles`.
    The function is jittable with `num_particles` static.

    Returns:
        dict with the particles after each step (after resampling, if it ran): "weights" (ntime, num_particles),
        "states" (ntime, num_particles), "means" (ntime, num_particles, state_dim) and
        "covariances" (ntime, num_particles, state_dim, state_dim).
    """
    _check_params(params)
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, params.linear_gaussian.dynamics_input_weights.shape[-1])) if inputs is None else inputs
    lg_params = params.linear_gaussian
    log_transition = jnp.log(params.discrete.transition_matrix)
    log_proposal = jnp.log(params.discrete.proposal_transition_matrix)

    def _update(weights, log_ratios, states, pred_means, pred_covs, u, y, key):
        """Condition each particle on y_t, reweight it, and resample if the effective sample size is low."""
        lls, filtered_means, filtered_covs = vmap(_condition_on, in_axes=(0, 0, 0, None, None, None))(states, pred_means, pred_covs, lg_params, u, y)
        log_weights = jnp.log(weights) + log_ratios + lls
        weights = jnp.exp(log_weights - logsumexp(log_weights))
        resample_cond = 1.0 / jnp.sum(jnp.square(weights)) < ess_threshold * num_particles
        return lax.cond(resample_cond, partial(resampling, key=key), lambda *args: args,
                        weights, states, filtered_means, filtered_covs)

    def _step(carry, args):
        r"""
        carry = (weights, prev_states, filtered_means, filtered_covs, key)

        where,
            weights : (num_particles, )
            prev_states : (num_particles, )
            filtered_means : (num_particles, state_dim)
            filtered_covs : (num_particles, state_dim, state_dim)
        """
        weights, prev_states, filtered_means, filtered_covs, key = carry
        u, y = args
        key, sample_key, resample_key = jr.split(key, 3)

        # Sample z_t from the proposal; the importance ratio corrects for it not being the transition matrix
        new_states = jr.categorical(sample_key, log_proposal[prev_states])
        log_ratios = log_transition[prev_states, new_states] - log_proposal[prev_states, new_states]

        # Predict x_t given z_t, then condition on y_t, reweight and resample
        pred_means, pred_covs = vmap(_predict, in_axes=(0, 0, 0, None, None))(new_states, filtered_means, filtered_covs, lg_params, u)
        particles = _update(weights, log_ratios, new_states, pred_means, pred_covs, u, y, resample_key)
        return (*particles, key), particles

    # t = 0: sample z_0 from the initial distribution and condition x_0 ~ N(mu_{z_0}, Sigma_{z_0}) on y_0
    key, sample_key, resample_key = jr.split(key, 3)
    initial_states = jr.choice(sample_key, params.discrete.initial_distribution.shape[0], shape=(num_particles,), p=params.discrete.initial_distribution)
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    initial_particles = _update(initial_weights, jnp.zeros(num_particles), initial_states,
                                lg_params.initial_mean[initial_states], lg_params.initial_cov[initial_states],
                                inputs[0], emissions[0], resample_key)

    _, particles = lax.scan(_step, (*initial_particles, key), (inputs[1:], emissions[1:]))
    return _stack_outputs(initial_particles, particles)

def rbpfilter_optimal(
    num_particles: int,
    params: ParamsRBPF,
    emissions:  Float[Array, "ntime emission_dim"],
    key: PRNGKeyT = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ):
    r"""
    Implementation of the Rao-Blackwellized particle filter with optimal resampling. At each step every particle
    is extended to all discrete states $z_t$ (to all $z_0$ at $t=0$, which has no transition), each extension is
    weighted by its transition (initial) probability times its emission likelihood, and `num_particles` of the
    extensions are kept with the optimal resampling of Fearnhead and Clifford (2003).
    The function is jittable with `num_particles` static.

    Returns:
        dict with the particles after each step, with the same keys and shapes as `rbpfilter`.
    """
    _check_params(params)
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, params.linear_gaussian.dynamics_input_weights.shape[-1])) if inputs is None else inputs
    lg_params = params.linear_gaussian
    num_states, state_dim = lg_params.initial_mean.shape
    log_transition = jnp.log(params.discrete.transition_matrix)

    def _update(weights, log_trans_probs, pred_means, pred_covs, u, y, key):
        """Condition every (particle, z_t) extension on y_t, weight it, and keep num_particles of them.

        log_trans_probs, pred_means and pred_covs have leading axes (num_particles, num_states).
        """
        _condition_all = vmap(vmap(_condition_on, in_axes=(0, 0, 0, None, None, None)), in_axes=(None, 0, 0, None, None, None))
        lls, filtered_means, filtered_covs = _condition_all(jnp.arange(num_states), pred_means, pred_covs, lg_params, u, y)
        log_weights = jnp.log(weights)[:, None] + log_trans_probs + lls
        weights = jnp.exp(log_weights - logsumexp(log_weights))

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
        return res_weights, new_states, filtered_means, filtered_covs

    def _step(carry, args):
        r"""
        carry = (weights, prev_states, filtered_means, filtered_covs, key)

        where,
            weights : (num_particles, )
            prev_states : (num_particles, )
            filtered_means : (num_particles, state_dim)
            filtered_covs : (num_particles, state_dim, state_dim)
        """
        weights, prev_states, filtered_means, filtered_covs, key = carry
        u, y = args
        key, resample_key = jr.split(key, 2)

        # Predict x_t for every particle and every value of z_t
        _predict_all = vmap(vmap(_predict, in_axes=(0, None, None, None, None)), in_axes=(None, 0, 0, None, None))
        pred_means, pred_covs = _predict_all(jnp.arange(num_states), filtered_means, filtered_covs, lg_params, u)
        particles = _update(weights, log_transition[prev_states], pred_means, pred_covs, u, y, resample_key)
        return (*particles, key), particles

    # t = 0: extend each particle to every z_0, with x_0 ~ N(mu_{z_0}, Sigma_{z_0}) and no transition
    key, resample_key = jr.split(key, 2)
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    expand = lambda x: jnp.broadcast_to(x, (num_particles,) + x.shape)
    initial_particles = _update(initial_weights, expand(jnp.log(params.discrete.initial_distribution)),
                                expand(lg_params.initial_mean), expand(lg_params.initial_cov),
                                inputs[0], emissions[0], resample_key)

    _, particles = lax.scan(_step, (*initial_particles, key), (inputs[1:], emissions[1:]))
    return _stack_outputs(initial_particles, particles)
