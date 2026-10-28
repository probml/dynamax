r"""
Structured mean-field Laplace variational inference for switching linear dynamical systems.

The posterior is approximated by :math:`q(z_{1:T})\, q(x_{1:T})`: :math:`q(z)` is a Markov chain
(an HMM with unary and pairwise potentials, smoothed in log space), :math:`q(x)` a Gaussian
with block-tridiagonal precision (handled by the primitives in
:mod:`dynamax.linear_gaussian_ssm.block_tridiag`). Coordinate ascent on the ELBO alternates
:func:`update_discrete_posterior` and :func:`update_continuous_posterior`; the latter is a Newton
step at the mode of :math:`\mathbb{E}_{q(z)}[\log p(z, x, y)]` (exact for Gaussian SLDS, a Laplace
approximation for recurrent transitions). Ported from ``lindermanlab/ssm``.
"""
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.scipy.special import logsumexp, xlogy
from jax.tree_util import tree_map
from jaxtyping import Array, Float, Int

from dynamax.linear_gaussian_ssm.block_tridiag import (
    BlockTridiagPotentials, block_tridiag_mvn_entropy, block_tridiag_mvn_matvec, block_tridiag_mvn_smoother,
    block_tridiag_mvn_solve)
from dynamax.types import PRNGKeyT, Scalar


class DiscretePosteriorSLDS(NamedTuple):
    r"""The chain :math:`q(z)`: the potentials of :func:`log_space_hmm_smoother` and its output.

    The transition matrices are kept in log space. In float32 the transition probabilities of a stiff chain
    underflow to exact zeros (half of them on the NASCAR rSLDS), so anything re-reading a stored
    probability-space chain (the sampler, the entropy) would see a different chain from the smoother,
    which inside ``jit`` runs on the simplified ``log(exp(.))``.

    :param initial_probs: :math:`\pi_0`
    :param log_transition_matrices: log of the (K, K), or (T-1, K, K) for recurrent transitions, transition matrices
    :param log_likelihoods: unary potentials (T, K), including the folded row normalizer for recurrent transitions
    :param log_normalizer: log partition function of the chain
    :param smoothed_probs: :math:`\mathbb{E}_q[z_t = k]`
    :param trans_probs: :math:`\mathbb{E}_q[z_t = i, z_{t+1} = j]`, summed over t for (K, K) transitions
    """
    initial_probs: Float[Array, " num_states"]
    log_transition_matrices: Array
    log_likelihoods: Float[Array, "num_timesteps num_states"]
    log_normalizer: Scalar
    smoothed_probs: Float[Array, "num_timesteps num_states"]
    trans_probs: Array


class ContinuousPosteriorSLDS(NamedTuple):
    r"""The Gaussian :math:`q(x)` in information form, with its filtered messages and marginals.

    :param J_diag, J_lower, h: the block-tridiagonal natural parameters
    :param filtered_precisions, filtered_linear_potentials: cached forward messages for the sampler
    :param log_normalizer: :math:`\log Z`
    :param log_det_precision: :math:`\log |J|`
    :param smoothed_means, smoothed_covariances: marginals
    :param smoothed_cross_covariances: :math:`\mathbb{E}[x_t x_{t+1}^\top]`, uncentered
    """
    J_diag: Float[Array, "num_timesteps state_dim state_dim"]
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"]
    h: Float[Array, "num_timesteps state_dim"]
    filtered_precisions: Float[Array, "num_timesteps state_dim state_dim"]
    filtered_linear_potentials: Float[Array, "num_timesteps state_dim"]
    log_normalizer: Scalar
    log_det_precision: Scalar
    smoothed_means: Float[Array, "num_timesteps state_dim"]
    smoothed_covariances: Float[Array, "num_timesteps state_dim state_dim"]
    smoothed_cross_covariances: Float[Array, "num_timesteps_minus_1 state_dim state_dim"]


class PosteriorSLDS(NamedTuple):
    """The variational posterior q(z) q(x)."""
    discrete: DiscretePosteriorSLDS
    continuous: ContinuousPosteriorSLDS


def continuous_posterior_from_potentials(potentials: BlockTridiagPotentials, jitter: float = 0.0) -> ContinuousPosteriorSLDS:
    """Run the block-tridiagonal smoother and package the result with the potentials."""
    post = block_tridiag_mvn_smoother(potentials, jitter)
    return ContinuousPosteriorSLDS(J_diag=potentials.J_diag, J_lower=potentials.J_lower, h=potentials.h,
                                   filtered_precisions=post.filtered_precisions,
                                   filtered_linear_potentials=post.filtered_linear_potentials,
                                   log_normalizer=post.log_normalizer,
                                   log_det_precision=post.log_det_precision,
                                   smoothed_means=post.smoothed_means,
                                   smoothed_covariances=post.smoothed_covariances,
                                   smoothed_cross_covariances=post.smoothed_cross_covariances)


def _trivial_discrete_posterior(num_timesteps: int, time_varying: bool) -> DiscretePosteriorSLDS:
    """q(z) for a single state: all probabilities one, zero entropy, zero log normalizer."""
    T = num_timesteps
    if time_varying:
        log_transition_matrices, trans_probs = jnp.zeros((T - 1, 1, 1)), jnp.ones((T - 1, 1, 1))
    else:
        log_transition_matrices, trans_probs = jnp.zeros((1, 1)), jnp.array([[T - 1.0]])
    return DiscretePosteriorSLDS(initial_probs=jnp.ones(1), log_transition_matrices=log_transition_matrices,
                                 log_likelihoods=jnp.zeros((T, 1)), log_normalizer=jnp.zeros(()),
                                 smoothed_probs=jnp.ones((T, 1)), trans_probs=trans_probs)


def _log_space_forward(initial_probs: Array, log_transition_matrices: Array, log_likelihoods: Array
                       ) -> Tuple[Scalar, Array, Array]:
    """The log forward messages, renormalized at every step: ``(log_normalizer, log_alphas (T, K), log_P_t (T-1, K, K))``.

    The messages are renormalized at every step (their log normalizers are accumulated separately), so
    that float32 keeps its relative precision when the potentials are in the thousands.
    """
    num_timesteps, num_states = log_likelihoods.shape
    log_P_t = (log_transition_matrices if log_transition_matrices.ndim == 3
               else jnp.broadcast_to(log_transition_matrices, (num_timesteps - 1, num_states, num_states)))

    def _forward(carry, args):
        log_alpha, log_Z = carry
        log_P_prev, ll = args
        log_alpha = logsumexp(log_alpha[:, None] + log_P_prev, axis=0) + ll
        c = logsumexp(log_alpha)
        return (log_alpha - c, log_Z + c), log_alpha - c

    log_alpha0 = jnp.log(initial_probs) + log_likelihoods[0]
    c0 = logsumexp(log_alpha0)
    (_, log_normalizer), log_alphas = lax.scan(_forward, (log_alpha0 - c0, c0), (log_P_t, log_likelihoods[1:]))
    return log_normalizer, jnp.concatenate([(log_alpha0 - c0)[None], log_alphas]), log_P_t


def log_space_hmm_smoother(initial_probs: Array, log_transition_matrices: Array, log_likelihoods: Array
                           ) -> Tuple[Scalar, Array, Array]:
    r"""Forward-backward in log space (as ``ssm``'s ``hmm_expected_states``).

    The unary potentials of :math:`q(z)` are expected log densities whose gaps between states reach
    thousands of nats when the dynamics noise is small, so probabilities underflow in the
    probability-space :func:`hmm_smoother`: a state whose forward message underflows to zero can never
    be revived by later evidence. Working with log messages keeps every state alive. The transition
    matrices are given in log space, (K, K) or (T-1, K, K). Returns the log normalizer, the smoothed
    probabilities (T, K) and the pairwise expectations, summed over time (K, K) for a single transition
    matrix and per time step (T-1, K, K) for time-varying ones.
    """
    num_states = log_likelihoods.shape[1]
    time_varying = log_transition_matrices.ndim == 3
    log_normalizer, log_alphas, log_P_t = _log_space_forward(initial_probs, log_transition_matrices, log_likelihoods)

    def _backward(log_beta, args):
        log_P_next, ll_next = args
        log_beta = logsumexp(log_P_next + (ll_next + log_beta)[None, :], axis=1)
        return log_beta - logsumexp(log_beta), log_beta - logsumexp(log_beta)

    _, log_betas = lax.scan(_backward, jnp.zeros(num_states), (log_P_t, log_likelihoods[1:]), reverse=True)
    log_betas = jnp.concatenate([log_betas, jnp.zeros((1, num_states))])
    smoothed_probs = jax.nn.softmax(log_alphas + log_betas, axis=1)
    log_pairs = log_alphas[:-1, :, None] + log_P_t + (log_likelihoods[1:] + log_betas[1:])[:, None, :]
    trans_probs = jnp.exp(log_pairs - logsumexp(log_pairs, axis=(1, 2), keepdims=True))
    if not time_varying:
        trans_probs = trans_probs.sum(0)
    return log_normalizer, smoothed_probs, trans_probs


def log_space_hmm_posterior_sample(key: PRNGKeyT, initial_probs: Array, log_transition_matrices: Array,
                                   log_likelihoods: Array) -> Int[Array, " num_timesteps"]:
    r"""A sample of :math:`z_{0:T-1}` from the chain, as :func:`~dynamax.hidden_markov_model.hmm_posterior_sample`
    but with log messages (and returning the states only).

    The forward messages are those of :func:`log_space_hmm_smoother`; the backward pass draws
    :math:`z_t \sim \mathrm{Cat}(\alpha_t(z_t)\, P_t(z_t, z_{t+1}))` from logits. The probability-space
    sampler fails on the same potentials that break the probability-space smoother: once the filtered
    probabilities underflow to an all-zero row, every later state is drawn from garbage.
    """
    _, log_alphas, log_P_t = _log_space_forward(initial_probs, log_transition_matrices, log_likelihoods)
    keys = jr.split(key, log_likelihoods.shape[0])
    last_state = jr.categorical(keys[-1], log_alphas[-1])

    def _step(next_state, args):
        subkey, log_alpha, log_P_next = args
        state = jr.categorical(subkey, log_alpha + log_P_next[:, next_state])
        return state, state

    _, states = lax.scan(_step, last_state, (keys[:-1], log_alphas[:-1], log_P_t), reverse=True)
    return jnp.concatenate([states, last_state[None]])


def _discrete_posterior(initial_probs: Array, log_transition_matrices: Array, log_likelihoods: Array) -> DiscretePosteriorSLDS:
    """Run the log-space smoother on the chain potentials."""
    log_normalizer, smoothed_probs, trans_probs = log_space_hmm_smoother(initial_probs, log_transition_matrices, log_likelihoods)
    return DiscretePosteriorSLDS(initial_probs=initial_probs, log_transition_matrices=log_transition_matrices,
                                 log_likelihoods=log_likelihoods, log_normalizer=log_normalizer,
                                 smoothed_probs=smoothed_probs, trans_probs=trans_probs)


def initialize_posterior(model, params, emissions: Array, inputs: Array) -> PosteriorSLDS:
    """Uniform q(z) and q(x) = N(x̂, initial_variance I) around the inverted emissions."""
    num_timesteps = emissions.shape[0]
    K, D = model.num_states, model.state_dim
    time_varying = model.transition_component.depends_on_x
    if K == 1:
        qz = _trivial_discrete_posterior(num_timesteps, time_varying)
    else:
        shape = (num_timesteps - 1, K, K) if time_varying else (K, K)
        qz = _discrete_posterior(jnp.full(K, 1.0 / K), jnp.full(shape, -jnp.log(K)), jnp.zeros((num_timesteps, K)))
    x_hat = model.emission_component.invert(params.emissions, emissions, inputs)
    potentials = BlockTridiagPotentials(J_diag=jnp.tile(jnp.eye(D) / model.initial_variance, (num_timesteps, 1, 1)),
                                        J_lower=jnp.zeros((num_timesteps - 1, D, D)),
                                        h=x_hat / model.initial_variance)
    return PosteriorSLDS(discrete=qz, continuous=continuous_posterior_from_potentials(potentials, model.jitter))


def update_discrete_posterior(model, params, qx: ContinuousPosteriorSLDS, x_samples: Array,
                              emissions: Array, inputs: Array) -> DiscretePosteriorSLDS:
    r"""Coordinate-ascent update :math:`q(z) \propto \exp \mathbb{E}_{q(x)}[\log p(z, x, y)]`.

    The dynamics (and multi-subspace emission) terms are exact Gaussian expectations. Recurrent
    transition terms are Monte Carlo over ``x_samples``: the pairwise potential is
    :math:`\exp \mathbb{E}[\log P_t]`, row-normalized, with its row normalizer folded into the unary
    potential on :math:`z_t` so the chain is exact given the estimate.
    """
    transitions, dynamics, emissions_component = model._components
    num_timesteps = emissions.shape[0]
    if model.num_states == 1:
        return _trivial_discrete_posterior(num_timesteps, transitions.depends_on_x)
    initial_probs = params.initial.probs
    log_likelihoods = dynamics.expected_log_likelihoods(params.dynamics, qx, inputs)
    if not emissions_component.single_subspace:
        log_likelihoods = log_likelihoods + emissions_component.expected_log_likelihoods(params.emissions, emissions, qx, inputs)
    if transitions.depends_on_x:
        expected_log_P = vmap(lambda x: transitions.log_transition_matrices(params.transitions, x, inputs))(x_samples).mean(0)
        log_c = logsumexp(expected_log_P, axis=-1)                       # (T-1, K)
        log_transition_matrices = expected_log_P - log_c[..., None]
        log_likelihoods = log_likelihoods.at[:-1].add(log_c)
    else:
        log_transition_matrices = transitions.log_transition_matrices(params.transitions, qx.smoothed_means, inputs)
    return _discrete_posterior(initial_probs, log_transition_matrices, log_likelihoods)


def backtracking_line_search(f: Callable, x: Array, dx: Array, g: Array,
                             alpha: float = 0.2, beta: float = 0.7, min_step: float = 1e-8) -> Scalar:
    """Armijo backtracking: the largest step in {1, beta, beta², ...} with
    ``f(x + step dx) <= f(x) + alpha step gᵀdx``, where NaN objective values count as no decrease.
    If no step above ``min_step`` qualifies, the first step at or below ``min_step`` is returned
    (and taken, as in ``ssm``)."""
    f0 = f(x)
    slope = jnp.sum(g * dx)

    def _sufficient_decrease(step):
        value = f(x + step * dx)
        value = jnp.where(jnp.isnan(value), jnp.inf, value)
        return value <= f0 + alpha * step * slope

    # The decision is carried rather than recomputed in the loop condition: under vmap, JAX
    # evaluates the condition in two separately compiled copies (loop test and carry select),
    # and in float32 their roundings can disagree, leaving the loop spinning without progress.
    def _body(carry):
        step, _ = carry
        step = beta * step
        return step, _sufficient_decrease(step)

    step, _ = lax.while_loop(lambda c: (c[0] > min_step) & ~c[1], _body,
                             (jnp.ones(()), _sufficient_decrease(jnp.ones(()))))
    return step


def laplace_newton(f: Callable, grad_f: Callable, hess_blocks: Callable, x0: Array,
                   tolerance: float, maxiter: int, quadratic: bool, jitter: float = 0.0) -> Array:
    r"""Newton's method with the block-tridiagonal Hessian (Boyd & Vandenberghe, Alg. 9.5).

    The decrement :math:`\lambda^2 = g^\top J^{-1} g` is checked at the current point before
    stepping, so a converged point is returned as is (``ssm``'s semantics). With ``quadratic``
    one unit step is taken, which is exact. At most ``maxiter`` steps.
    """
    if quadratic:
        maxiter = 1

    def _body(carry):
        x, _, it = carry
        J_diag, J_lower = hess_blocks(x)
        g = grad_f(x)
        dx = -block_tridiag_mvn_solve(J_diag, J_lower, g, jitter)
        lambdasq = -jnp.sum(g * dx)
        if quadratic:
            step = 1.0
        else:
            step = lax.cond(lambdasq / 2 <= tolerance,
                            lambda: jnp.zeros(()),
                            lambda: backtracking_line_search(f, x, dx, g))
        return x + step * dx, lambdasq, it + 1

    x, _, _ = lax.while_loop(lambda c: (c[1] / 2 > tolerance) & (c[2] < maxiter), _body, (x0, jnp.inf, 0))
    return x


def update_continuous_posterior(model, params, qz: DiscretePosteriorSLDS, qx: ContinuousPosteriorSLDS,
                                emissions: Array, inputs: Array) -> ContinuousPosteriorSLDS:
    r"""Laplace update of :math:`q(x)`: the mode of :math:`\mathbb{E}_{q(z)}[\log p(z, x, y)]` by Newton's
    method, with the negative Hessian at the mode as the precision. The objective is scaled by
    :math:`1/(TD)` as in ``ssm`` so that the tolerance is comparable across sizes."""
    transitions, dynamics, emissions_component = model._components
    num_timesteps, state_dim = qx.smoothed_means.shape
    scale = num_timesteps * state_dim
    Ez, Ezzp1 = qz.smoothed_probs, qz.trans_probs

    def neg_expected_log_joint(x):
        lp = jnp.sum(xlogy(Ez[0], params.initial.probs))
        lp += transitions.expected_log_trans_prob(params.transitions, x, inputs, Ezzp1)
        lp += dynamics.expected_log_joint_x(params.dynamics, x, Ez, inputs)
        lp += emissions_component.expected_log_joint_x(params.emissions, emissions, x, Ez, inputs)
        return -lp

    def hess_blocks(x):
        J_diag, J_lower = dynamics.neg_hessian_blocks(params.dynamics, Ez, inputs)
        J_diag = J_diag + emissions_component.neg_hessian_blocks(params.emissions, emissions, x, Ez, inputs)
        if transitions.depends_on_x:
            J_diag = J_diag.at[:-1].add(transitions.neg_hessian_x(params.transitions, x, inputs, Ezzp1))
        return J_diag, J_lower

    f = lambda x: neg_expected_log_joint(x) / scale
    x_star = laplace_newton(f, jax.grad(f), lambda x: tree_map(lambda a: a / scale, hess_blocks(x)),
                            qx.smoothed_means, model.newton_tolerance, model.newton_maxiter, model.quadratic, model.jitter)
    J_diag, J_lower = hess_blocks(x_star)
    h = block_tridiag_mvn_matvec(J_diag, J_lower, x_star)
    return continuous_posterior_from_potentials(BlockTridiagPotentials(J_diag, J_lower, h), model.jitter)


def discrete_entropy(qz: DiscretePosteriorSLDS) -> Scalar:
    r"""Entropy of the chain :math:`q(z)`: :math:`\log Z_z - \mathbb{E}_q[\log \tilde q(z)]`; terms with
    zero probability contribute zero."""
    return (qz.log_normalizer
            - jnp.sum(xlogy(qz.smoothed_probs[0], qz.initial_probs))
            - jnp.sum(qz.smoothed_probs * qz.log_likelihoods)
            - jnp.sum(jnp.where(qz.trans_probs > 0, qz.trans_probs * qz.log_transition_matrices, 0.0)))


def slds_elbo(model, params, posterior: PosteriorSLDS, emissions: Array, inputs: Array, x_samples: Array) -> Scalar:
    r"""Evidence lower bound :math:`\mathbb{E}_q[\log p(z, x, y)] + \mathcal{H}[q(z)] + \mathcal{H}[q(x)]` of one sequence.

    Every term is exact except the recurrent transition term, a Monte Carlo average over ``x_samples``.
    """
    transitions, dynamics, emissions_component = model._components
    qz, qx = posterior.discrete, posterior.continuous
    Ez, Ezzp1 = qz.smoothed_probs, qz.trans_probs
    num_timesteps, state_dim = qx.smoothed_means.shape
    elbo = jnp.sum(xlogy(Ez[0], params.initial.probs))
    if transitions.depends_on_x:
        log_P = vmap(lambda x: transitions.log_transition_matrices(params.transitions, x, inputs))(x_samples).mean(0)
        elbo += jnp.sum(Ezzp1 * log_P)
    else:
        elbo += transitions.expected_log_trans_prob(params.transitions, qx.smoothed_means, inputs, Ezzp1)
    elbo += jnp.sum(Ez * dynamics.expected_log_likelihoods(params.dynamics, qx, inputs))
    emission_terms = emissions_component.expected_log_likelihoods(params.emissions, emissions, qx, inputs)
    elbo += jnp.sum(emissions_component.state_weights(Ez) * emission_terms)
    elbo += discrete_entropy(qz) + block_tridiag_mvn_entropy(qx.log_det_precision, num_timesteps, state_dim)
    return elbo
