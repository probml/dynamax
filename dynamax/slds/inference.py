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
from typing import NamedTuple

from jaxtyping import Array, Float

from dynamax.linear_gaussian_ssm.block_tridiag import BlockTridiagPotentials, block_tridiag_mvn_smoother
from dynamax.types import Scalar


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

