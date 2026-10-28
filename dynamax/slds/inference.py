"""
Variational inference for switching linear dynamical systems.

The approximate posterior has a discrete part and a Gaussian part, each updated with the other held
fixed. The Gaussian part is centered at the most likely continuous states, found in one linear solve
when the log joint is quadratic and by Newton's method otherwise; this is exact when the transitions do
not depend on the continuous state and an approximation when they do.
"""
from typing import NamedTuple

from jaxtyping import Array, Float

from dynamax.linear_gaussian_ssm.block_tridiag import BlockTridiagPosterior
from dynamax.types import Scalar


class DiscretePosteriorSLDS(NamedTuple):
    """Approximate posterior over the discrete states of an SLDS, with the Markov chain that defines it.

    :param initial_probs: initial distribution of the chain
    :param log_transition_matrices: log transition matrices of the chain, shape (K, K) when the
        transitions are stationary, (T-1, K, K) otherwise
    :param log_likelihoods: log likelihood of each time step under each discrete state, shape (T, K)
    :param log_normalizer: log normalizer of the chain
    :param smoothed_probs: posterior probability of each discrete state at each time step, shape (T, K)
    :param trans_probs: posterior probability of each pair of consecutive discrete states, summed over
        time to shape (K, K) when the transitions are stationary, (T-1, K, K) otherwise
    """
    initial_probs: Float[Array, " num_states"]
    log_transition_matrices: Array
    log_likelihoods: Float[Array, "num_timesteps num_states"]
    log_normalizer: Scalar
    smoothed_probs: Float[Array, "num_timesteps num_states"]
    trans_probs: Array


ContinuousPosteriorSLDS = BlockTridiagPosterior
"""Gaussian approximate posterior over the continuous states of an SLDS, in information form, with its marginals."""


class PosteriorSLDS(NamedTuple):
    """Approximate posterior over the states of an SLDS: a distribution over the discrete states times a
    Gaussian over the continuous states.

    :param discrete: posterior over the discrete states
    :param continuous: posterior over the continuous states
    """
    discrete: DiscretePosteriorSLDS
    continuous: ContinuousPosteriorSLDS
