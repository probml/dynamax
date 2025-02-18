"""
Parallel implementations of the forward filtering and backward smoothing algorithms
"""
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap, value_and_grad
from jaxtyping import Array, Float, Int
from typing import NamedTuple, Tuple

from dynamax.hidden_markov_model.inference import HMMPosterior, HMMPosteriorFiltered
from dynamax.types import Scalar

#---------------------------------------------------------------------------#
#                                Filtering                                  #
#---------------------------------------------------------------------------#

class FilterMessage(NamedTuple):
    r"""Filtering associative scan elements.

    Attributes:
        A: $p(z_j \mid z_i)$
        log_b: $\log P(y_{i+1}, ..., y_j \mid z_i)$
    """    
    A: Float[Array, "num_timesteps num_states num_states"]
    log_b: Float[Array, "num_timesteps num_states"]


def _condition_on(A : Float[Array, "num_states num_states"], 
                  ll : Float[Array, " num_states"], 
                  axis : int=-1) -> \
                  Tuple[Float[Array, "num_states num_states"], Float[Array, "num_states"]]:
    """ Update the message by conditioning on new observations.
    """
    ll_max = ll.max(axis=axis)
    A_cond = A * jnp.exp(ll - ll_max)
    norm = A_cond.sum(axis=axis)
    A_cond /= jnp.expand_dims(norm, axis=axis)
    return A_cond, jnp.log(norm) + ll_max


def hmm_filter(initial_probs: Float[Array, " num_states"],
               transition_matrix: Float[Array, "num_states num_states"],
               log_likelihoods: Float[Array, "num_timesteps num_states"]
) -> HMMPosteriorFiltered:
    r"""Parallel implementation of the forward filtering algorithm with `jax.lax.associative_scan`.

    *Note: for this function, the transition matrix must be fixed. We may add support
    for nonstationary transition matrices in a future release.*

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.

    Returns:
        filtered posterior distribution

    """
    T, K = log_likelihoods.shape

    @vmap
    def marginalize(m_ij, m_jk):
        """
        Compute the message from time i to time k by marginalizing out 
        the hidden state at time j.
        """
        A_ij_cond, lognorm = _condition_on(m_ij.A, m_jk.log_b)
        A_ik = A_ij_cond @ m_jk.A
        log_b_ik = m_ij.log_b + lognorm
        return FilterMessage(A=A_ik, log_b=log_b_ik)


    # Initialize the messages
    A0, log_b0 = _condition_on(initial_probs, log_likelihoods[0])
    A0 *= jnp.ones((K, K))
    log_b0 *= jnp.ones(K)
    A1T, log_b1T = vmap(_condition_on, in_axes=(None, 0))(transition_matrix, log_likelihoods[1:])
    initial_messages = FilterMessage(
        A=jnp.concatenate([A0[None, :, :], A1T]),
        log_b=jnp.vstack([log_b0, log_b1T])
    )

    # Run the associative scan
    partial_messages = lax.associative_scan(marginalize, initial_messages)

    # Extract the marginal log likelihood and filtered probabilities
    marginal_loglik = partial_messages.log_b[-1,0]
    filtered_probs = partial_messages.A[:, 0, :]

    # Compute the predicted probabilities
    predicted_probs = jnp.vstack([initial_probs, filtered_probs[:-1] @ transition_matrix])

    # Package into a posterior object
    return HMMPosteriorFiltered(marginal_loglik=marginal_loglik,
                                filtered_probs=filtered_probs,
                                predicted_probs=predicted_probs)


#---------------------------------------------------------------------------#
#                                 Smoothing                                 #
#---------------------------------------------------------------------------#


def hmm_smoother(initial_probs: Float[Array, " num_states"],
                 transition_matrix: Float[Array, "num_states num_states"],
                 log_likelihoods: Float[Array, "num_timesteps num_states"]
) -> HMMPosterior:
    r"""Parallel implementation of HMM smoothing with `jax.lax.associative_scan`.

    **Notes:**

    * This implementation uses the automatic differentiation of the HMM log normalizer rather than an explicit implementation of the backward message passing.
    * The transition matrix must be fixed. We may add support for nonstationary transition matrices in a future release.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.

    Returns:
        smoothed posterior distribution

    """
    def log_normalizer(log_initial_probs, log_transition_matrix, log_likelihoods):
        """Compute the log normalizer of the HMM model."""
        post = hmm_filter(jnp.exp(log_initial_probs),
                                   jnp.exp(log_transition_matrix),
                                   log_likelihoods)
        return post.marginal_loglik, post

    f = value_and_grad(log_normalizer, has_aux=True, argnums=(1, 2))
    (marginal_loglik, fwd_post), (trans_probs, smoothed_probs) = \
        f(jnp.log(initial_probs), jnp.log(transition_matrix), log_likelihoods)

    return HMMPosterior(
        marginal_loglik=marginal_loglik,
        filtered_probs=fwd_post.filtered_probs,
        predicted_probs=fwd_post.predicted_probs,
        initial_probs=smoothed_probs[0],
        smoothed_probs=smoothed_probs,
        trans_probs=trans_probs
    )


#---------------------------------------------------------------------------#
#                                  Sampling                                 #
#---------------------------------------------------------------------------#
r"""Associative scan elements $E_ij$ are vectors specifying a sample::

    $z_j ~ p(z_j \mid z_i)$ 
    
for each possible value of $z_i$.
"""

def _initialize_sampling_messages(key, transition_matrix, filtered_probs):
    """Preprocess filtering output to construct input for sampling assocative scan."""

    T, K = filtered_probs.shape
    keys = jr.split(key, T)

    def _last_message(key, probs):
        """Sample the last hidden state."""
        state = jr.choice(key, K, p=probs)
        return jnp.repeat(state, K)

    @vmap
    def _generic_message(key, probs):
        """Sample a hidden state given the previous state."""
        smoothed_probs = probs * transition_matrix.T
        smoothed_probs = smoothed_probs / smoothed_probs.sum(1).reshape(K,1)
        return vmap(lambda p: jr.choice(key, K, p=p))(smoothed_probs)

    En = _last_message(keys[-1], filtered_probs[-1])
    Et = _generic_message(keys[:-1], filtered_probs[:-1])
    return jnp.concatenate([Et, En[None]])


def hmm_posterior_sample(key: Array,
                         initial_distribution: Float[Array, " num_states"],
                         transition_matrix: Float[Array, "num_states num_states"],
                         log_likelihoods: Float[Array, "num_timesteps num_states"]
) -> Tuple[Scalar, Int[Array, " num_timesteps"]]:
    r"""Sample a sequence of hidden states from the posterior.

    Args:
        key: random number generator
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
 
    Returns:
        log_normalizer: $\log P(y_{1:T} \mid u_{1:T}, \theta)$
        states: sequence of hidden states $z_{1:T}$
    """
    # Run the HMM filter
    post = hmm_filter(initial_distribution, transition_matrix, log_likelihoods)
    log_normalizer = post.marginal_loglik
    filtered_probs = post.filtered_probs

    @vmap
    def _operator(E_jk, E_ij):
        """Sample a hidden state given the previous state."""
        return jnp.take(E_ij, E_jk)

    initial_messages = _initialize_sampling_messages(key, transition_matrix, filtered_probs)
    final_messages = lax.associative_scan(_operator, initial_messages, reverse=True)
    states = final_messages[:,0]
    return log_normalizer, states
