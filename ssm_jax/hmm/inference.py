import jax.numpy as jnp
import jax.random as jr
from jax import lax

import chex

@chex.dataclass
class HMMPosterior:
    """Simple wrapper for properties of an HMM posterior distribution.
    """
    marginal_log_lkhd: chex.Scalar
    filtered_probs: chex.Array
    smoothed_probs: chex.Array
    smoothed_transition_probs: chex.Array


# Helper functions for the two key filtering steps
def _condition_on(probs, ll):
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state.

    Args:
        probs (Array): current probabilities
        ll (Array): log likelihoods of new emissions

    Returns:

    """
    ll_max = ll.max()
    new_probs = probs * jnp.exp(ll - ll_max)
    norm = new_probs.sum()
    new_probs /= norm
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(probs, A):
    return A.T @ probs


def hmm_filter(initial_distribution,
               transition_matrix,
               log_likelihoods):
    """_summary_

    Args:
        hmm_params (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """
    def _step(carry, t):
        log_normalizer, predicted_probs = carry

        # Get parameters for time t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)
        ll = log_likelihoods[t]

        # Condition on emissions at time t, being careful not to overflow
        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        # Update the log normalizer
        log_normalizer += log_norm
        # Predict the next state
        predicted_probs = _predict(filtered_probs, A)

        return (log_normalizer, predicted_probs), (filtered_probs, predicted_probs)

    num_timesteps = len(log_likelihoods)
    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
    return log_normalizer, filtered_probs, predicted_probs


def hmm_posterior_sample(rng,
                         initial_distribution,
                         transition_matrix,
                         log_likelihoods):
    # Run the HMM filter
    log_normalizer, filtered_probs, _ = \
        hmm_filter(initial_distribution, transition_matrix, log_likelihoods)

    # Run the sampler backward in time
    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        t, rng, filtered_probs = args

        # Get parameters for time t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)

        # Fold in the next state and renormalize
        smoothed_probs = filtered_probs * A[:, next_state]
        smoothed_probs /= smoothed_probs.sum()

        # Sample current state
        state = jr.choice(rng, p=smoothed_probs)

        return state, state

    # Run the HMM smoother
    num_timesteps = len(log_likelihoods)
    rngs = jr.split(rng, num_timesteps)
    last_state = jr.choice(rngs[-1], filtered_probs[-1])
    args = (jnp.arange(num_timesteps - 1, -1, -1),
            rngs[:-1],
            filtered_probs[:-1][::-1])
    _, rev_states = lax.scan(_step, last_state, args)

    # Reverse the arrays and return
    states = jnp.row_stack([rev_states[::-1], last_state])
    return log_normalizer, states


def hmm_backward_filter(transition_matrix,
                        log_likelihoods):
    """_summary_

    Args:
        hmm_params (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """
    def _step(carry, t):
        log_normalizer, backward_pred_probs = carry

        # Get parameters for time t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)
        ll = log_likelihoods[t]

        # Condition on emission at time t, being careful not to overflow.
        backward_filt_probs, log_norm = _condition_on(backward_pred_probs, ll)
        # Update the log normalizer.
        log_normalizer += log_norm
        # Predict the next state (going backward in time).
        next_backward_pred_probs = _predict(backward_filt_probs, A.T)
        return (log_normalizer, next_backward_pred_probs), backward_pred_probs

    num_timesteps, num_states = log_likelihoods.shape
    carry = (0.0, jnp.ones(num_states))
    (log_normalizer, _), rev_backward_pred_probs = lax.scan(
        _step, carry, jnp.arange(num_timesteps)[::-1])
    backward_pred_probs = rev_backward_pred_probs[::-1]
    return log_normalizer, backward_pred_probs


def hmm_two_filter_smoother(initial_distribution,
                            transition_matrix,
                            log_likelihoods):
    """Computed the smoothed state probabilities using the two-filter
    smoother, a.k.a. the forward-backward algorithm.

    Args:
        initial_distribution (_type_): _description_
        transition_matrix (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Run the filters forward and backward
    ll, filtered_probs, _ = hmm_filter(initial_distribution,
                                       transition_matrix,
                                       log_likelihoods)

    _, backward_pred_probs = hmm_backward_filter(transition_matrix,
                                                 log_likelihoods)

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1, keepdims=True)
    smoothed_probs /= norm

    # Compute smoothed transition probabilities
    ll_max = jnp.max(log_likelihoods, axis=1, keepdims=True)
    smoothed_trans_probs = filtered_probs[:-1, :, None] * \
                           transition_matrix * \
                           jnp.exp(log_likelihoods[1:] - ll_max[1:])[:, None, :] * \
                           backward_pred_probs[1:, None, :]
    smoothed_trans_probs /= smoothed_trans_probs.sum(axis=(1, 2), keepdims=True)

    return HMMPosterior(marginal_log_lkhd=ll,
                        filtered_probs=filtered_probs,
                        smoothed_probs=smoothed_probs,
                        smoothed_transition_probs=smoothed_trans_probs)


def hmm_smoother(initial_distribution,
                 transition_matrix,
                 log_likelihoods):
    """Computed the smoothed state probabilities using a general
    Bayesian smoother.

    Note: This is the discrete SSM analog of the RTS smoother for
    linear Gaussian SSMs.

    Args:
        initial_distribution (_type_): _description_
        transition_matrix (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Run the HMM filter
    ll, filtered_probs, predicted_probs = \
        hmm_filter(initial_distribution, transition_matrix, log_likelihoods)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, filtered_probs, predicted_probs_next = args

        # Get parameters for time t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)

        # Fold in the next state (Eq. 8.2 of Saarka, 2013)
        relative_probs_next = smoothed_probs_next / predicted_probs_next
        smoothed_probs = filtered_probs * (A @ relative_probs_next)
        smoothed_probs /= smoothed_probs.sum()

        # Compute smoothed transition probabilities (Eq. 8.4 of Saarka, 2013)
        smoothed_trans_probs = filtered_probs[:, None] * A * relative_probs_next[None, :]
        smoothed_trans_probs /= smoothed_trans_probs.sum()

        return smoothed_probs, (smoothed_probs, smoothed_trans_probs)

    # Run the HMM smoother
    num_timesteps = len(log_likelihoods)
    carry = filtered_probs[-1]
    args = (jnp.arange(num_timesteps - 2, -1, -1),
            filtered_probs[:-1][::-1],
            predicted_probs[:-1][::-1])
    _, (rev_smoothed_probs, rev_smoothed_trans_probs) = lax.scan(_step, carry, args)

    # Reverse the arrays and return
    smoothed_probs = jnp.row_stack([rev_smoothed_probs[::-1], filtered_probs[-1]])
    smoothed_trans_probs = rev_smoothed_trans_probs[::-1]

    return HMMPosterior(marginal_log_lkhd=ll,
                        filtered_probs=filtered_probs,
                        smoothed_probs=smoothed_probs,
                        smoothed_transition_probs=smoothed_trans_probs)


def hmm_posterior_mode(initial_distribution,
                       transition_matrix,
                       log_likelihoods):
    """Compute the most likely state sequence. This is called the Viterbi algorithm.

    Args:
        initial_distribution (_type_): _description_
        transition_matrix (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Run the backward pass
    def _backward_pass(best_next_score, t):
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)

        scores = jnp.log(A) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    best_second_score, rev_best_next_states = \
        lax.scan(_backward_pass,
                 jnp.zeros(num_states),
                 jnp.arange(num_timesteps - 2, -1, -1))
    best_next_states = rev_best_next_states[::-1]

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score)
    _, states = lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])
