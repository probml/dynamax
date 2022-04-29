
import jax.numpy as np
import jax.random as jr
from jax import lax

import chex

@chex.dataclass
class HMMParams:
    initial_distribution: chex.Array
    transition_matrix: chex.Array
    is_stationary: chex.Array = True

    @property
    def num_states(self):
        return self.transition_matrix.shape[-1]


def hmm_filter(hmm_params, log_likelihoods):
    """_summary_

    Args:
        hmm_params (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """
    T = len(log_likelihoods)

    def _step(carry, t):
        log_normalizer, pred_probs = carry

        # Get parameters for time t
        A = lax.cond(hmm_params.is_stationary,
                     hmm_params.transition_matrix,
                     hmm_params.transition_matrix[t])
        ll = log_likelihoods[t]

        # Condition on data at time t, being careful not to overflow.
        ll_max = ll.max()
        filtered_probs = pred_probs * np.exp(ll - ll_max)

        # Renormalize to make filtered_probs a distribution
        norm = filtered_probs.sum()
        filtered_probs /= norm

        # Update the log normalizer
        log_normalizer += np.log(norm) + ll_max

        # Predict the next state
        pred_probs = A.T @ filtered_probs

        return (log_normalizer, pred_probs), filtered_probs

    carry = (0.0, hmm_params.initial_distribution)
    (log_normalizer, _), filtered_probs = lax.scan(_step, carry, np.arange(T))
    return log_normalizer, filtered_probs


def hmm_posterior_sample(rng, hmm_params, log_likelihoods):

    # Run the hmm filter
    log_normalizer, filtered_probs = hmm_filter(hmm_params, log_likelihoods)

    # Run the sampler backward in time
    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        t, rng, filtered_probs = args

        # Get parameters for time t
        A = lax.cond(hmm_params.is_stationary,
                     hmm_params.transition_matrix,
                     hmm_params.transition_matrix[t])

        # Fold in the next state and renormalize
        smoothed_probs = filtered_probs * (A @ next_state)
        smoothed_probs /= smoothed_probs.sum()

        # Sample current state
        state = jr.choice(rng, p=smoothed_probs)

        return state, state

    # Run the HMM smoother
    num_timesteps = len(log_likelihoods)
    rngs = jr.split(rng, num_timesteps)
    last_state = jr.choice(rngs[-1], filtered_probs[-1])
    args = (np.arange(num_timesteps - 1, -1, -1),
            rngs[:-1],
            filtered_probs[:-1][::-1])
    _, rev_states = lax.scan(_step, last_state, args)

    # Reverse the arrays and return
    states = np.row_stack([rev_states[::-1], last_state])
    return log_normalizer, states


def hmm_backward_filter(hmm_params, log_likelihoods):
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
        A = lax.cond(hmm_params.is_stationary,
                     hmm_params.transition_matrix,
                     hmm_params.transition_matrix[t])
        ll = log_likelihoods[t]

        ll_max = ll.max()
        backward_pred_probs = A @ (backward_pred_probs * np.exp(ll - ll_max))

        # Renormalize
        norm = backward_pred_probs.sum()
        backward_pred_probs /= norm

        # Update the log log_normalizer
        log_normalizer += np.log(norm) + ll_max

        return (log_normalizer, backward_pred_probs), backward_pred_probs

    num_timesteps = len(log_likelihoods)
    carry = (0.0, np.ones(hmm_params.num_states))
    (log_normalizer, _), backward_pred_probs = lax.scan(
        _step, carry, np.arange(num_timesteps)[::-1])
    return log_normalizer, backward_pred_probs


def hmm_two_filter_smoother(hmm_params, log_likelihoods):

    ll, filtered_probs = hmm_filter(hmm_params, log_likelihoods)
    _, backward_pred_probs = hmm_backward_filter(hmm_params, log_likelihoods)

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1)
    smoothed_probs /= norm

    return smoothed_probs, ll


def hmm_smoother(hmm_params, log_likelihoods):
    """_summary_

    Args:
        hmm_params (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_timesteps = len(log_likelihoods)

    # Run the hmm filter
    log_normalizer, filtered_probs = hmm_filter(hmm_params, log_likelihoods)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, filtered_probs = args

        # Get parameters for time t
        A = lax.cond(hmm_params.is_stationary,
                     hmm_params.transition_matrix,
                     hmm_params.transition_matrix[t])

        # Fold in the next state and renormalize
        smoothed_probs = filtered_probs * (A @ smoothed_probs_next)
        smoothed_probs /= smoothed_probs.sum()

        # Compute p(z_t=i, z_{t+1}=j)
        p_cross = filtered_probs[:, None] * A * smoothed_probs_next[None, :]
        p_cross /= p_cross.sum()

        return smoothed_probs, (smoothed_probs, p_cross)

    # Run the HMM smoother
    carry = filtered_probs[-1]
    args = (np.arange(num_timesteps - 1, -1, -1), filtered_probs[:-1][::-1])
    _, (rev_smoothed_probs, rev_smoothed_cross) = lax.scan(_step, carry, args)

    # Reverse the arrays and return
    smoothed_probs = np.row_stack([rev_smoothed_probs[::-1], filtered_probs[-1]])
    smoothed_cross = rev_smoothed_cross[::-1]
    return log_normalizer, smoothed_probs, smoothed_cross
