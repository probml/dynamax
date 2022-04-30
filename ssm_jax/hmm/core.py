import jax.numpy as jnp
import jax.random as jr
from jax import lax


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
        log_normalizer, pred_probs = carry

        # Get parameters for time t
        get = lambda x, ndim: x[t] if x.ndim == ndim + 1 else x
        A = get(transition_matrix, 2)
        ll = log_likelihoods[t]

        # Condition on data at time t, being careful not to overflow.
        ll_max = ll.max()
        filtered_probs = pred_probs * jnp.exp(ll - ll_max)

        # Renormalize to make filtered_probs a distribution
        norm = filtered_probs.sum()
        filtered_probs /= norm

        # Update the log normalizer
        log_normalizer += jnp.log(norm) + ll_max

        # Predict the next state
        pred_probs = A.T @ filtered_probs

        return (log_normalizer, pred_probs), filtered_probs

    num_timesteps = len(log_likelihoods)
    carry = (0.0, initial_distribution)
    (log_normalizer, _), filtered_probs = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
    return log_normalizer, filtered_probs


def hmm_posterior_sample(rng,
                         initial_distribution,
                         transition_matrix,
                         log_likelihoods):
    # Run the HMM filter
    log_normalizer, filtered_probs = hmm_filter(initial_distribution,
                                                transition_matrix,
                                                log_likelihoods)

    # Run the sampler backward in time
    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        t, rng, filtered_probs = args

        # Get parameters for time t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)

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

        ll_max = ll.max()
        backward_pred_probs = A @ (backward_pred_probs * jnp.exp(ll - ll_max))

        # Renormalize
        norm = backward_pred_probs.sum()
        backward_pred_probs /= norm

        # Update the log log_normalizer
        log_normalizer += jnp.log(norm) + ll_max

        return (log_normalizer, backward_pred_probs), backward_pred_probs

    num_timesteps, num_states = log_likelihoods.shape
    carry = (0.0, jnp.ones(num_states))
    (log_normalizer, _), backward_pred_probs = lax.scan(
        _step, carry, jnp.arange(num_timesteps)[::-1])
    return log_normalizer, backward_pred_probs


def hmm_two_filter_smoother(initial_distribution,
                            transition_matrix,
                            log_likelihoods):
    # Run the filters forward and backward
    ll, filtered_probs = hmm_filter(initial_distribution,
                                    transition_matrix,
                                    log_likelihoods)

    _, backward_pred_probs = hmm_backward_filter(transition_matrix,
                                                 log_likelihoods)

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1)
    smoothed_probs /= norm

    return smoothed_probs, ll


def hmm_smoother(initial_distribution,
                 transition_matrix,
                 log_likelihoods):
    """_summary_

    Args:
        hmm_params (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Run the HMM filter
    log_normalizer, filtered_probs = hmm_filter(initial_distribution,
                                                transition_matrix,
                                                log_likelihoods)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, filtered_probs = args

        # Get parameters for time t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)

        # Fold in the next state and renormalize
        smoothed_probs = filtered_probs * (A @ smoothed_probs_next)
        smoothed_probs /= smoothed_probs.sum()

        # Compute p(z_t=i, z_{t+1}=j)
        p_cross = filtered_probs[:, None] * A * smoothed_probs_next[None, :]
        p_cross /= p_cross.sum()

        return smoothed_probs, (smoothed_probs, p_cross)

    # Run the HMM smoother
    num_timesteps = len(log_likelihoods)
    carry = filtered_probs[-1]
    args = (jnp.arange(num_timesteps - 1, -1, -1), filtered_probs[:-1][::-1])
    _, (rev_smoothed_probs, rev_smoothed_cross) = lax.scan(_step, carry, args)

    # Reverse the arrays and return
    smoothed_probs = jnp.row_stack([rev_smoothed_probs[::-1], filtered_probs[-1]])
    smoothed_cross = rev_smoothed_cross[::-1]
    return log_normalizer, smoothed_probs, smoothed_cross


def hmm_posterior_mode(initial_distribution,
                       transition_matrix,
                       log_likelihoods):

    # Run the backward pass
    def _backward_pass(best_next_score, t):
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)

        scores = jnp.log(A) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    best_second_score, best_next_states = \
        lax.scan(_backward_pass,
                 jnp.zeros(num_states),
                 jnp.arange(num_timesteps - 2, -1, -1))

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score)
    _, states = lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])
