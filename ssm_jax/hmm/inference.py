import jax.numpy as jnp
import jax.random as jr
from jax import lax

import chex

@chex.dataclass
class HMMPosterior:
    """Simple wrapper for properties of an HMM posterior distribution.

    marginal_loglik: log sum_{hidden(1:t)} prob(hidden(1:t), obs(1:t) | params)
    filtered_probs(t,k) = p(hidden(t)=k | obs(1:t))
    predicted_probs(t,k) = p(hidden(t+1)=k | obs(1:t)) // one-step-ahead
    smoothed_probs(t,k) = p(hidden(t)=k | obs(1:T))
    """
    marginal_loglik: chex.Scalar = None
    filtered_probs: chex.Array = None
    predicted_probs: chex.Array = None
    smoothed_probs: chex.Array = None


# Helper function to access parameters
_get_params = lambda x, dim, t: x[t] if x.ndim == dim+1 else x


# Helper functions for the two key filtering steps
def _condition_on(probs, ll):
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state, while avoiding numerical underflow.

    Args:
        probs(k): prior for state k 
        ll(k): log likelihood for state k

    Returns:
        probs(k): posterior for state k 
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
        initial_distribution(k): prob(hid(1)=k)
        transition_matrix(j,k): prob(hid(t)=k | hid(t-1)=j)
        log_likelihoods(t,k): p(obs(t) | hid(t)=k)

    Returns:
        marginal_loglik: log sum_{hidden(1:t)} prob(hidden(1:t), obs(1:t) | params)
        filtered_probs(t,k) = p(hidden(t)=k | obs(1:t))
        predicted_probs(t,k) = p(hidden(t+1)=k | obs(1:t)) // one-step-ahead
    """
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, predicted_probs = carry

        # Get parameters for time t
        A = _get_params(transition_matrix, 2, t)
        ll = log_likelihoods[t]

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        # Update the log normalizer
        log_normalizer += log_norm
        # Predict the next state
        predicted_probs_next = _predict(filtered_probs, A)

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
        
    post = HMMPosterior(marginal_loglik = log_normalizer,
                        filtered_probs = filtered_probs,
                        predicted_probs = predicted_probs)
    return log_normalizer, filtered_probs, predicted_probs


def hmm_posterior_sample(rng,
                         initial_distribution,
                         transition_matrix,
                         log_likelihoods):
    num_timesteps, num_states = log_likelihoods.shape

    # Run the HMM filter
    log_normalizer, filtered_probs, _ = hmm_filter(initial_distribution,
                                                   transition_matrix,
                                                   log_likelihoods)

    # Run the sampler backward in time
    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        t, rng, filtered_probs = args

        # Get parameters for time t
        A = _get_params(transition_matrix, 2, t)

        # Fold in the next state and renormalize
        smoothed_probs = filtered_probs * A[:, next_state]
        smoothed_probs /= smoothed_probs.sum()

        # Sample current state
        state = jr.choice(rng, p=smoothed_probs)

        return state, state

    # Run the HMM smoother
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
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, backward_pred_probs = carry

        # Get parameters for time t
        A = _get_params(transition_matrix, 2, t)
        ll = log_likelihoods[t]

        # Condition on emission at time t, being careful not to overflow.
        backward_filt_probs, log_norm = _condition_on(backward_pred_probs, ll)
        # Update the log normalizer.
        log_normalizer += log_norm
        # Predict the next state (going backward in time).
        next_backward_pred_probs = _predict(backward_filt_probs, A.T)
        return (log_normalizer, next_backward_pred_probs), backward_pred_probs

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
    num_timesteps, num_states = log_likelihoods.shape

    # Run the filters forward and backward
    ll, filtered_probs, predicted_probs = hmm_filter(initial_distribution,
                                                     transition_matrix,
                                                     log_likelihoods)

    _, backward_pred_probs = hmm_backward_filter(transition_matrix,
                                                 log_likelihoods)

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1, keepdims=True)
    smoothed_probs /= norm

    return HMMPosterior(marginal_loglik=ll,
                        filtered_probs=filtered_probs,
                        predicted_probs=predicted_probs,
                        smoothed_probs=smoothed_probs)


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
    num_timesteps, num_states = log_likelihoods.shape

    # Run the HMM filter
    ll, filtered_probs, predicted_probs = hmm_filter(initial_distribution,
                                                     transition_matrix,
                                                     log_likelihoods)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, filtered_probs, predicted_probs_next = args

        # Get parameters for time t
        A = _get_params(transition_matrix, 2, t)

        # Fold in the next state (Eq. 8.2 of Saarka, 2013)
        relative_probs_next = smoothed_probs_next / predicted_probs_next
        smoothed_probs = filtered_probs * (A @ relative_probs_next)
        smoothed_probs /= smoothed_probs.sum()

        return smoothed_probs, smoothed_probs

    # Run the HMM smoother
    carry = filtered_probs[-1]
    args = (jnp.arange(num_timesteps - 2, -1, -1),
            filtered_probs[:-1][::-1],
            predicted_probs[1:][::-1])
    _, rev_smoothed_probs = lax.scan(_step, carry, args)

    # Reverse the arrays and return
    smoothed_probs = jnp.row_stack([rev_smoothed_probs[::-1], filtered_probs[-1]])

    return HMMPosterior(marginal_loglik=ll,
                        filtered_probs=filtered_probs,
                        predicted_probs=predicted_probs,
                        smoothed_probs=smoothed_probs)


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
    num_timesteps, num_states = log_likelihoods.shape

    # Run the backward pass
    def _backward_pass(best_next_score, t):
        A = _get_params(transition_matrix, 2, t)

        scores = jnp.log(A) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_states = log_likelihoods.shape[1]
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


def _compute_sum_transition_probs(transition_matrix, hmm_posterior):
    """Compute the transition probabilities from the HMM posterior messages.

    Args:
        transition_matrix (_type_): _description_
        hmm_posterior (_type_): _description_
    """
    def _step(carry, args):
        filtered_probs, smoothed_probs_next, predicted_probs_next, t = args

        # Get parameters for time t
        A = _get_params(transition_matrix, 2, t)

        # Compute smoothed transition probabilities (Eq. 8.4 of Saarka, 2013)
        relative_probs_next = smoothed_probs_next / predicted_probs_next
        smoothed_trans_probs = filtered_probs[:, None] * A * relative_probs_next[None, :]
        smoothed_trans_probs /= smoothed_trans_probs.sum()
        return carry + smoothed_trans_probs, None

    # Initialize the recursion
    num_states = transition_matrix.shape[-1]
    num_timesteps = len(hmm_posterior.filtered_probs)
    sum_transition_probs, _ = lax.scan(_step, jnp.zeros((num_states, num_states)),
                                       (hmm_posterior.filtered_probs[:-1],
                                        hmm_posterior.smoothed_probs[1:],
                                        hmm_posterior.predicted_probs[1:],
                                        jnp.arange(num_timesteps - 1)))
    return sum_transition_probs


def _compute_all_transition_probs(transition_matrix, hmm_posterior):
    """Compute the transition probabilities from the HMM posterior messages.

    Args:
        transition_matrix (_type_): _description_
        hmm_posterior (_type_): _description_
    """
    filtered_probs = hmm_posterior.filtered_probs[:-1]
    smoothed_probs_next = hmm_posterior.smoothed_probs[1:]
    predicted_probs_next = hmm_posterior.predicted_probs[1:]
    relative_probs_next =  smoothed_probs_next / predicted_probs_next
    transition_probs = filtered_probs[:, :, None] * \
        transition_matrix * relative_probs_next[:, None, :]
    return transition_probs


def compute_transition_probs(transition_matrix, hmm_posterior, reduce_sum=True):
    """Computer the posterior marginal distributions over (hid(t), hid(t+1)),
    ..math:
        q_{tij} = Pr(z_t=i, z_{t+1}=j | obs_{1:T})  for t=1,...,T-1

    If `reduce_sum` is True, return :math:`\sum_t q_{tij}`.

    Args:
        transition_matrix (array): the transition matrix
        hmm_posterior (HMMPosterior): Output of `hmm_smoother` or `hmm_two_filter_smoother`
        reduce_sum (bool, optional): Whether or not to return the
            sum of transition probabilities over time. Defaults to True, which is
            more memory efficient.

    Returns:
        array of transition probabilities. The shape is (num_states, num_states) if
            reduce_sum==True, otherwise (num_timesteps, num_states, num_states).
    """
    if reduce_sum:
        return _compute_sum_transition_probs(transition_matrix, hmm_posterior)
    else:
        return _compute_all_transition_probs(transition_matrix, hmm_posterior)
