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
    """Forwards filtering.  

    Args:
        initial_distribution(k): prob(hid(1)=k)
        transition_matrix(j,k): prob(hid(t)=k | hid(t-1)=j)
        log_likelihoods(t,k): p(obs(t) | hid(t)=k)

    Returns: HMMPosterior object (smoothed_probs=None)
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
    return post


def hmm_posterior_sample(rng,
                         initial_distribution,
                         transition_matrix,
                         log_likelihoods):
    """Sample a latent sequence from the posterior.  

    Args:
        initial_distribution(k): prob(hid(1)=k)
        transition_matrix(j,k): prob(hid(t)=k | hid(t-1)=j)
        log_likelihoods(t,k): p(obs(t) | hid(t)=k)

    Returns:
        log_prob
        sampled_states(1:T)
    """
    num_timesteps, num_states = log_likelihoods.shape

    # Run the HMM filter
    post = hmm_filter(initial_distribution,
                      transition_matrix,
                      log_likelihoods)
    log_normalizer, filtered_probs = post.marginal_loglik, post.filtered_probs

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
        state = jr.choice(rng, a=num_states, p=smoothed_probs)

        return state, state

    # Run the HMM smoother
    rngs = jr.split(rng, num_timesteps)
    last_state = jr.choice(rngs[-1], a=num_states, p=filtered_probs[-1])
    args = (jnp.arange(num_timesteps - 1, 0, -1),
            rngs[:-1][::-1],
            filtered_probs[:-1][::-1])
    _, rev_states = lax.scan(_step, last_state, args)

    # Reverse the arrays and return
    states = jnp.concatenate([rev_states[::-1], jnp.array([last_state])])
    return log_normalizer, states
    

def hmm_backward_filter(transition_matrix,
                        log_likelihoods):
    """_summary_

    Args:
        hmm_params (_type_): _description_
        log_likelihoods (_type_): _description_

    Returns:
        log_marginal_lik
        backwards_probs(t,k)
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
        HMMPosterior object
    """
    num_timesteps, num_states = log_likelihoods.shape

    # Run the filters forward and backward
    post = hmm_filter(initial_distribution, transition_matrix, log_likelihoods)
    ll = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

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
        HMMPosterior object
    """
    num_timesteps, num_states = log_likelihoods.shape

    # Run the HMM filter
    post = hmm_filter(initial_distribution, transition_matrix, log_likelihoods)
    ll = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

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
        map_state_seq(1:T)
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

