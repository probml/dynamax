
import jax.numpy as np
from jax import lax

def hmm_filter(hmm, inputs, data):
    """_summary_

    Notes:
        p_ttm1 = p(z_t | x_{1:t-1})
        p_tt = p(z_t | x_{1:t})
        p_tp1t = p(z_{t+1} | x_{1:t})

    Args:
        hmm (_type_): _description_
        inputs (_type_): _description_
        data (_type_): _description_
    """
    T = len(data)

    def _step(carry, t):
        log_normalizer, p_ttm1 = carry

        # Get parameters and inputs for time index t
        A = hmm.transition_matrix(t, inputs[t])
        ll = hmm.log_likelihood(t, inputs[t], data[t])

        # Condition on data at time t
        # Note: be careful not to overflow!
        ll_max = ll.max()
        p_tt = p_ttm1 * np.exp(ll - ll_max)

        # Renormalize to make p_tt a distribution
        norm = p_tt.sum()
        p_tt /= norm

        # Update the log normalizer
        log_normalizer += np.log(norm) + ll_max

        # Predict the next state
        p_tp1t = A.T @ p_tt

        return (log_normalizer, p_tp1t), p_tt

    carry = (0.0, hmm.initial_dist(inputs[0]))
    (ll, _), filtered_probs = lax.scan(_step, carry, np.arange(T))
    return ll, filtered_probs


def hmm_reverse_filter(hmm, inputs, data):
    """_summary_

    Notes:
        beta_t[j]   = p(x_{t+1:T} | z_t = j)
        beta_tm1[j] = p(x_{t:T}   | z_{t-1} = j)

    Args:
        hmm (_type_): _description_
        inputs (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    T = len(data)

    def _step(carry, t):
        log_normalizer, beta_t = carry

        # Get parameters and inputs for time index t
        A = hmm.transition_matrix(t, inputs[t])
        ll = hmm.log_likelihood(t, inputs[t], data[t])

        ll_max = ll.max()
        beta_tm1 = A @ (beta_t * np.exp(ll - ll_max))

        # Renormalize
        norm = beta_tm1.sum()
        beta_tm1 /= norm

        # Update the log log_normalizer
        log_normalizer += np.log(norm) + ll_max

        return (log_normalizer, beta_tm1), beta_tm1

    carry = (0.0, np.ones(hmm.n_states))
    (ll, _), backwards_messages = lax.scan(_step, carry, np.arange(T)[::-1])
    return ll, backwards_messages


def hmm_two_filter_smoother(hmm, inputs, data):

    ll, filtered_probs = hmm_filter(hmm, inputs, data)
    _, reverse_filtered_probs = hmm_reverse_filter(hmm, inputs, data)

    # Compute E[z_t] for t = 1, ..., T
    # p_zt1T := p(z_t = j | x_{1:T})
    p_zt1T = filtered_probs * reverse_filtered_probs
    norm = p_zt1T.sum(axis=1)
    p_zt1T /= norm

    return p_zt1T, ll


def hmm_smoother(hmm, inputs, data):
    """_summary_

    Math:

    p(z_t | x_{1:T}) = \sum_{z_{t+1}} p(z_t, z_{t+1} | x_{1:T})
      = \sum_{z_{t+1}} p(z_t | x_{1:t}) p(z_{t+1} | z_t, x_{t+1:T})
      \propto p(z_t | x_{1:t}) \sum_{z_{t+1}} p(z_{t+1} | z_t) p(x_{t+1:T} | z_{t+1})
    and
      p(z_{t+1} | x_{t+1:T}, z_t) = p(x_{t+1:T} | z_{t+1}) * p(z_{t+1} | z_t) / p(x_{t+1:T} | z_t)

    Naming convention:

        p_tt = p(z_t | x_{1:t})
        p_tT = p(z_t | x_{1:T})
        p_tp1T = p(z_{t+1} | x_{1:T})
        p_cross = p(z_t, z_{t+1} | x_{1:T})

    Args:
        hmm (_type_): _description_
        inputs (_type_): _description_
        data (_type_): _description_

    Returns:
        smoothed_probs: p(z_t | x_{1:T}) for all t=1...T
        smoothed_cross: p(z_t, z_{t+1} | x_{1:T}) for all t=1...T-1
    """
    T = len(data)

    # Run the hmm filter
    ll, filtered_probs = hmm_filter(hmm, inputs, data)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        p_tp1T = carry
        t, p_tt = args

        # Get parameters and inputs for time index t
        A = hmm.transition_matrix(t, inputs[t])

        # Fold in the next state and renormalize
        p_tT = p_tt * (A @ p_tp1T)
        p_tT /= p_tT.sum()

        # Compute the cross probs p(z_t=i, z_{t+1}=j)
        p_cross = p_tt[:, None] * A * p_tp1T[None, :]
        p_cross /= p_cross.sum()

        return p_tT, (p_tT, p_cross)

    # Run the HMM smoother
    carry = filtered_probs[-1]
    args = (np.arange(T-1, -1, -1), filtered_probs[:-1][::-1])
    _, (smoothed_probs, smoothed_cross) = lax.scan(_step, carry, args)

    # Reverse the arrays and return
    smoothed_probs = np.row_stack([smoothed_probs[::-1], filtered_probs[-1]])
    smoothed_cross = smoothed_cross[::-1]
    return ll, smoothed_probs, smoothed_cross
