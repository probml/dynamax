
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


def hmm_backwards(hmm, inputs, data):
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


def hmm_smooth(hmm, inputs, data):
    
    ll, filtered_probs = hmm_filter(hmm, inputs, data)
    _, betas = hmm_backwards(hmm, inputs, data)
    
    # Compute E[z_t] for t = 1, ..., T
    # p_zt1T := p(z_t = j | x_{1:T})
    p_zt1T = filtered_probs * betas
    norm = p_zt1T.sum(axis=1)
    p_zt1T /= norm
    
    return p_zt1T, ll
