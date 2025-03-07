"""
Tests for the HMM inference functions.
"""
import itertools as it
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import dynamax.hidden_markov_model.inference as core
import dynamax.hidden_markov_model.parallel_inference as parallel

from jax.scipy.special import logsumexp

def big_log_joint(initial_probs, transition_matrix, log_likelihoods):
    """Compute the big log joint probability array

    Args:
        initial_probs (_type_): _description_
        transition_matrix (_type_): _description_
        log_lkhds (_type_): _description_
    """
    num_timesteps, num_states = log_likelihoods.shape
    flat_log_joint = jnp.zeros(num_states**num_timesteps)

    # Compute each entry in the exponentially large log joint table
    for states in it.product(*([jnp.arange(num_states)] * num_timesteps)):
        states = jnp.array(states)
        lp = jnp.log(initial_probs[states[0]])
        lp += jnp.log(transition_matrix)[states[:-1], states[1:]].sum()
        lp += log_likelihoods[jnp.arange(num_timesteps), states].sum()
        flat_index = jnp.ravel_multi_index(states, (num_states,) * num_timesteps)
        flat_log_joint = flat_log_joint.at[flat_index].set(lp)

    return flat_log_joint.reshape((num_states,) * num_timesteps)


def random_hmm_args(key, num_timesteps, num_states, scale=1.0):
    """
    Generate random arguments for the HMM functions.
    """
    k1, k2, k3 = jr.split(key, 3)
    initial_probs = jr.uniform(k1, (num_states,))
    initial_probs /= initial_probs.sum()
    transition_matrix = jr.uniform(k2, (num_states, num_states))
    transition_matrix /= transition_matrix.sum(1, keepdims=True)
    log_likelihoods = scale * jr.normal(k3, (num_timesteps, num_states))
    return initial_probs, transition_matrix, log_likelihoods

def random_hmm_args_nonstationary(key, num_timesteps, num_states, scale=1.0):
    """
    Generate random *time-varying* arguments for the HMM functions.
    """
    k1, k2, k3 = jr.split(key, 3)
    initial_probs = jr.uniform(k1, (num_states,))
    initial_probs /= initial_probs.sum()
    log_likelihoods = scale * jr.normal(k3, (num_timesteps, num_states))

    # we use numpy so we can assign to the matrix.
    # Then we convert to jnp.
    trans_mat = jnp.zeros((num_timesteps - 1, num_states, num_states))
    for t in range(num_timesteps):
      A = jr.uniform(k2, (num_states, num_states))
      A /= A.sum(1, keepdims=True)
      trans_mat = trans_mat.at[t].set(A)
    return initial_probs, jnp.array(trans_mat), log_likelihoods

def test_hmm_filter(key=0, num_timesteps=3, num_states=2):
    """
    Test the HMM filter function.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    initial_probs, transition_matrix, log_lkhds = random_hmm_args(key, num_timesteps, num_states)

    # Run the HMM filter
    post = core.hmm_filter(initial_probs, transition_matrix, log_lkhds)
    log_normalizer = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

    # Compare log_normalizer to manually computed entries
    log_joint = big_log_joint(initial_probs, transition_matrix, log_lkhds)
    assert jnp.allclose(log_normalizer, logsumexp(log_joint), atol=1e-4)

    # Compare filtered_probs to manually computed entries
    for t in range(num_timesteps):
        log_joint_t = big_log_joint(initial_probs, transition_matrix, log_lkhds[:(t + 1)])
        log_joint_t -= logsumexp(log_joint_t)
        filtered_probs_t = jnp.exp(logsumexp(log_joint_t, axis=tuple(jnp.arange(t))))
        assert jnp.allclose(filtered_probs[t], filtered_probs_t, atol=1e-4)

    # Compare predicted_probs to manually computed entries
    for t in range(num_timesteps):
        log_joint_t = big_log_joint(initial_probs, transition_matrix,
                                    jnp.vstack([log_lkhds[:t], jnp.zeros(num_states)]))

        log_joint_t -= logsumexp(log_joint_t)
        predicted_probs_t = jnp.exp(logsumexp(log_joint_t, axis=tuple(jnp.arange(t))))
        assert jnp.allclose(predicted_probs[t], predicted_probs_t, atol=1e-4)


# def test_hmm_posterior_sample(key=0, num_timesteps=5, num_states=2, eps=1e-3, num_samples=1000000, num_iterations=5):
#     if isinstance(key, int):
#         key = jr.PRNGKey(key)

#     max_unique_size = 1 << num_timesteps

#     def iterate_test(key_iter):
#         keys_iter = jr.split(key_iter, num_samples)
#         args = random_hmm_args(key_iter, num_timesteps, num_states)

#         # Sample sequences from posterior
#         state_seqs = vmap(core.hmm_posterior_sample, (0, None, None, None), (0, 0))(keys_iter, *args)[1]
#         unique_seqs, counts = jnp.unique(state_seqs, axis=0, size=max_unique_size, return_counts=True)
#         blj_sample = counts / counts.sum()

#         # Compute joint probabilities
#         blj = jnp.exp(big_log_joint(*args))
#         blj = jnp.ravel(blj / blj.sum())

#         # Compare the joint distributions
#         return jnp.allclose(blj_sample, blj, rtol=0, atol=eps)

#     keys = jr.split(key, num_iterations)
#     assert iterate_test(keys[0])


def test_two_filter_smoother(key=0, num_timesteps=5, num_states=2):
    """
    Test the two-filter smoother function.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    args = random_hmm_args(key, num_timesteps, num_states)

    # Run the HMM filter
    posterior = core.hmm_two_filter_smoother(*args)

    # Compare log_normalizer to manually computed entries
    log_joint = big_log_joint(*args)
    assert jnp.allclose(posterior.marginal_loglik, logsumexp(log_joint), atol=1e-4)

    # Compare the smooth probabilities to the manually computed ones
    joint = jnp.exp(log_joint - logsumexp(log_joint))
    for t in range(num_timesteps):
        smoothed_probs_t = jnp.sum(joint, axis=tuple(jnp.arange(t)) + tuple(jnp.arange(t + 1, num_timesteps)))
        assert jnp.allclose(posterior.smoothed_probs[t], smoothed_probs_t, atol=1e-4)


def test_hmm_smoother(key=0, num_timesteps=5, num_states=2):
    """
    Test the HMM smoother function.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    args = random_hmm_args(key, num_timesteps, num_states)

    # Run the HMM smoother
    posterior = core.hmm_smoother(*args)

    # Compare log_normalizer to manually computed entries
    log_joint = big_log_joint(*args)
    assert jnp.allclose(posterior.marginal_loglik, logsumexp(log_joint), atol=1e-4)

    # Compare the smooth probabilities to the manually computed ones
    joint = jnp.exp(log_joint - logsumexp(log_joint))
    for t in range(num_timesteps):
        smoothed_probs_t = jnp.sum(joint, axis=tuple(jnp.arange(t)) + tuple(jnp.arange(t + 1, num_timesteps)))
        assert jnp.allclose(posterior.smoothed_probs[t], smoothed_probs_t, atol=1e-4)


def test_hmm_fixed_lag_smoother(key=0, num_timesteps=5, num_states=2):
    """
    Test the HMM fixed-lag smoother function.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    args = random_hmm_args(key, num_timesteps, num_states)

    # Run the HMM smoother
    posterior = core.hmm_smoother(*args)

    # Run the HMM fixed-lag smoother with full window size
    posterior_fl = core.hmm_fixed_lag_smoother(*args, window_size=num_timesteps)

    # Compare posterior values of fixed-lag smoother to those of smoother
    assert jnp.allclose(posterior.marginal_loglik, posterior_fl.marginal_loglik[-1], atol=1e-3)
    assert jnp.allclose(posterior.filtered_probs, posterior_fl.filtered_probs[-1], atol=1e-4)
    assert jnp.allclose(posterior.predicted_probs, posterior_fl.predicted_probs[-1], atol=1e-4)
    assert jnp.allclose(posterior.smoothed_probs, posterior_fl.smoothed_probs[-1], atol=1e-3)


def test_compute_transition_probs(key=0, num_timesteps=5, num_states=2):
    """
    Test the transition probability computation function
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    args = random_hmm_args(key, num_timesteps, num_states)

    # Run the HMM smoother
    posterior = core.hmm_smoother(*args)
    sum_trans_probs = core.compute_transition_probs(args[1], posterior)

    # Compare log_normalizer to manually computed entries
    log_joint = big_log_joint(*args)
    joint = jnp.exp(log_joint - logsumexp(log_joint))

    # Compare the smooth transition probabilities to the manually computed ones
    sum_trans_probs_t = 0
    for t in range(num_timesteps - 1):
        sum_trans_probs_t += jnp.sum(joint, axis=tuple(jnp.arange(t)) + tuple(jnp.arange(t + 2, num_timesteps)))

    assert jnp.allclose(sum_trans_probs, sum_trans_probs_t, atol=1e-3)


def test_hmm_posterior_mode(key=0, num_timesteps=5, num_states=2):
    """
    Test the HMM posterior mode function.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    args = random_hmm_args(key, num_timesteps, num_states)

    # Run Viterbi
    mode = core.hmm_posterior_mode(*args)

    # Compare log_normalizer to manually computed entries and find the mode
    log_joint = big_log_joint(*args)
    mode_t = jnp.stack(jnp.unravel_index(jnp.argmax(log_joint), log_joint.shape))

    # Compare the posterior modes
    assert jnp.all(mode == mode_t)


def test_hmm_smoother_stability(key=0, num_timesteps=10000, num_states=100, scale=100.0):
    """
    Test the HMM smoother function with a large number of states and timesteps.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    args = random_hmm_args(key, num_timesteps, num_states, scale)

    # Run the HMM smoother
    posterior = core.hmm_smoother(*args)

    assert jnp.all(jnp.isfinite(posterior.smoothed_probs))
    assert jnp.allclose(posterior.smoothed_probs.sum(1), 1.0)

def test_hmm_non_stationary(key=0, num_timesteps=10, num_states=5, scale=1):
    """
    Test the HMM functions with time-varying transition matrices.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    initial_probs, transition_matrices, log_lkhds= random_hmm_args_nonstationary(key, num_timesteps, num_states)
    assert jnp.shape(transition_matrices)[0] == num_timesteps - 1
    assert jnp.shape(transition_matrices)[1] == num_states

    def trans_mat_callable(t):
        """Callable to return the transition matrix at time t."""
        return transition_matrices[t]

    # Run the HMM filter with a 3d list of transition matrices and a callable
    post = core.hmm_filter(initial_probs, transition_matrices, log_lkhds)
    post2 = core.hmm_filter(initial_probs, None, log_lkhds, trans_mat_callable)
    assert jnp.allclose(post.marginal_loglik, post2.marginal_loglik, atol=1e-4)
    assert jnp.allclose(post.filtered_probs, post2.filtered_probs, atol=1e-4)

    # Run the HMM smoother with a 3d list of transition matrices and a callable
    post = core.hmm_smoother(initial_probs, transition_matrices, log_lkhds)
    post2 = core.hmm_smoother(initial_probs, None, log_lkhds, trans_mat_callable)
    assert jnp.allclose(post.smoothed_probs, post2.smoothed_probs, atol=1e-4)

    # Run Viterbi
    mode = core.hmm_posterior_mode(initial_probs, transition_matrices, log_lkhds)
    mode2 = core.hmm_posterior_mode(initial_probs, None, log_lkhds, trans_mat_callable)
    assert jnp.allclose(mode, mode2)

    # Draw a single sample path
    key = jr.PRNGKey(0)
    ll, sample = core.hmm_posterior_sample(key, initial_probs, transition_matrices, log_lkhds)
    ll2, sample2 = core.hmm_posterior_sample(key, initial_probs, None, log_lkhds, trans_mat_callable)
    assert jnp.allclose(ll, ll2)
    assert jnp.allclose(sample, sample2)


def test_parallel_filter(key=0, num_timesteps=100, num_states=3):
    """
    Test the parallel HMM filter function
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    initial_probs, transition_matrix, log_likelihoods = \
        random_hmm_args(key, num_timesteps, num_states)

    posterior = core.hmm_filter(initial_probs, transition_matrix, log_likelihoods)
    posterior2 = parallel.hmm_filter(initial_probs, transition_matrix, log_likelihoods)
    assert jnp.allclose(posterior.marginal_loglik / num_timesteps,
                        posterior2.marginal_loglik / num_timesteps, atol=1e-3)

    assert jnp.allclose(posterior.filtered_probs, posterior2.filtered_probs, atol=1e-1)
    assert jnp.allclose(posterior.predicted_probs, posterior2.predicted_probs, atol=1e-1)


def test_parallel_smoother(key=0, num_timesteps=100, num_states=3):
    """ 
    Test the parallel HMM smoother function
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    initial_probs, transition_matrix, log_likelihoods = \
        random_hmm_args(key, num_timesteps, num_states)

    posterior = core.hmm_smoother(initial_probs, transition_matrix, log_likelihoods)
    posterior2 = parallel.hmm_smoother(initial_probs, transition_matrix, log_likelihoods)
    assert jnp.allclose(posterior.smoothed_probs, posterior2.smoothed_probs, atol=1e-1)


def test_parallel_posterior_sample(
        key=0, num_timesteps=5, num_states=2, eps=1e-3, 
        num_samples=1000000, num_iterations=5
    ):
    """
    Test the parallel HMM posterior sample function
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    max_unique_size = 1 << num_timesteps
    
    k1, k2 = jr.split(key)
    args = random_hmm_args(k1, num_timesteps, num_states)

    # Sample sequences from posterior
    state_seqs = vmap(parallel.hmm_posterior_sample, (0, None, None, None), (0, 0))(jr.split(k2, num_samples), *args)[1]
    unique_seqs, counts = jnp.unique(state_seqs, axis=0, size=max_unique_size, return_counts=True)
    blj_sample = counts / counts.sum()

    # Compute joint probabilities
    blj = jnp.exp(big_log_joint(*args))
    blj = jnp.ravel(blj / blj.sum())

    # Compare the joint distributions
    assert jnp.allclose(blj_sample, blj, rtol=0, atol=eps)
