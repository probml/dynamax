import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import vmap
from jax import jit
from functools import partial

from typing import Callable, Optional, Tuple, Union, NamedTuple
from jaxtyping import Int, Float, Array

from dynamax.types import Scalar, PRNGKey

_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x

def get_trans_mat(transition_matrix, transition_fn, t):
    if transition_fn is not None:
        return transition_fn(t)
    else:
        if transition_matrix.ndim == 3: # (T,K,K)
            return transition_matrix[t]
        else:
            return transition_matrix

class HMMPosteriorFiltered(NamedTuple):
    r"""Simple wrapper for properties of an HMM filtering posterior.

    :param marginal_loglik: $p(y_{1:T} \mid \theta) = \log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.
    :param filtered_probs: $p(z_t \mid y_{1:t}, \theta)$ for $t=1,\ldots,T$
    :param predicted_probs: $p(z_t \mid y_{1:t-1}, \theta)$ for $t=1,\ldots,T$

    """
    marginal_loglik: Scalar
    filtered_probs: Float[Array, "num_timesteps num_states"]
    predicted_probs: Float[Array, "num_timesteps num_states"]

class HMMPosterior(NamedTuple):
    r"""Simple wrapper for properties of an HMM posterior distribution.

    Transition probabilities may be either 2D or 3D depending on whether the
    transition matrix is fixed or time-varying.

    :param marginal_loglik: $p(y_{1:T} \mid \theta) = \log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.
    :param filtered_probs: $p(z_t \mid y_{1:t}, \theta)$ for $t=1,\ldots,T$
    :param predicted_probs: $p(z_t \mid y_{1:t-1}, \theta)$ for $t=1,\ldots,T$
    :param smoothed_probs: $p(z_t \mid y_{1:T}, \theta)$ for $t=1,\ldots,T$
    :param initial_probs: $p(z_1 \mid y_{1:T}, \theta)$ (also present in `smoothed_probs` but here for convenience)
    :param trans_probs: $p(z_t, z_{t+1} \mid y_{1:T}, \theta)$ for $t=1,\ldots,T-1$. (If the transition matrix is fixed, these probabilities may be summed over $t$. See note above.)
    """
    marginal_loglik: Scalar
    filtered_probs: Float[Array, "num_timesteps num_states"]
    predicted_probs: Float[Array, "num_timesteps num_states"]
    smoothed_probs: Float[Array, "num_timesteps num_states"]
    initial_probs: Float[Array, "num_states"]
    trans_probs: Optional[Union[Float[Array, "num_timesteps num_states num_states"],
                                Float[Array, "num_states num_states"]]] = None


def _normalize(u, axis=0, eps=1e-15):
    """Normalizes the values within the axis in a way that they sum up to 1.

    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.

    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c, c


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
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(probs, A):
    return A.T @ probs


@partial(jit, static_argnames=["transition_fn"])
def hmm_filter(
    initial_distribution: Float[Array, "num_states"],
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]] = None,
    num_timesteps: Optional[Int] = None,
) -> HMMPosteriorFiltered:
    r"""Forwards filtering

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.
        num_timesteps: number of "valid" timesteps, to support vmapping with padded arrays.

    Returns:
        filtered posterior distribution

    """
    max_num_timesteps = log_likelihoods.shape[0]
    num_timesteps = num_timesteps if num_timesteps is not None else max_num_timesteps

    def _step(carry, t):
        log_normalizer, predicted_probs = carry

        A = get_trans_mat(transition_matrix, transition_fn, t)
        ll = log_likelihoods[t]

        # Ignore observations after specified number of timesteps
        ll = jnp.where(t < num_timesteps, ll, 0.0)

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = _predict(filtered_probs, A)

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(_step, carry, jnp.arange(max_num_timesteps))

    post = HMMPosteriorFiltered(marginal_loglik=log_normalizer,
                                filtered_probs=filtered_probs,
                                predicted_probs=predicted_probs)
    return post



@partial(jit, static_argnames=["transition_fn"])
def hmm_backward_filter(
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]]= None,
    num_timesteps: Optional[Int] = None
) -> Tuple[Float, Float[Array, "num_timesteps num_states"]]:
    r"""Run the filter backwards in time. This is the second step of the forward-backward algorithm.

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.
        num_timesteps: number of "valid" timesteps, to support vmapping with padded arrays.

    Returns:
        marginal log likelihood and backward messages.

    """
    max_num_timesteps, num_states = log_likelihoods.shape
    num_timesteps = num_timesteps if num_timesteps is not None else max_num_timesteps

    def _step(carry, t):
        log_normalizer, backward_pred_probs = carry

        A = get_trans_mat(transition_matrix, transition_fn, t)
        ll = log_likelihoods[t]

        # Ignore observations after specified number of timesteps
        ll = jnp.where(t < num_timesteps, ll, 0.0)

        # Condition on emission at time t, being careful not to overflow.
        backward_filt_probs, log_norm = _condition_on(backward_pred_probs, ll)
        
        # Predict the next state (going backward in time).
        next_backward_pred_probs = _predict(backward_filt_probs, A.T)
        return (log_normalizer, next_backward_pred_probs), backward_pred_probs

    carry = (0.0, jnp.ones(num_states))
    (log_normalizer, _), backward_pred_probs = lax.scan(_step, carry, jnp.arange(max_num_timesteps), reverse=True)
    return log_normalizer, backward_pred_probs


@partial(jit, static_argnames=["transition_fn"])
def hmm_two_filter_smoother(
    initial_distribution: Float[Array, "num_states"],
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]]= None,
    num_timesteps: Optional[Int] = None,
    compute_trans_probs: bool = True
) -> HMMPosterior:
    r"""Computed the smoothed state probabilities using the two-filter
    smoother, a.k.a. the **forward-backward algorithm**.

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.
        num_timesteps: number of "valid" timesteps, to support vmapping with padded arrays.

    Returns:
        posterior distribution

    """
    # Forward
    post = hmm_filter(initial_distribution, transition_matrix, log_likelihoods, transition_fn, num_timesteps)
    ll = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

    # Backward
    _, backward_pred_probs = hmm_backward_filter(transition_matrix, log_likelihoods, transition_fn, num_timesteps)

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1, keepdims=True)
    smoothed_probs /= norm

    posterior = HMMPosterior(
        marginal_loglik=ll,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
        smoothed_probs=smoothed_probs,
        initial_probs=smoothed_probs[0]
    )

    # Compute the transition probabilities if specified
    if compute_trans_probs:
        trans_probs = compute_transition_probs(transition_matrix, posterior, transition_fn)
        posterior = posterior._replace(trans_probs=trans_probs)

    return posterior


@partial(jit, static_argnames=["transition_fn"])
def hmm_smoother(
    initial_distribution: Float[Array, "num_states"],
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]]= None,
    num_timesteps: Optional[Int]=None,
    compute_trans_probs: bool = True
) -> HMMPosterior:
    r"""Computed the smoothed state probabilities using a general
    Bayesian smoother.

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    *Note: This is the discrete SSM analog of the RTS smoother for linear Gaussian SSMs.*

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.
        num_timesteps: number of "valid" timesteps, to support vmapping with padded arrays.

    Returns:
        posterior distribution

    """
    max_num_timesteps, _ = log_likelihoods.shape
    num_timesteps = num_timesteps if num_timesteps is not None else max_num_timesteps

    # Run the HMM filter
    post = hmm_filter(initial_distribution, transition_matrix, log_likelihoods, transition_fn, num_timesteps)
    ll = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, filtered_probs, predicted_probs_next = args

        A = get_trans_mat(transition_matrix, transition_fn, t)

        # Fold in the next state (Eq. 8.2 of Saarka, 2013)
        # If hard 0. in predicted_probs_next, set relative_probs_next as 0. to avoid NaN values
        relative_probs_next = jnp.where(jnp.isclose(predicted_probs_next, 0.0), 0.0,
                                        smoothed_probs_next / predicted_probs_next)
        smoothed_probs = filtered_probs * (A @ relative_probs_next)
        smoothed_probs /= smoothed_probs.sum()
        return smoothed_probs, smoothed_probs

    # Run the HMM smoother
    carry = filtered_probs[-1]
    args = (jnp.arange(max_num_timesteps - 1), filtered_probs[:-1], predicted_probs[1:])
    _, smoothed_probs = lax.scan(_step, carry, args, reverse=True)

    # Reverse the arrays and return
    smoothed_probs = jnp.vstack([smoothed_probs, filtered_probs[-1]])

    # Package into a posterior
    posterior = HMMPosterior(
        marginal_loglik=ll,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
        smoothed_probs=smoothed_probs,
        initial_probs=smoothed_probs[0]
    )

    # Compute the transition probabilities if specified
    if compute_trans_probs:
        trans_probs = compute_transition_probs(transition_matrix, posterior, transition_fn)
        posterior = posterior._replace(trans_probs=trans_probs)

    return posterior

@partial(jit, static_argnames=["transition_fn", "window_size"])
def hmm_fixed_lag_smoother(
    initial_distribution: Float[Array, "num_states"],
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    window_size: Int,
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]]= None
) -> HMMPosterior:
    r"""Compute the smoothed state probabilities using the fixed-lag smoother.

    The smoothed probability estimates

    $$p(z_t \mid y_{1:t+L}, u_{1:t+L}, \theta)$$

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        window_size: the number of future steps to use, $L$
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        posterior distribution

    """
    # TODO: Update to allow variable length time series
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        # Unpack the inputs
        log_normalizers, filtered_probs, predicted_probs, bmatrices = carry

        # Get parameters for time t
        A_fwd = get_trans_mat(transition_matrix, transition_fn, t-1)
        A_bwd = get_trans_mat(transition_matrix, transition_fn, t)
        ll = log_likelihoods[t]

        # Shift window forward by 1
        log_normalizers = log_normalizers[1:]
        predicted_probs = predicted_probs[1:]
        filtered_probs = filtered_probs[1:]
        bmatrices = bmatrices[1:]

        # Perform forward operation
        predicted_probs_next = _predict(filtered_probs[-1], A_fwd)
        filtered_probs_next, log_norm = _condition_on(predicted_probs_next, ll)
        log_normalizers = jnp.concatenate((log_normalizers, jnp.array([log_norm])))
        filtered_probs = jnp.concatenate((filtered_probs, jnp.array([filtered_probs_next])))
        predicted_probs = jnp.concatenate((predicted_probs, jnp.array([predicted_probs_next])))

        # Smooth inside the window in parallel
        def update_bmatrix(bmatrix):
            return (bmatrix @ A_bwd) * jnp.exp(ll)

        bmatrices = vmap(update_bmatrix)(bmatrices)
        bmatrices = jnp.concatenate((bmatrices, jnp.eye(num_states)[None, :]))

        # Compute beta values by row-summing bmatrices
        def compute_beta(bmatrix):
            beta = bmatrix.sum(axis=1)
            return jnp.where(beta.sum(), beta / beta.sum(), beta)

        betas = vmap(compute_beta)(bmatrices)

        # Compute posterior values
        def compute_posterior(filtered_probs, beta):
            smoothed_probs = filtered_probs * beta
            return jnp.where(smoothed_probs.sum(), smoothed_probs / smoothed_probs.sum(), smoothed_probs)

        smoothed_probs = vmap(compute_posterior, (0, 0))(filtered_probs, betas)

        post = HMMPosterior(
            marginal_loglik=log_normalizers.sum(),
            filtered_probs=filtered_probs,
            predicted_probs=predicted_probs,
            smoothed_probs=smoothed_probs,
            initial_probs=smoothed_probs[0]
        )

        return (log_normalizers, filtered_probs, predicted_probs, bmatrices), post

    # Filter on first observation
    ll = log_likelihoods[0]
    filtered_probs, log_norm = _condition_on(initial_distribution, ll)

    # Reshape for lax.scan
    filtered_probs = jnp.pad(jnp.expand_dims(filtered_probs, axis=0), ((window_size - 1, 0), (0, 0)))
    predicted_probs = jnp.pad(jnp.expand_dims(initial_distribution, axis=0), ((window_size - 1, 0), (0, 0)))
    log_normalizers = jnp.pad(jnp.array([log_norm]), (window_size - 1, 0))
    bmatrices = jnp.pad(jnp.expand_dims(jnp.eye(num_states), axis=0), ((window_size - 1, 0), (0, 0), (0, 0)))

    carry = (log_normalizers, filtered_probs, predicted_probs, bmatrices)
    _, posts = lax.scan(_step, carry, jnp.arange(1, num_timesteps))

    # Include initial values
    marginal_loglik = jnp.concatenate((jnp.array([log_normalizers.sum()]), posts.marginal_loglik))
    predicted_probs = jnp.concatenate((jnp.expand_dims(predicted_probs, axis=0), posts.predicted_probs))
    smoothed_probs = jnp.concatenate((jnp.expand_dims(filtered_probs, axis=0), posts.smoothed_probs))
    filtered_probs = jnp.concatenate((jnp.expand_dims(filtered_probs, axis=0), posts.filtered_probs))

    return HMMPosterior(
        marginal_loglik=marginal_loglik,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
        smoothed_probs=smoothed_probs,
        initial_probs=smoothed_probs[0]
    )


@partial(jit, static_argnames=["transition_fn"])
def hmm_posterior_mode(
    initial_distribution: Float[Array, "num_states"],
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]]= None,
    num_timesteps: Optional[Int]=None,
) -> Int[Array, "num_timesteps"]:
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.
        num_timesteps: number of "valid" timesteps, to support vmapping with padded arrays.

    Returns:
        most likely state sequence

    """
    max_num_timesteps, _ = log_likelihoods.shape
    num_timesteps = num_timesteps if num_timesteps is not None else max_num_timesteps

    # Run the backward pass
    def _backward_pass(best_next_score, t):
        A = get_trans_mat(transition_matrix, transition_fn, t)

        scores = jnp.log(A) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)

        # Only update if log_likelihoods[t+1] is valid
        best_next_score = jnp.where(t + 1 < num_timesteps, best_next_score, jnp.zeros(num_states))
        best_next_state = jnp.where(t + 1 < num_timesteps, best_next_state, jnp.zeros(num_states, dtype=int))

        return best_next_score, best_next_state

    num_states = log_likelihoods.shape[1]
    best_second_score, best_next_states = lax.scan(
        _backward_pass, jnp.zeros(num_states), jnp.arange(max_num_timesteps - 1),
        reverse=True
    )
    
    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score)
    _, states = lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


@partial(jit, static_argnames=["transition_fn"])
def hmm_posterior_sample(
    rng: jr.PRNGKey,
    initial_distribution: Float[Array, "num_states"],
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]] = None,
    num_timesteps: Optional[Int] = None,
) -> Int[Array, "num_timesteps"]:
    r"""Sample a latent sequence from the posterior.

    Args:
        rng: random number generator
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        :sample of the latent states, $z_{1:T}$

    """
    max_num_timesteps, num_states = log_likelihoods.shape
    num_timesteps = num_timesteps if num_timesteps is not None else max_num_timesteps

    # Run the HMM filter
    post = hmm_filter(initial_distribution, transition_matrix, log_likelihoods, transition_fn, num_timesteps)
    log_normalizer, filtered_probs = post.marginal_loglik, post.filtered_probs

    # Run the sampler backward in time
    def _step(carry, args):
        next_state = carry
        t, rng, filtered_probs = args

        A = get_trans_mat(transition_matrix, transition_fn, t)

        # Fold in the next state and renormalize
        smoothed_probs = filtered_probs * A[:, next_state]
        smoothed_probs /= smoothed_probs.sum()

        # Sample current state
        state = jr.choice(rng, a=num_states, p=smoothed_probs)

        return state, state

    # Run the HMM smoother
    rngs = jr.split(rng, max_num_timesteps)
    last_state = jr.choice(rngs[-1], a=num_states, p=filtered_probs[-1])
    args = (jnp.arange(max_num_timesteps - 1), rngs[:-1], filtered_probs[:-1])
    _, states = lax.scan(_step, last_state, args, reverse=True)

    # Reverse the arrays and return
    states = jnp.concatenate([states, jnp.array([last_state])])
    return log_normalizer, states

def _compute_sum_transition_probs(
    transition_matrix: Float[Array, "num_states num_states"],
    hmm_posterior: HMMPosterior) -> Float[Array, "num_states num_states"]:
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
        # If hard 0. in predicted_probs_next, set relative_probs_next as 0. to avoid NaN values
        relative_probs_next = jnp.where(jnp.isclose(predicted_probs_next, 0.0), 0.0,
                                        smoothed_probs_next / predicted_probs_next)
        smoothed_trans_probs = filtered_probs[:, None] * A * relative_probs_next[None, :]
        smoothed_trans_probs /= smoothed_trans_probs.sum()
        return carry + smoothed_trans_probs, None

    # Initialize the recursion
    num_states = transition_matrix.shape[-1]
    num_timesteps = len(hmm_posterior.filtered_probs)
    sum_transition_probs, _ = lax.scan(
        _step,
        jnp.zeros((num_states, num_states)),
        (
            hmm_posterior.filtered_probs[:-1],
            hmm_posterior.smoothed_probs[1:],
            hmm_posterior.predicted_probs[1:],
            jnp.arange(num_timesteps - 1),
        ),
    )
    return sum_transition_probs


def _compute_all_transition_probs(
    transition_matrix: Float[Array, "num_timesteps num_states num_states"],
    hmm_posterior: HMMPosterior,
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]] = None
    ) -> Float[Array, "num_timesteps num_states num_states"]:
    """Compute the transition probabilities from the HMM posterior messages.
    Args:
        transition_matrix (_type_): _description_
        hmm_posterior (_type_): _description_
    """
    filtered_probs = hmm_posterior.filtered_probs[:-1]
    smoothed_probs_next = hmm_posterior.smoothed_probs[1:]
    predicted_probs_next = hmm_posterior.predicted_probs[1:]
    relative_probs_next = smoothed_probs_next / predicted_probs_next

    def _compute_probs(t):
        A = get_trans_mat(transition_matrix, transition_fn, t)
        return jnp.einsum('i,ij,j->ij', filtered_probs[t], A, relative_probs_next[t])

    transition_probs = vmap(_compute_probs)(jnp.arange(len(filtered_probs)-1))
    return transition_probs


# TODO: Consider alternative annotation for return type:
#  Float[Array, "*num_timesteps num_states num_states"] I think this would allow multiple prepended dims.
#  Float[Array, "#num_timesteps num_states num_states"] this might accept (1, sd, sd) but not (sd, sd).
def compute_transition_probs(
    transition_matrix: Union[Float[Array, "num_timesteps num_states num_states"],
                             Float[Array, "num_states num_states"]],
    hmm_posterior: HMMPosterior,
    transition_fn: Optional[Callable[[Int], Float[Array, "num_states num_states"]]] = None
) -> Union[Float[Array, "num_timesteps num_states num_states"],
            Float[Array, "num_states num_states"]]:
    r"""Compute the posterior marginal distributions $p(z_{t+1}, z_t \mid y_{1:T}, u_{1:T}, \theta)$.

    Args:
        transition_matrix: the (possibly time-varying) transition matrix
        hmm_posterior: Output of `hmm_smoother` or `hmm_two_filter_smoother`
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        array of smoothed transition probabilities.
    """
    reduce_sum = transition_matrix is not None and transition_matrix.ndim == 2
    if reduce_sum:
        return _compute_sum_transition_probs(transition_matrix, hmm_posterior)
    else:
        return _compute_all_transition_probs(transition_matrix, hmm_posterior, transition_fn=transition_fn)
