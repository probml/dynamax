import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
import chex
import ssm_jax.hmm.learning as learn
from ssm_jax.hmm.models import GaussianHMM

# =============================================================================
# "STANDARD" EM steps for Gaussian HMM
#   This perfroms out-of-place HMM updates (i.e. a new HMM is returned)
#   These results are used as reference for test cases
# =============================================================================
from functools import partial
from jax import vmap, tree_map
from ssm_jax.hmm.inference import hmm_smoother, compute_transition_probs
from tensorflow_probability.substrates.jax.distributions import Dirichlet

@chex.dataclass
class GaussianHMMSuffStats:
    # Wrapper for sufficient statistics of a GaussianHMM
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    sum_w: chex.Array
    sum_x: chex.Array
    sum_xxT: chex.Array
    
def standard_e_step(hmm: GaussianHMM, batch_emissions: chex.Array) -> GaussianHMMSuffStats:
    def _single_e_step(emissions):
        # Run the smoother
        posterior = hmm_smoother(hmm.initial_probabilities,
                                 hmm.transition_matrix,
                                 hmm._conditional_logliks(emissions))

        # Compute the initial state and transition probabilities
        initial_probs = posterior.smoothed_probs[0]
        trans_probs = compute_transition_probs(hmm.transition_matrix, posterior)

        # Compute the expected sufficient statistics
        sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
        sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, emissions)
        sum_xxT = jnp.einsum("tk, ti, tj->kij", posterior.smoothed_probs, emissions, emissions)

        stats = GaussianHMMSuffStats(
            marginal_loglik=posterior.marginal_loglik,
            initial_probs=initial_probs,
            trans_probs=trans_probs,
            sum_w=sum_w,
            sum_x=sum_x,
            sum_xxT=sum_xxT
        )
        return stats

    # Map the E step calculations over batches
    return vmap(_single_e_step)(batch_emissions)

def standard_m_step(batch_stats: GaussianHMMSuffStats) -> GaussianHMM:
    # Sum the statistics across all batches
    stats = tree_map(partial(jnp.sum, axis=0), batch_stats)

    # Initial distribution
    initial_probs = Dirichlet(1.0001 + stats.initial_probs).mode()

    # Transition distribution
    transition_matrix = Dirichlet(1.0001 + stats.trans_probs).mode()

    # Gaussian emission distribution
    emission_dim = stats.sum_x.shape[-1]
    emission_means = stats.sum_x / stats.sum_w[:, None]
    emission_covs = (
        stats.sum_xxT / stats.sum_w[:, None, None]
        - jnp.einsum("ki,kj->kij", emission_means, emission_means)
        + 1e-4 * jnp.eye(emission_dim)
    )

    # Pack the results into a new GaussianHMM
    return GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)

# =============================================================================
# Setup
# =============================================================================
                 
def make_rnd_hmm(num_states=5, emission_dim=2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    # Make a true HMM
    true_hmm = GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)

    return true_hmm


def make_rnd_model_and_data(num_states=5, emission_dim=2, num_timesteps=2000):
    true_hmm = make_rnd_hmm(num_states, emission_dim)
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)
    batch_emissions = emissions[None, ...]
    return true_hmm, true_states, batch_emissions


def test_loglik():
    true_hmm, true_states, batch_emissions = make_rnd_model_and_data()
    assert jnp.allclose(true_hmm.log_prob(true_states, batch_emissions[0]), 3149.1013, atol=1e-1)
    assert jnp.allclose(true_hmm.marginal_log_prob(batch_emissions[0]), 3149.1047, atol=1e-1)


def test_hmm_fit_em(num_iters=2):
    true_hmm, _, batch_emissions = make_rnd_model_and_data()
    test_hmm_em = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    # Quick test: 2 iterations
    test_hmm_em, logprobs_em = learn.hmm_fit_em(test_hmm_em, batch_emissions, num_iters=num_iters)
    assert jnp.allclose(logprobs_em[-1], -3600.2395, atol=1e-1)
    mu = test_hmm_em.emission_means.value
    assert jnp.alltrue(mu.shape == (10, 2))
    assert jnp.allclose(mu[0, 0], -0.712, atol=1e-1)

def test_hmm_fit_normd_vs_standard(num_iters=4):    
    true_hmm, _, batch_emissions = make_rnd_model_and_data()
    batch_emissions = batch_emissions.reshape(4, -1, true_hmm.num_obs)

    # Quick test: 2 iterations
    test_hmm_em = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    test_hmm_em, logprobs_em = learn.hmm_fit_em(test_hmm_em, batch_emissions, num_iters=num_iters)
    
    # Compute reference values from "standard" formulation
    ref_hmm_em = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    for _ in range(num_iters):
        ref_batch_stats = standard_e_step(ref_hmm_em, batch_emissions)
        ref_hmm_em = standard_m_step(ref_batch_stats)
    ref_logprobs = ref_batch_stats.marginal_loglik.sum()

    assert jnp.allclose(logprobs_em[-1], ref_logprobs, atol=1)
    mu = np.array(test_hmm_em.emission_means)
    assert jnp.alltrue(mu.shape == (10, 2))
    assert jnp.allclose(mu[0, 0], ref_hmm_em.emission_means[0,0], atol=1e-3)

def test_hmm_fit_sgd(num_iters=2):
    true_hmm, _, batch_emissions = make_rnd_model_and_data()
    print(batch_emissions.shape)
    test_hmm_sgd = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    # Quick test: 2 iterations
    optimizer = optax.adam(learning_rate=1e-2)
    test_hmm_sgd, losses = learn.hmm_fit_sgd(test_hmm_sgd, batch_emissions, optimizer=optimizer, num_iters=num_iters)
    assert jnp.allclose(losses[-1], 2.852, atol=1e-1)
    mu = test_hmm_sgd.emission_means.value
    assert jnp.alltrue(mu.shape == (10, 2))
    assert jnp.allclose(mu[0, 0], -1.827, atol=1e-1)
