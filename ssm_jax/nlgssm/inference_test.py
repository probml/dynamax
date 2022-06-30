import jax.random as jr
import jax.numpy as jnp

from ssm_jax.lgssm.inference import lgssm_filter
from ssm_jax.lgssm.models import LinearGaussianSSM
from ssm_jax.nlgssm.extended_inference import extended_kalman_filter
from ssm_jax.nlgssm.models import NLGSSMParams


def random_linear_args(key=0, num_timesteps=15, state_dim=4, emission_dim=2):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    *keys, subkey = jr.split(key, 9)
    
    # Generate random parameters
    #   (Could use lgssm.random_initialization(...) instead
    #   if the code is completed.)
    lgssm = LinearGaussianSSM(
        initial_mean = jr.normal(keys[0], (state_dim,)),
        initial_covariance = jnp.eye(state_dim) * jr.uniform(keys[1]),
        dynamics_matrix = jr.normal(keys[2], (state_dim, state_dim)),
        dynamics_covariance = jnp.eye(state_dim) * jr.uniform(keys[3]),
        dynamics_bias = jr.normal(keys[4], (state_dim,)),
        emission_matrix = jr.normal(keys[5], (emission_dim, state_dim)),
        emission_covariance = jnp.eye(emission_dim) * jr.uniform(keys[6]),
        emission_bias = jr.normal(keys[7], (emission_dim,))
    )

    # Generate random samples
    key, subkey = jr.split(subkey, 2)
    states, emissions = lgssm.sample(key, num_timesteps)
    return lgssm, states, emissions


def test_extended_kalman_filter_linear(key=0, num_timesteps=15):
    lgssm, states, emissions = \
        random_linear_args(key=key, num_timesteps=num_timesteps)
    
    # Run standard Kalman filter
    kf_post = lgssm_filter(lgssm, emissions)
    # Run extended Kalman filter
    nlgssm = NLGSSMParams(
        initial_mean = lgssm.initial_mean,
        initial_covariance = lgssm.initial_covariance,
        dynamics_function = lambda x: lgssm.dynamics_matrix @ x + lgssm.dynamics_bias,
        dynamics_covariance = lgssm.dynamics_covariance,
        emission_function = lambda x: lgssm.emission_matrix @ x + lgssm.emission_bias,
        emission_covariance = lgssm.emission_covariance
    )
    ekf_post = extended_kalman_filter(nlgssm, emissions)

    # Compare filter results
    assert jnp.isclose(kf_post.marginal_loglik, ekf_post.marginal_loglik)
    assert jnp.allclose(kf_post.filtered_means, ekf_post.filtered_means)
    assert jnp.allclose(kf_post.filtered_covariances, ekf_post.filtered_covariances)
