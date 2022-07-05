import jax.random as jr
import jax.numpy as jnp

from ssm_jax.lgssm.inference import lgssm_filter
from ssm_jax.lgssm.models import LinearGaussianSSM
from ssm_jax.ekf.inference import extended_kalman_filter
from ssm_jax.nlgssm.models import NonLinearGaussianSSM

from filterpy.kalman import ExtendedKalmanFilter


def random_args(key=0, num_timesteps=15, state_dim=4, emission_dim=2, linear=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    *keys, subkey = jr.split(key, 5)
    
    # Generate random parameters
    initial_mean = jr.normal(keys[0], (state_dim,))
    initial_covariance = jnp.eye(state_dim) * jr.uniform(keys[1])
    dynamics_covariance = jnp.eye(state_dim) * jr.uniform(keys[2])
    emission_covariance = jnp.eye(emission_dim) * jr.uniform(keys[3])

    if linear:
        *keys, subkey = jr.split(subkey, 5)
        params = LinearGaussianSSM(
            initial_mean = initial_mean,
            initial_covariance = initial_covariance,
            dynamics_matrix = jr.normal(keys[0], (state_dim, state_dim)),
            dynamics_covariance = dynamics_covariance,
            dynamics_bias = jr.normal(keys[1], (state_dim,)),
            emission_matrix = jr.normal(keys[2], (emission_dim, state_dim)),
            emission_covariance = emission_covariance,
            emission_bias = jr.normal(keys[3], (emission_dim,))
        )
    else:
        *keys, subkey = jr.split(subkey, 3)
        # Some arbitrary non-linear functions
        c_dynamics = jr.normal(keys[0], (3,))
        dynamics_function = lambda x: jnp.sin(c_dynamics[0] * jnp.power(x, 3) + \
            c_dynamics[1] * jnp.square(x) + c_dynamics[2])
        c_emission = jr.normal(keys[1], (2,))
        h = lambda x: jnp.cos(c_emission[0] * jnp.square(x) + c_emission[1])
        emission_function = lambda x: h(x)[:emission_dim] if state_dim >= emission_dim \
            else jnp.pad(h(x), (0, emission_dim - state_dim))

        params = NonLinearGaussianSSM(
            initial_mean = initial_mean,
            initial_covariance = initial_covariance,
            dynamics_function = dynamics_function,
            dynamics_covariance = dynamics_covariance,
            emission_function = emission_function,
            emission_covariance = emission_covariance
        )

    # Generate random samples
    key, subkey = jr.split(subkey, 2)
    states, emissions = params.sample(key, num_timesteps)
    return params, states, emissions


def test_extended_kalman_filter_linear(key=0, num_timesteps=15):
    lgssm, _, emissions = \
        random_args(key=key, num_timesteps=num_timesteps, linear=True)
    
    # Run standard Kalman filter
    kf_post = lgssm_filter(lgssm, emissions)
    # Run extended Kalman filter
    nlgssm = NonLinearGaussianSSM(
        initial_mean = lgssm.initial_mean,
        initial_covariance = lgssm.initial_covariance,
        dynamics_function = lambda x: lgssm.dynamics_matrix @ x + lgssm.dynamics_bias,
        dynamics_covariance = lgssm.dynamics_covariance,
        emission_function = lambda x: lgssm.emission_matrix @ x + lgssm.emission_bias,
        emission_covariance = lgssm.emission_covariance
    )
    ekf_post = extended_kalman_filter(nlgssm, emissions)

    # Compare filter results
    assert jnp.allclose(kf_post.marginal_loglik, ekf_post.marginal_loglik)
    assert jnp.allclose(kf_post.filtered_means, ekf_post.filtered_means)
    assert jnp.allclose(kf_post.filtered_covariances, ekf_post.filtered_covariances)


def test_extended_kalman_filter_nonlinear(key=0, num_timesteps=15):
    nlgssm, _, emissions = \
        random_args(key=key, num_timesteps=num_timesteps, linear=False)
    
    # Run EKF from filterpy library
    