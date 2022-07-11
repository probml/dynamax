import jax.random as jr
import jax.numpy as jnp

from ssm_jax.lgssm.inference import lgssm_filter, lgssm_smoother
from ssm_jax.lgssm.models import LinearGaussianSSM
from ssm_jax.ekf.inference import extended_kalman_filter, extended_kalman_smoother
from ssm_jax.nlgssm.models import NonLinearGaussianSSM
from ssm_jax.nlgssm.sarkka_lib import ekf, eks

from filterpy.kalman import ExtendedKalmanFilter


# Test closeness with slightly more lenient tolerance than the default
_all_close = lambda x, y: jnp.allclose(x, y, rtol=1e-3)


# Helper function to turn linear transform into function form
def _lgssm_to_nlgssm(params):
    """Generates NonLinearGaussianSSM params from LinearGaussianSSM params

    Args:
        params: LinearGaussianSSM object

    Returns:
        nlgssm_params: NonLinearGaussianSSM object
    """    
    nlgssm_params = NonLinearGaussianSSM(
        initial_mean = params.initial_mean,
        initial_covariance = params.initial_covariance,
        dynamics_function = lambda x: params.dynamics_matrix @ x + params.dynamics_bias,
        dynamics_covariance = params.dynamics_covariance,
        emission_function = lambda x: params.emission_matrix @ x + params.emission_bias,
        emission_covariance = params.emission_covariance
    )
    return nlgssm_params


def random_args(key=0, num_timesteps=15, state_dim=4, emission_dim=2, linear=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    *keys, subkey = jr.split(key, 9)
    
    # Generate random parameters
    initial_mean = jr.normal(keys[0], (state_dim,))
    initial_covariance = jnp.eye(state_dim) * jr.uniform(keys[1])
    dynamics_covariance = jnp.eye(state_dim) * jr.uniform(keys[2])
    emission_covariance = jnp.eye(emission_dim) * jr.uniform(keys[3])

    if linear:
        params = LinearGaussianSSM(
            initial_mean = initial_mean,
            initial_covariance = initial_covariance,
            dynamics_matrix = jr.normal(keys[4], (state_dim, state_dim)),
            dynamics_covariance = dynamics_covariance,
            dynamics_bias = jr.normal(keys[5], (state_dim,)),
            emission_matrix = jr.normal(keys[6], (emission_dim, state_dim)),
            emission_covariance = emission_covariance,
            emission_bias = jr.normal(keys[7], (emission_dim,))
        )
    else:
        # Some arbitrary non-linear functions
        c_dynamics = jr.normal(keys[4], (3,))
        dynamics_function = lambda x: jnp.sin(c_dynamics[0] * jnp.power(x, 3) + \
            c_dynamics[1] * jnp.square(x) + c_dynamics[2])
        c_emission = jr.normal(keys[5], (2,))
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
    ekf_post = extended_kalman_filter(_lgssm_to_nlgssm(lgssm), emissions)

    # Compare filter results
    assert _all_close(kf_post.marginal_loglik, ekf_post.marginal_loglik)
    assert _all_close(kf_post.filtered_means, ekf_post.filtered_means)
    assert _all_close(kf_post.filtered_covariances, ekf_post.filtered_covariances)


def test_extended_kalman_filter_nonlinear(key=0, num_timesteps=15):
    nlgssm, _, emissions = \
        random_args(key=key, num_timesteps=num_timesteps, linear=False)
    
    # Run EKF from sarkka-jax library
    means_ext, covs_ext = ekf(*(nlgssm.return_params), emissions)
    # Run EKF from SSM-Jax
    ekf_post = extended_kalman_filter(nlgssm, emissions)

    # Compare filter results
    assert _all_close(means_ext, ekf_post.filtered_means)
    assert _all_close(covs_ext, ekf_post.filtered_covariances)


def test_extended_kalman_smoother_linear(key=0, num_timesteps=15):
    lgssm, _, emissions = \
        random_args(key=key, num_timesteps=num_timesteps, linear=True)
    
    # Run standard Kalman smoother
    kf_post = lgssm_smoother(lgssm, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_smoother(_lgssm_to_nlgssm(lgssm), emissions)

    # Compare smoother results
    assert _all_close(kf_post.smoothed_means, ekf_post.smoothed_means)
    assert _all_close(kf_post.smoothed_covariances, ekf_post.smoothed_covariances)


def test_extended_kalman_smoother_nonlinear(key=0, num_timesteps=15):
    nlgssm, _, emissions = \
        random_args(key=key, num_timesteps=num_timesteps, linear=False)
    
    # Run EK smoother from sarkka-jax library
    means_ext, covs_ext = eks(*(nlgssm.return_params), emissions)
    # Run EK smoother from SSM-Jax
    ekf_post = extended_kalman_smoother(nlgssm, emissions)

    # Compare filter results
    assert _all_close(means_ext, ekf_post.smoothed_means)
    assert _all_close(covs_ext, ekf_post.smoothed_covariances)