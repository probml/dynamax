from jax import random as jr
from jax import numpy as jnp
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from ssm_jax.lgssm.models import LGSSMParams, lgssm_joint_sample
from ssm_jax.lgssm.inference import lgssm_filter

def tfp_filter(timesteps, A, transition_noise_scale, C, observation_noise_scale, mu0, x_hist):
    """ Perform filtering using tensorflow probability """
    state_size, _ = A.shape
    observation_size, _ = C.shape
    transition_noise = tfd.MultivariateNormalDiag(
        scale_diag=jnp.ones(state_size) * transition_noise_scale
    )
    obs_noise = tfd.MultivariateNormalDiag(
        scale_diag=jnp.ones(observation_size) * observation_noise_scale
    )
    prior = tfd.MultivariateNormalDiag(mu0, tf.ones([state_size]))

    LGSSM = tfd.LinearGaussianStateSpaceModel(
        timesteps, A, transition_noise, C, obs_noise, prior
    )

    _, filtered_means, filtered_covs, _, _, _, _ = LGSSM.forward_filter(x_hist)
    return filtered_means.numpy(), filtered_covs.numpy()


def test_kalman_filter():
    ### LDS Parameters ###
    state_size = 2
    observation_size  = 2
    F = jnp.eye(state_size)
    H = jnp.eye(observation_size)

    G = jnp.zeros((state_size,1))
    J = jnp.zeros((observation_size,1))

    transition_noise_scale = 1.0
    observation_noise_scale = 1.0
    Q = jnp.eye(state_size) * transition_noise_scale
    R = jnp.eye(observation_size) * observation_noise_scale


    ### Prior distribution params ###
    mu0 = jnp.array([8, 10]).astype(float)
    Sigma0 = jnp.eye(state_size) * 1.0

    ### Sample data ###
    lgssm = LGSSMParams(initial_mean = mu0,
                        initial_covariance = Sigma0,
                        dynamics_matrix = F,
                        dynamics_input_weights = G,
                        dynamics_covariance = Q,
                        emission_matrix = H,
                        emission_input_weights = J,
                        emission_covariance = R)

    key = jr.PRNGKey(111)
    num_timesteps = 15 

    inputs = jnp.zeros((num_timesteps,1))
    x, y = lgssm_joint_sample(key,lgssm,num_timesteps,inputs)

    ssm_ll_filt, ssm_filtered_means, ssm_filtered_covs = lgssm_filter(lgssm, inputs, y)
    tfp_filtered_means, tfp_filtered_covs = tfp_filter(
        num_timesteps, F, transition_noise_scale, H, observation_noise_scale, mu0, y
    )

    assert jnp.allclose(ssm_filtered_means, tfp_filtered_means, rtol=1e-2)
    assert jnp.allclose(ssm_filtered_covs, tfp_filtered_covs, rtol=1e-2)
