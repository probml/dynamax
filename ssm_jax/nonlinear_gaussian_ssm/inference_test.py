import jax.random as jr
import jax.numpy as jnp

from ssm_jax.linear_gaussian_ssm.models.linear_gaussian_ssm import LinearGaussianSSM
from ssm_jax.nonlinear_gaussian_ssm.models import NonLinearGaussianSSM


def lgssm_to_nlgssm(params):
    """Generates NonLinearGaussianSSM params from LinearGaussianSSM params

    Args:
        params: LinearGaussianSSM object

    Returns:
        nlgssm_params: NonLinearGaussianSSM object
    """
    nlgssm_params = NonLinearGaussianSSM(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_function=lambda x: params.dynamics_matrix @ x + params.dynamics_bias,
        dynamics_covariance=params.dynamics_covariance,
        emission_function=lambda x: params.emission_matrix @ x + params.emission_bias,
        emission_covariance=params.emission_covariance,
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
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_matrix=jr.normal(keys[4], (state_dim, state_dim)),
            dynamics_covariance=dynamics_covariance,
            dynamics_bias=jr.normal(keys[5], (state_dim,)),
            emission_matrix=jr.normal(keys[6], (emission_dim, state_dim)),
            emission_covariance=emission_covariance,
            emission_bias=jr.normal(keys[7], (emission_dim,)),
        )
    else:
        # Some arbitrary non-linear functions
        c_dynamics = jr.normal(keys[4], (3,))
        dynamics_function = lambda x: jnp.sin(
            c_dynamics[0] * jnp.power(x, 3) + c_dynamics[1] * jnp.square(x) + c_dynamics[2]
        )
        c_emission = jr.normal(keys[5], (2,))
        h = lambda x: jnp.cos(c_emission[0] * jnp.square(x) + c_emission[1])
        emission_function = (
            lambda x: h(x)[:emission_dim] if state_dim >= emission_dim else jnp.pad(h(x), (0, emission_dim - state_dim))
        )

        params = NonLinearGaussianSSM(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_function=dynamics_function,
            dynamics_covariance=dynamics_covariance,
            emission_function=emission_function,
            emission_covariance=emission_covariance,
        )

    # Generate random samples
    key, subkey = jr.split(subkey, 2)
    states, emissions = params.sample(key, num_timesteps)
    return params, states, emissions
