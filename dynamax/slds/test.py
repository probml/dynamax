import dynamax
import jax.numpy as jnp
import jax.random as jr
import inference
from functools import partial

from jax import vmap, tree_map, jit
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, make_lgssm_params
# import MVN from tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


num_states = 3
num_particles = 10
state_dim = 4
emission_dim = 4

TT = 0.1
A = jnp.array([[1, TT, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, TT],
            [0, 0, 0, 1]])


B1 = jnp.array([0, 0, 0, 0])
B2 = jnp.array([-1.225, -0.35, 1.225, 0.35])
B3 = jnp.array([1.225, 0.35,  -1.225,  -0.35])
B = jnp.stack([B1, B2, B3], axis=0)

Q = 0.2 * jnp.eye(4)
R = 10 * jnp.diag(jnp.array([2, 1, 2, 1]))
C = jnp.eye(4)


transition_matrix = jnp.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])


discr_params = inference.DiscreteParamsSLDS(
    initial = jnp.ones(num_states)/num_states,
    transition_matrix=transition_matrix,
    proposal_transition_matrix=transition_matrix
)

params_lds1 = make_lgssm_params(initial_mean = jnp.ones(state_dim),
                      initial_cov = jnp.eye(state_dim),
                      dynamics_weights = A,
                      dynamics_cov = Q,
                      emissions_weights = C,
                      emissions_cov = R,
                      dynamics_bias=B1,
                      dynamics_input_weights=None,
                      emissions_bias=None,
                      emissions_input_weights=None)
params_lds2 = make_lgssm_params(initial_mean = jnp.ones(state_dim),
                        initial_cov = jnp.eye(state_dim),
                        dynamics_weights = A,
                        dynamics_cov = Q,
                        emissions_weights = C,
                        emissions_cov = R,
                        dynamics_bias=B2,
                        dynamics_input_weights=None,
                        emissions_bias=None,
                        emissions_input_weights=None)
params_lds3 = make_lgssm_params(initial_mean = jnp.ones(state_dim),
                        initial_cov = jnp.eye(state_dim),
                        dynamics_weights = A,
                        dynamics_cov = Q,
                        emissions_weights = C,
                        emissions_cov = R,
                        dynamics_bias=B3,
                        dynamics_input_weights=None,
                        emissions_bias=None,
                        emissions_input_weights=None)

key = jr.PRNGKey(7)

params = inference.ParamsSLDS(
    discrete=discr_params,
    linear_gaussian=[params_lds1, params_lds2, params_lds3])

emissions = jnp.ones((10, emission_dim))
inputs = jnp.ones((10, 0))

out = inference.rbpfilter(num_particles, params, emissions, inputs, key)

#print(out)
