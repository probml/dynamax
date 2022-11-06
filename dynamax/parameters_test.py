import chex 
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves

from jaxtyping import Float, Array

from dynamax.parameters import ParameterProperties, to_unconstrained, from_unconstrained
import tensorflow_probability.substrates.jax.bijectors as tfb


@chex.dataclass
class InitialParams:
    probs: Float[Array, "state_dim"]

@chex.dataclass
class TransitionsParams:
    transition_matrix: Float[Array, "state_dim state_dim"]

@chex.dataclass
class EmissionsParams:
    means: Float[Array, "state_dim emission_dim"]
    scales: Float[Array, "state_dim emission_dim"]

@chex.dataclass
class Params:
    initial: InitialParams
    transitions: TransitionsParams
    emissions: EmissionsParams

def test_parameter_tofrom_unconstrained():
    params = Params(
        initial=InitialParams(probs=jnp.ones(3) / 3.0),
        transitions=TransitionsParams(transition_matrix=0.9 * jnp.eye(3) + 0.1 * jnp.ones((3, 3)) / 3),
        emissions=EmissionsParams(means=jnp.zeros((3, 2)), scales=jnp.ones((3, 2)))
    )

    param_props = dict(
        initial=dict(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
        transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
        emissions=dict(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus(), trainable=False))
    )

    unc_params, fixed_params = to_unconstrained(params, param_props)
    recon_params = from_unconstrained(unc_params, fixed_params, param_props)
    assert all(tree_leaves(tree_map(jnp.allclose, params, recon_params)))
