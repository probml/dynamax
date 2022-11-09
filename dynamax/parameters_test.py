import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties, to_unconstrained, from_unconstrained
import tensorflow_probability.substrates.jax.bijectors as tfb
from typing import NamedTuple, Union

class InitialParams(NamedTuple):
    probs: Union[Float[Array, "state_dim"], ParameterProperties]

class TransitionsParams(NamedTuple):
    transition_matrix: Union[Float[Array, "state_dim state_dim"], ParameterProperties]

class EmissionsParams(NamedTuple):
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    scales: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]

class Params(NamedTuple):
    initial: InitialParams
    transitions: TransitionsParams
    emissions: EmissionsParams

def test_parameter_tofrom_unconstrained():
    params = Params(
        initial=InitialParams(probs=jnp.ones(3) / 3.0),
        transitions=TransitionsParams(transition_matrix=0.9 * jnp.eye(3) + 0.1 * jnp.ones((3, 3)) / 3),
        emissions=EmissionsParams(means=jnp.zeros((3, 2)), scales=jnp.ones((3, 2)))
    )

    param_props = Params(
        initial=InitialParams(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
        transitions=TransitionsParams(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
        emissions=EmissionsParams(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus(), trainable=False))
    )

    unc_params = to_unconstrained(params, param_props)
    recon_params = from_unconstrained(unc_params, param_props)
    assert all(tree_leaves(tree_map(jnp.allclose, params, recon_params)))
