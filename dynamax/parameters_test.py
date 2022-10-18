import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves

from dynamax.parameters import ParameterProperties, to_unconstrained, from_unconstrained
import tensorflow_probability.substrates.jax.bijectors as tfb


def test_parameter_tofrom_unconstrained():
    params = dict(
        initial=dict(probs=jnp.ones(3) / 3.0),
        transitions=dict(transition_matrix=0.9 * jnp.eye(3) + 0.1 * jnp.ones((3, 3)) / 3),
        emissions=dict(means=jnp.zeros((3, 2)), scales=jnp.ones((3, 2)))
    )

    param_props = dict(
        initial=dict(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
        transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
        emissions=dict(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus(), trainable=False))
    )

    unc_params, fixed_params = to_unconstrained(params, param_props)
    recon_params = from_unconstrained(unc_params, fixed_params, param_props)
    assert all(tree_leaves(tree_map(jnp.allclose, params, recon_params)))
