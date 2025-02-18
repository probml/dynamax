"""Tests for dynamax.parameters module"""
import copy
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb

from dynamax.parameters import ParameterProperties, to_unconstrained, from_unconstrained, log_det_jac_constrain
from jax import jit, value_and_grad, lax
from jax.tree_util import tree_map, tree_leaves
from jaxtyping import Float, Array
from typing import NamedTuple, Union


class InitialParams(NamedTuple):
    """Dummy Initial state distribution parameters"""
    probs: Union[Float[Array, " state_dim"], ParameterProperties]

class TransitionsParams(NamedTuple):
    """Dummy Transition matrix parameters"""
    transition_matrix: Union[Float[Array, "state_dim state_dim"], ParameterProperties]

class EmissionsParams(NamedTuple):
    """Dummy Emission distribution parameters"""
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    scales: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]

class Params(NamedTuple):
    """Dummy SSM parameters"""
    initial: InitialParams
    transitions: TransitionsParams
    emissions: EmissionsParams


def make_params():
    """Create a dummy set of parameters and properties"""
    params = Params(
        initial=InitialParams(probs=jnp.ones(3) / 3.0),
        transitions=TransitionsParams(transition_matrix=0.9 * jnp.eye(3) + 0.1 * jnp.ones((3, 3)) / 3),
        emissions=EmissionsParams(means=jnp.zeros((3, 2)), scales=jnp.ones((3, 2)))
    )

    props = Params(
        initial=InitialParams(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
        transitions=TransitionsParams(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
        emissions=EmissionsParams(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus(), trainable=False))
    )
    return params, props


def test_parameter_tofrom_unconstrained():
    """Test that to_unconstrained and from_unconstrained are inverses"""
    params, props = make_params()
    unc_params = to_unconstrained(params, props)
    recon_params = from_unconstrained(unc_params, props)
    assert all(tree_leaves(tree_map(jnp.allclose, params, recon_params)))


def test_parameter_pytree_jittable():
    """Test that the parameter PyTree is jittable"""
    # If there's a problem with our PyTree registration, this should catch it.
    params, props = make_params()

    @jit
    def get_trainable(params, props):
        """Return a PyTree of trainable parameters"""
        return tree_map(lambda node, prop: node if prop.trainable else None,
                        params, props,
                        is_leaf=lambda node: isinstance(node, ParameterProperties))

    # first call, jit
    get_trainable(params, props)
    assert get_trainable._cache_size() == 1

    # change param values, don't jit
    params = params._replace(initial=params.initial._replace(probs=jnp.zeros(3)))
    get_trainable(params, props)
    assert get_trainable._cache_size() == 1

    # change param dtype, jit
    params = params._replace(initial=params.initial._replace(probs=jnp.zeros(3, dtype=int)))
    get_trainable(params, props)
    assert get_trainable._cache_size() == 2

    # change props, jit
    props.transitions.transition_matrix.trainable = False
    get_trainable(params, props)
    assert get_trainable._cache_size() == 3


def test_parameter_constrained():
    """Test that only trainable params are updated in gradient descent.
    """
    params, props = make_params()
    original_params = copy.deepcopy(params)

    unc_params = to_unconstrained(params, props)
    def loss(unc_params):
        """Dummy loss function"""
        params = from_unconstrained(unc_params, props)
        log_initial_probs = jnp.log(params.initial.probs)
        log_transition_matrix = jnp.log(params.transitions.transition_matrix)
        means = params.emissions.means
        scales = params.emissions.scales

        lp = log_initial_probs[1]
        lp += log_transition_matrix[0,0]
        lp += log_transition_matrix[1,1]
        lp += log_transition_matrix[2,2]
        lp += jnp.sum(-0.5 * (1.0 - means[0])**2 / scales[0]**2)
        lp += jnp.sum(-0.5 * (2.0 - means[1])**2 / scales[1]**2)
        lp += jnp.sum(-0.5 * (3.0 - means[2])**2 / scales[2]**2)
        return -lp

    # Run a dummy optimization
    f = jit(value_and_grad(loss))
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(unc_params)

    def step(carry, args):
        """Optimization step"""
        unc_params, opt_state = carry
        loss, grads = f(unc_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        unc_params = optax.apply_updates(unc_params, updates)
        return (unc_params, opt_state), loss

    initial_carry =  (unc_params, opt_state)
    (unc_params, opt_state), losses = \
        lax.scan(step, initial_carry, None, length=10)
    params = from_unconstrained(unc_params, props)

    assert jnp.allclose(params.initial.probs, original_params.initial.probs)
    assert not jnp.allclose(params.transitions.transition_matrix, original_params.transitions.transition_matrix)
    assert not jnp.allclose(params.emissions.means, original_params.emissions.means)
    assert jnp.allclose(params.emissions.scales, original_params.emissions.scales)


def test_logdet_jacobian():
    """Test that log_det_jac_constrain is correct"""
    params, props = make_params()
    unc_params = to_unconstrained(params, props)
    logdet = log_det_jac_constrain(params, props)

    # only the transition matrix is constrained and trainable
    f = props.transitions.transition_matrix.constrainer.forward_log_det_jacobian
    logdet_manual = f(unc_params.transitions.transition_matrix).sum()
    assert jnp.isclose(logdet, logdet_manual)
