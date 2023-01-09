import pytest
import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model.models import transitions


def test_hmm_sparse_transitions(num_states=4):
    key1, key2 = jr.split(jr.PRNGKey(42))
    # no provided transition_matrix
    mask = jnp.eye(num_states) + jnp.eye(num_states, k=1)

    transition_component = transitions.StandardHMMSparseTransitions(
        num_states, mask)
    params, _ = transition_component.initialize(key1)
    assert params.transition_matrix.shape == (num_states, num_states)

    # provided transition_matrix
    mask = jnp.eye(num_states)
    transition_matrix = jnp.eye(num_states) * 0.75 + jnp.roll(
        jnp.eye(num_states) * 0.25, shift=1, axis=1)
    transition_component = transitions.StandardHMMSparseTransitions(
        num_states, mask)
    with pytest.raises(ValueError):
        params, _ = transition_component.initialize(
            key2, transition_matrix=transition_matrix)
