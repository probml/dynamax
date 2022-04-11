import jax
import jax.numpy as jnp

def markov_chain_sample(rng_key, init_dist, trans_mat, seq_len):
    n_states = len(init_dist)

    def draw_state(prev_state, key):
        state = jax.random.choice(key, n_states, p=trans_mat[prev_state])
        return state, state

    rng_key, rng_state = jax.random.split(rng_key, 2)
    keys = jax.random.split(rng_state, seq_len - 1)
    initial_state = jax.random.choice(rng_key, n_states, p=init_dist)
    final_state, states = jax.lax.scan(draw_state, initial_state, keys)
    state_seq = jnp.append(jnp.array([initial_state]), states)

    return state_seq
