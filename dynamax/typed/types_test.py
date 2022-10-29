from jaxtyping import install_import_hook
install_import_hook(["dynamax.types"], ("typeguard", "typechecked"))

from dynamax.typed.types import *
from dynamax.typed.hmm_typed import *
from dynamax.typed.lgssm_typed import *

import jax.random as jr
import jax.numpy as jnp

key = jr.PRNGKey(0)

T = 5
Ny = 2
Nu = 3
Nz = 4

model = MyLGSSM(state_dim = Nz, emission_dim = Ny, covariate_dim = Nu)
(params, param_props) = model.initialize_params(key)

(states, emissions) = model.sample(params, key, T)
print('states ', states.shape)
print('emissions ', emissions.shape)

post = model.filter(params, emissions)
print('post ', post.filtered_means.shape)

emissions = jnp.ones((T, Ny))
inputs = jnp.ones((T, Nu))
post = model.filter(params, emissions, inputs=inputs)

try:
    emissions = jnp.ones(T)
    post = model.filter(params, emissions)
except:
    print('Bad emissions')

def test_lgssm():
    T = 5
    Ny = 2
    Nu = 3
    Nz = 4

    model = MyLGSSM(state_dim = Nz, emission_dim = Ny, covariate_dim = Nu)
    (params, param_props) = model.initialize_params(key)

    (states, emissions) = model.sample(params, key, T)
    print('states ', states.shape)
    print('emissions ', emissions.shape)

    post = model.filter(params, emissions)
    print('post ', post.filtered_means.shape)

    emissions = jnp.ones((T, Ny))
    inputs = jnp.ones((T, Nu))
    post = model.filter(params, emissions, inputs=inputs)

    try:
        emissions = jnp.ones(T)
        post = model.filter(params, emissions)
    except:
        print('Bad emissions')

    try:
        emissions = jnp.ones((T, Ny))
        inputs = jnp.ones((T, Nu, Nu))
        post = model.filter(params, emissions, inputs=inputs)
        # TypeError: type of argument "inputs" must be one of (jaxtyping.Float[Array, 'Ntime Ninput'], NoneType); got jaxlib.xla_extension.DeviceArray instead
    except:
        print('Bad inputs')


def test_hmm():
    T = 5
    Ny = 2
    Nu = 3
    Nz = 4

    model = MyHMM(nstates = Nz, emission_dim = Ny, covariate_dim = Nu)
    (params, param_props) = model.initialize_params(key)

    (states, emissions) = model.sample(params, key, T)
    print('states ', states.shape)
    print('emissions ', emissions.shape)

    post = model.filter(params, emissions)
    print('post ', post.filtered_probs.shape)

#if __name__ == "__main__":
#    test_lgssm()
#    test_hmm()


