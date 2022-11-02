from jaxtyping import install_import_hook

install_import_hook(["dynamax.typed.types", "dynamax.typed.lgssm_typed", "dynamax.typed.hmm_typed"], ("typeguard", "typechecked"))

from dynamax.typed.types import *
from dynamax.typed.hmm_typed import *
from dynamax.typed.lgssm_typed import *

import jax.random as jr
import jax.numpy as jnp


def test_lgssm():
    print('LGSSM')
    T = 5
    Ny = 2
    Nu = 3
    Nz = 4
    key = jr.PRNGKey(0) 

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
        emissions = jnp.ones(T) # wrong shape!
        post = model.filter(params, emissions)
    except:
        print('Bad emissions')

    try:
        emissions = jnp.ones((T, Ny))
        inputs = jnp.ones((T, Nu, Nu)) # wrong shape!
        post = model.filter(params, emissions, inputs=inputs)
    except:
        print('Bad inputs')


def test_hmm():
    print('HMM')
    T = 5
    Ny = 2
    Nu = 3
    Nz = 4
    key = jr.PRNGKey(0)

    model = MyHMMD(num_states = Nz, emission_dim = Ny, covariate_dim = Nu)
    (params, param_props) = model.initialize_params(key)
    (states, emissions) = model.sample(params, key, T)
    print('states ', states.shape)
    print('emissions ', emissions.shape, 'rank ', emissions.ndim)
    post = model.filter(params, emissions)
    print('post ', post.filtered_probs.shape)

    modelV = MyHMMV(num_states = Nz, emission_dim = 1, covariate_dim = Nu)
    (paramsV, param_propsV) = modelV.initialize_params(key)
    (statesV, emissionsV) = modelV.sample(paramsV, key, T)
    print('states ', statesV.shape)
    print('emissions ', emissionsV.shape, 'rank ', emissionsV.ndim)
    postV = modelV.filter(paramsV, emissionsV)
    print('post ', postV.filtered_probs.shape)

    try:
        postV = modelV.filter(paramsV, emissions) # fail
    except:
        print('bad emissions')


if __name__ == "__main__":
    test_lgssm()
    test_hmm()


