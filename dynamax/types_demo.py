from jaxtyping import install_import_hook
install_import_hook(["dynamax.types"], ("typeguard", "typechecked"))
from dynamax.types import *

import jax.random as jr
import jax.numpy as jnp


key = jr.PRNGKey(0)

T = 10
Ny = 2
Nu = 3
Nz = 4

model = MyLGSSM(state_dim = Nz, emission_dim = Ny, input_dim = Nu)
(params, param_props) = model.initialize_params(key)

(states, emissions) = model.sample(params, key, T)
print('states ', states.shape)
print('emissions ', emissions.shape)

post = model.filter(params, emissions)

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



