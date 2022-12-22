from typing import Sequence
from functools import partial
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import flax.linen as nn
from jax.flatten_util import ravel_pytree
from jax.experimental import host_callback


### MLP

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def get_mlp_flattened_params(model_dims, key=0):
    """Generate MLP model, initialize it using dummy input, and
    return the model, its flattened initial parameters, function
    to unflatten parameters, and apply function for the model.
    Args:
        model_dims (List): List of [input_dim, hidden_dim, ..., output_dim]
        key (PRNGKey): Random key. Defaults to 0.
    Returns:
        model: MLP model with given feature dimensions.
        flat_params: Flattened parameters initialized using dummy input.
        unflatten_fn: Function to unflatten parameters.
        apply_fn: fn(flat_params, x) that returns the result of applying the model.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    # Define MLP model
    input_dim, features = model_dims[0], model_dims[1:]
    model = MLP(features)
    dummy_input = jnp.ones((input_dim,))

    # Initialize parameters using dummy input
    params = model.init(key, dummy_input)
    flat_params, unflatten_fn = ravel_pytree(params)

    # Define apply function
    def apply(flat_params, x, model, unflatten_fn):
        return model.apply(unflatten_fn(flat_params), jnp.atleast_1d(x))

    apply_fn = partial(apply, model=model, unflatten_fn=unflatten_fn)

    return model, flat_params, unflatten_fn, apply_fn

### SGD

def fit_optax(params, optimizer, input, output, loss_fn, num_epochs, return_history=False):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    if return_history:
        params_history=[]
    
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(input, output)):
            params, opt_state, loss_value = step(params, opt_state, x, y)
            if return_history:
                params_history.append(params)
    
    if return_history:
        return jnp.array(params_history)
    return params

# Generic loss function
def loss_optax(params, x, y, loss_fn, apply_fn):
    y, y_hat = jnp.atleast_1d(y), apply_fn(params, x)
    loss_value = loss_fn(y, y_hat)
    return loss_value.mean()

# Define SGD optimizer
sgd_optimizer = optax.sgd(learning_rate=1e-2)
