# Pendulum simulation
# Taken from:
# Simo Särkkä (2013), “Bayesian Filtering and Smoothing,” (pg. 74)
# Example 5.1 (Pendulum tracking with EKF)
# Available: https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf

import jax.numpy as jnp
import jax.random as jr
from jax import lax


class PendulumSimulation:
    def __init__(self):
        g, q_c, r = 9.8, 1, 0.3
        # Parameters for pendulum simulation
        self.dt = 0.0125
        self.initial_state = jnp.array([jnp.pi / 2, 0])
        self.dynamics_function = lambda x: jnp.array([x[0] + x[1] * self.dt, x[1] - g * jnp.sin(x[0]) * self.dt])
        self.dynamics_covariance = jnp.array(
            [[q_c * self.dt ** 3 / 3, q_c * self.dt ** 2 / 2], [q_c * self.dt ** 2 / 2, q_c * self.dt]]
        )
        self.emission_function = lambda x: jnp.array([jnp.sin(x[0])])
        self.emission_covariance = jnp.eye(1) * (r ** 2)

    def sample(self, key=0, num_steps=400):
        if isinstance(key, int):
            key = jr.PRNGKey(key)
        # Unpack parameters
        M, N = self.initial_state.shape[0], self.emission_covariance.shape[0]
        f, h = self.dynamics_function, self.emission_function
        Q, R = self.dynamics_covariance, self.emission_covariance

        def _step(carry, rng):
            state = carry
            rng1, rng2 = jr.split(rng, 2)

            next_state = f(state) + jr.multivariate_normal(rng1, jnp.zeros(M), Q)
            obs = h(next_state) + jr.multivariate_normal(rng2, jnp.zeros(N), R)
            return next_state, (next_state, obs)

        rngs = jr.split(key, num_steps)
        _, (states, observations) = lax.scan(_step, self.initial_state, rngs)

        # Generate time grid
        time_grid = jnp.arange(0.0, num_steps * self.dt, step=self.dt)

        return states, observations, time_grid
