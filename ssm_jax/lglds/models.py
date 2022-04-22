import jax.numpy as jnp
from jax import lax
import jax.random as jr
from jax.tree_util import register_pytree_node_class

from distrax import MultivariateNormalFullCovariance as MVN


@register_pytree_node_class
class LinearGaussianSSM:
    def __init__(self,
                 initial_mean,
                 initial_covariance,
                 dynamics_matrix,
                 dynamics_input_weights,
                 dynamics_covariance,
                 emissions_matrix,
                 emissions_input_weights,
                 emissions_covariance) -> None:
        """
        A simple implementation of a linear Gaussian dynamical system,
        sometimes simply called a "linear dynamical system."

        TODO: Args:
            initial_mean (_type_): _description_
            initial_covariance (_type_): _description_
            dynamics_matrix (_type_): _description_
            dynamics_input_weights (_type_): _description_
            dynamics_covariance (_type_): _description_
            emissions_matrix (_type_): _description_
            emissions_input_weights (_type_): _description_
            emissions_covariance (_type_): _description_
        """
        self._initial_mean = initial_mean
        self._initial_covariance = initial_covariance
        self._dynamics_matrix = dynamics_matrix
        self._dynamics_input_weights = dynamics_input_weights
        self._dynamics_covariance = dynamics_covariance
        self._emissions_matrix = emissions_matrix
        self._emissions_input_weights = emissions_input_weights
        self._emissions_covariance = emissions_covariance

    # Make LinearGaussianSSM objects JAX PyTrees
    def tree_flatten(self):
        children = (self._initial_mean,
                    self._initial_covariance,
                    self._dynamics_matrix,
                    self._dynamics_input_weights,
                    self._dynamics_covariance,
                    self._emissions_matrix,
                    self._emissions_input_weights,
                    self._emissions_covariance)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    # Functions to return dynamics matrices, etc.
    # These parameters are not time varying.
    @property
    def initial_mean(self):
        return self._initial_mean

    @property
    def initial_covariance(self):
        return self._initial_covariance

    def dynamics_matrix(self, t):
        return self._dynamics_matrix

    def dynamics_inputs_weights(self, t):
        return self._dynamics_inputs_weights

    def dynamics_covariance(self, t):
        return self._dynamics_covariance

    def emissions_matrix(self, t):
        return self._emissions_matrix

    def emissions_inputs_weights(self, t):
        return self._emissions_inputs_weights

    def emissions_covariance(self, t):
        return self._emissions_covariance

    def sample(self, rng, num_steps, inputs):
        """
        Sample latent states and data from this LGSSM.

        Args:
            rng:        jax.random.PRNGKey
            num_steps:  number of time steps to simulte
            inputs:     array of inputs to the LDS

        Returns:
            xs:         (num_steps, latent_dim) array of latent states
            ys:         (num_steps, data_dim) array of data
        """
        def _step(carry, rng_and_t):
            xt = carry
            rng, t = rng_and_t

            # Get parameters and inputs for time index t
            At = self.dynamics_matrix(t)
            Bt = self.dynamics_input_weights(t)
            Qt = self.dynamics_covariance(t)
            Ct = self.emissions_matrix(t)
            Dt = self.emissions_input_weights(t)
            Rt = self.emissions_covariance(t)
            ut = inputs[t]

            # Sample data and next state
            rng1, rng2 = jr.split(rng, 2)
            yt = MVN(Ct @ xt + Dt @ ut, Rt).sample(seed=rng1)
            xtp1 = MVN(At @ xt + Bt @ ut, Qt).sample(seed=rng2)
            return xtp1, (xt, yt)

        # Initialize
        rng, this_rng = jr.split(rng, 2)
        x0 = MVN(self.m0, self.Q0).sample(seed=this_rng)

        # Run the sampler
        rngs = jr.split(rng, num_steps)
        _, (xs, ys) = lax.scan(_step, x0, (rngs, jnp.arange(num_steps), inputs))
        return xs, ys
