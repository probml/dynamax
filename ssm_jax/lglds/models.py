from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class LDS:

    def __init__(self,
                 initial_mean,
                 initial_covariance,
                 dynamics_matrix,
                 dynamics_input_weights,
                 dynamics_covariance,
                 emissions_matrix,
                 emissions_input_weights,
                 emissions_covariance) -> None:

        self._initial_mean = initial_mean
        self._initial_covariance = initial_covariance
        self._dynamics_matrix = dynamics_matrix
        self._dynamics_input_weights = dynamics_input_weights
        self._dynamics_covariance = dynamics_covariance
        self._emissions_matrix = emissions_matrix
        self._emissions_input_weights = emissions_input_weights
        self._emissions_covariance = emissions_covariance

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
