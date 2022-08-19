import jax.numpy as jnp
import jax.random as jr

from ssm_jax.hmm.models.linreg_hmm import LinearRegressionHMM

from jax import vmap, lax, tree_map

class LinearAutoregressiveHMM(LinearRegressionHMM):
    """A linear autoregressive HMM (ARHMM) is a special case of a 
    linear regression HMM where the covariates (i.e. features)
    are functions of the past emissions.
    """

    def __init__(self, 
        initial_probabilities, 
        transition_matrix, 
        emission_matrices, 
        emission_biases, 
        emission_covariance_matrices, 
        initial_probs_concentration=1.1,
        transition_matrix_concentration=1.1):

        assert emission_matrices.ndim == 3, "emission matrices must be 3d"
        num_states, dim, dim_times_lags = emission_matrices.shape
        assert dim_times_lags % dim == 0, "emission matrices must have integer number of lags"
        self._emission_dim = dim
        self._num_lags = dim_times_lags // dim

        super(LinearAutoregressiveHMM, self).__init__(
            initial_probabilities,
            transition_matrix,
            emission_matrices,
            emission_biases, 
            emission_covariance_matrices, 
            initial_probs_concentration=initial_probs_concentration,
            transition_matrix_concentration=transition_matrix_concentration)
        
    @property
    def num_lags(self):
        return self._num_lags

    @property 
    def emission_dim(self):
        return self._emission_dim

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim, num_lags):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))

        # Make random emission matrices that are close to identity
        emission_matrices = jnp.zeros((num_states, emission_dim, emission_dim * num_lags))
        emission_matrices = emission_matrices.at[:, :, :emission_dim].set(0.95 * jnp.eye(emission_dim))
        emission_matrices += 0.01 * jr.normal(key3, (num_states, emission_dim, emission_dim * num_lags))
        emission_biases = jr.normal(key4, (num_states, emission_dim))
        emission_covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))
        return cls(initial_probs, transition_matrix, emission_matrices, emission_biases, emission_covs)

    

    def sample(self, key, num_timesteps, prev_emissions=None):
        """
        """
        if prev_emissions is None:
            # Default to zeros 
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))


        def _step(carry, key):
            prev_state, prev_emissions = carry
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(prev_state).sample(seed=key2)
            emission = self.emission_distribution(state, features=jnp.ravel(prev_emissions)).sample(seed=key1)
            next_prev_emissions = jnp.row_stack([emission, prev_emissions[:-1]])
            return (state, next_prev_emissions), (state, emission, prev_emissions)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution().sample(seed=key1)
        initial_emission = self.emission_distribution(
            initial_state, features=jnp.ravel(prev_emissions)).sample(seed=key2)
        initial_prev_emissions = jnp.row_stack([initial_emission, prev_emissions[:-1]])

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions, next_prev_emissions) = lax.scan(
            _step, (initial_state, initial_prev_emissions), next_keys)

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        prev_emissions = tree_map(expand_and_cat, prev_emissions, next_prev_emissions)

        # Flatten the previous emissions into a vector
        prev_emissions = jnp.reshape(prev_emissions, (num_timesteps, -1))
        return states, emissions, prev_emissions


