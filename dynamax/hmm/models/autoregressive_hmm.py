import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree_map

from dynamax.hmm.models.linreg_hmm import LinearRegressionHMM
from dynamax.parameters import ParameterProperties
from dynamax.utils import PSDToRealBijector

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class LinearAutoregressiveHMM(LinearRegressionHMM):
    """A linear autoregressive HMM (ARHMM) is a special case of a
    linear regression HMM where the covariates (i.e. features)
    are functions of the past emissions.
    """

    def __init__(self,
                 num_states,
                 emission_dim,
                 num_lags=1,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1
                 ):
        self.num_lags = num_lags
        feature_dim = num_lags * emission_dim

        super().__init__(num_states, feature_dim, emission_dim,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return (self.feature_dim,)

    def _initialize_emissions(self, key):
        key1, key2 = jr.split(key, 2)

        # Make random emission matrices that are close to identity
        emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
        emission_weights = emission_weights.at[:, :, :self.emission_dim].set(0.95 * jnp.eye(self.emission_dim))
        emission_weights += 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
        emission_biases = jr.normal(key2, (self.num_states, self.emission_dim))
        emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))

        params = dict(weights=emission_weights, biases=emission_biases, covs=emission_covs)
        param_props = dict(weights=ParameterProperties(), biases=ParameterProperties(), covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return params, param_props

    def sample(self, params, key, num_timesteps, prev_emissions=None):
        """
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        def _step(carry, key):
            prev_state, prev_emissions = carry
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state).sample(seed=key2)
            emission = self.emission_distribution(params, state, covariates=jnp.ravel(prev_emissions)).sample(seed=key1)
            next_prev_emissions = jnp.row_stack([emission, prev_emissions[:-1]])
            return (state, next_prev_emissions), (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution(params).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, covariates=jnp.ravel(prev_emissions)).sample(seed=key2)
        initial_prev_emissions = jnp.row_stack([initial_emission, prev_emissions[:-1]])

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(
            _step, (initial_state, initial_prev_emissions), next_keys)

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def compute_covariates(self, emissions, prev_emissions=None):
        """Helper function to compute the matrix of lagged emissions. These are the
        covariates to the fitting functions.

        Args:
            emissions (array): (T, D) array of emissions

        Returns:
            lagged emissions (array): (T, D*num_lags) array of lagged emissions
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        padded_emissions = jnp.vstack((prev_emissions, emissions))
        num_timesteps = len(emissions)
        return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                 for lag in reversed(range(self.num_lags))])
