import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from ssm_jax.hmm.models.categorical_glm_hmm import CategoricalRegressionHMM


def new_hmm():
    """
    States : rainy, sunny
    Emissions : walk, shop, clean
    """
    num_states = 2
    num_classes = 3
    feature_dim = 2
    key = jr.PRNGKey(0)
    hmm = CategoricalRegressionHMM.random_initialization(key, num_states, num_classes, feature_dim)
    return hmm

class TestCategoricalGLMHMM:
    def setup(self):
        self.num_states = 2
        self.num_classes = 3
        self.num_features = 2

    def test_sample(self, key=jr.PRNGKey(0), num_timesteps=1000):
        hmm = new_hmm()
        features = jr.normal(key, (num_timesteps, self.num_features))
        state_sequence, emissions = hmm.sample(key, num_timesteps)
        assert len(emissions) == len(state_sequence) == num_timesteps
        assert len(jnp.unique(emissions)) == self.num_classes