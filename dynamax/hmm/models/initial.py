import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.hmm.models.base import HMMInitialState
from dynamax.parameters import ParameterProperties


class StandardHMMInitialState(HMMInitialState):
    """Abstract class for HMM initial distributions.
    """
    def __init__(self,
                 num_states,
                 initial_probs_concentration=1.1):
        """
        Args:
            initial_probabilities[k]: prob(hidden(1)=k)
        """
        self.num_states = num_states
        self.initial_probs_concentration = initial_probs_concentration * jnp.ones(num_states)

    def distribution(self, params, covariates=None):
        return tfd.Categorical(probs=params['probs'])

    def initialize(self, key=None, method="prior", initial_probs=None):
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".
            initial_probs (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Initialize the initial probabilities
        if initial_probs is None:
            this_key, key = jr.split(key)
            initial_probs = tfd.Dirichlet(self.initial_probs_concentration).sample(seed=this_key)

        # Package the results into dictionaries
        params = dict(probs=initial_probs)
        props = dict(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params):
        return tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['probs'])

    def compute_initial_probs(self, params, covariates=None):
        return params['probs']

    def collect_suff_stats(self, posterior, covariates=None):
        return posterior.smoothed_probs[0]

    def m_step(self, params, props, batch_stats):

        if not props['probs'].trainable:
            return params

        elif self.num_states == 1:
            params['probs'] = jnp.array([1.0])
            return params

        else:
            expected_initial_counts = batch_stats.sum(axis=0)
            post = tfd.Dirichlet(self.initial_probs_concentration + expected_initial_counts)
            params['probs'] = post.mode()
            return params
