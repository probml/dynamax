from abc import ABC
from abc import abstractmethod
from fastprogress.fastprogress import progress_bar
from functools import partial
import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, vmap
from jax.tree_util import tree_map
from jaxtyping import Float, Array
import optax
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Optional, Union, Tuple
from typing_extensions import Protocol

from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.optimize import run_sgd
from dynamax.utils.utils import ensure_array_has_batch_dim


class Posterior(Protocol):
    """A :class:`NamedTuple` with parameters stored as :class:`jax.DeviceArray` in the leaf nodes.

    """
    pass


class SSM(ABC):
    """A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form for optimization.

    Models that inherit from `SSM` must implement three key functions:

    * :meth:`initial_distribution`, which returns the distribution over the initial state
    * :meth:`transition_distribution`, which returns the conditional distribution over the next state given the current state
    * :meth:`emission_distribution`, which returns the conditional distribution over the emission given the current state

    Additionally, they must specify the shape of the emissions and inputs.
    These properties are required for properly handling batches of data.

    * :attr:`emission_shape` returns a tuple specification of the emission shape
    * :attr:`inputs_shape` returns a tuple specification of the input shape, or `None` if there are no inputs.


    """

    @abstractmethod
    def initial_distribution(self,
                             params: ParameterSet,
                             inputs: Optional[Float[Array, "input_dim"]]) \
                                 -> tfd.Distribution:
        """Return an initial distribution over latent states.

        Args:
            params: model parameters $\\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            distribution over initial latent state, $p(z_1 \mid \\theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(self,
                                params: ParameterSet,
                                state: Float[Array, "state_dim"],
                                inputs: Optional[Float[Array, "input_dim"]]) \
                                    -> tfd.Distribution:
        """Return a distribution over next latent state given current state.

        Args:
            params: model parameters $\\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of next latent state $p(z_{t+1} \mid z_t, u_t, \\theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def emission_distribution(self,
                              params: ParameterSet,
                              state: Float[Array, "state_dim"],
                              inputs: Optional[Float[Array, "input_dim"]]=None) \
                                  -> tfd.Distribution:
        """Return a distribution over emissions given current state.

        Args:
            params: model parameters $\\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of current emission $p(y_t \mid z_t, u_t, \\theta)$

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        """Return a pytree matching the pytree of tuples specifying the shape of a single time step's emissions.

        For example, a `GaussianHMM` with $D$ dimensional emissions would return `(D,)`.

        """
        raise NotImplementedError

    @property
    def inputs_shape(self) -> Optional[Tuple[int]]:
        """Return a pytree matching the pytree of tuples specifying the shape of a single time step's inputs.

        """
        return None

    # All SSMs support sampling
    def sample(self,
               params: ParameterSet,
               key: jr.PRNGKey,
               num_timesteps: int,
               inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    )-> Tuple[Float[Array, "num_timesteps state_dim"],
              Float[Array, "num_timesteps emission_dim"]]:
        """Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """
        def _step(prev_state, args):
            key, inpt = args
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state, inpt).sample(seed=key2)
            emission = self.emission_distribution(params, state, inpt).sample(seed=key1)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, initial_input).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def log_prob(
        self,
        params: ParameterSet,
        states: Float[Array, "num_timesteps state_dim"],
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Scalar:
        """Compute the log joint probability of the states and observations"""

        def _step(carry, args):
            lp, prev_state = carry
            state, emission, inpt = args
            lp += self.transition_distribution(params, prev_state, inpt).log_prob(state)
            lp += self.emission_distribution(params, state, inpt).log_prob(emission)
            return (lp, state), None

        # Compute log prob of initial time step
        initial_state = tree_map(lambda x: x[0], states)
        initial_emission = tree_map(lambda x: x[0], emissions)
        initial_input = tree_map(lambda x: x[0], inputs)
        lp = self.initial_distribution(params, initial_input).log_prob(initial_state)
        lp += self.emission_distribution(params, initial_state, initial_input).log_prob(initial_emission)

        # Scan over remaining time steps
        next_states = tree_map(lambda x: x[1:], states)
        next_emissions = tree_map(lambda x: x[1:], emissions)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        (lp, _), _ = lax.scan(_step, (lp, initial_state), (next_states, next_emissions, next_inputs))
        return lp

    def log_prior(
        self,
        params: ParameterSet
    ) -> Scalar:
        """Return the log prior probability of any model parameters.

        Returns:
            lp (Scalar): log prior probability.
        """
        return 0.0

    # Some SSMs will implement these inference functions.
    def marginal_log_prob(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Scalar:
        """Compute log marginal likelihood of observations, $\log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \\theta)$.

        Args:
            params: model parameters $\\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            marginal log probability

        """
        raise NotImplementedError

    def filter(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Posterior:
        """Compute filtering distributions, $p(z_t \mid y_{1:t}, u_{1:t}, \\theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            filtering distributions

        """
        raise NotImplementedError

    def smoother(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Posterior:
        """Compute smoothing distribution, $p(z_t \mid y_{1:T}, u_{1:T}, \\theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            smoothing distributions

        """
        raise NotImplementedError

    # Learning algorithms
    def e_step(self, params, emissions, inputs=None):
        """Perform an E-step to compute expected sufficient statistics under the posterior, $p(z_{1:T} \mid y_{1:T}, u_{1:T}, \\theta)$.

        Args:
            params: model parameters $\\theta$
            emissions: emissions $y_{1:T}$
            inputs: optional inputs $u_{1:T}$

        Returns:
            Expected sufficient statistics under the posterior.
        """
        raise NotImplementedError

    def fit_em(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
        num_iters: int=50,
        verbose: bool=True
    ) -> Tuple[ParameterSet, Float[Array, "niter"]]:
        """Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM)."""

        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        @jit
        def em_step(params, m_step_state):
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            return params, m_step_state, lp

        log_probs = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        for _ in pbar:
            params, m_step_state, marginal_loglik = em_step(params, m_step_state)
            log_probs.append(marginal_loglik)
        return params, jnp.array(log_probs)

    def fit_sgd(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        optimizer: optax.GradientTransformation=optax.adam(1e-3),
        batch_size: int=1,
        num_epochs: int=50,
        shuffle: bool=False,
        key: PRNGKey=jr.PRNGKey(0)
    ) -> Tuple[ParameterSet, Float[Array, "niter"]]:
        """Compute parameter MLE/ MAP estimate using Stochastic gradient descent (SGD).

        Fit the model by running SGD on the marginal log likelihood.

        Note that batch_emissions is initially of shape `(N,T)`
        where `N` is the number of independent sequences and
        `T` is the length of a sequence. Then, a random susbet with shape `(B, T)`
        of entire sequence, not time steps, is sampled at each step where `B` is
        batch size.

        Args:
            batch_emissions: Independent sequences.
            optmizer (optax.Optimizer): Optimizer.
            batch_size (int): Number of sequences used at each update step.
            num_epochs (int): Iterations made through entire dataset.
            shuffle (bool): Indicates whether to shuffle minibatches.
            key (jr.PRNGKey): RNG key to shuffle minibatches.
        Returns:
            losses: Output of loss_fn stored at each step.
        """
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        unc_params = to_unconstrained(params, props)

        def _loss_fn(unc_params, minibatch):
            """Default objective function."""
            params = from_unconstrained(unc_params, props)
            minibatch_emissions, minibatch_inputs = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            minibatch_lls = vmap(partial(self.marginal_log_prob, params))(minibatch_emissions, minibatch_inputs)
            lp = self.log_prior(params) + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

        dataset = (batch_emissions, batch_inputs)
        unc_params, losses = run_sgd(_loss_fn,
                                     unc_params,
                                     dataset,
                                     optimizer=optimizer,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     shuffle=shuffle,
                                     key=key)

        params = from_unconstrained(unc_params, props)
        return params, losses
