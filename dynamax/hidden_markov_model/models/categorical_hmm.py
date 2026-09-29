"""Categorical Hidden Markov Model."""
from typing import NamedTuple, Optional, Tuple, Union, List

import jax.numpy as jnp
from jax import lax
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.nn import one_hot
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions
from dynamax.hidden_markov_model.models.initial import ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import IntScalar, Scalar
from dynamax.types import PRNGKeyT
from dynamax.utils.utils import pytree_sum, ensure_array_has_batch_dim, low_rank_pinv, multilinear_product, cp_decomp


class ParamsCategoricalHMMEmissions(NamedTuple):
    """Parameters for the CategoricalHMM emission distribution."""
    probs: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]


class ParamsCategoricalHMM(NamedTuple):
    """Parameters for the CategoricalHMM model."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsCategoricalHMMEmissions


class CategoricalHMMEmissions(HMMEmissions):
    r"""Categorical emissions for a hidden Markov model."""

    def __init__(self,
                 num_states: int,
                 emission_dim: int,
                 num_classes: int,
                 emission_prior_concentration: Union[Scalar, Float[Array, " num_classes"]]=1.1):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.emission_prior_concentration = emission_prior_concentration  * jnp.ones(num_classes)

    @property
    def emission_shape(self) -> Tuple[int]:
        """Shape of the emission distribution."""
        return (self.emission_dim,)

    def distribution(self, params: ParamsCategoricalHMMEmissions, state: IntScalar, inputs=None) -> tfd.Distribution:
        """Return the emission distribution for a given state."""
        return tfd.Independent(
            tfd.Categorical(probs=params.probs[state]),
            reinterpreted_batch_ndims=1)

    def log_prior(self, params: ParamsCategoricalHMMEmissions) -> Scalar:
        """Return the log prior probability of the emission parameters."""
        return tfd.Dirichlet(self.emission_prior_concentration).log_prob(params.probs).sum()

    def initialize(self,
                   key:Optional[Array]=jr.PRNGKey(0),
                   method="prior",
                   emission_probs:Optional[Float[Array, "num_states emission_dim num_classes"]]=None
                   ) -> Tuple[ParamsCategoricalHMMEmissions, ParamsCategoricalHMMEmissions]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Initialize the emission probabilities
        if emission_probs is None:
            if method.lower() == "prior":
                if key is None:
                    raise ValueError("key must not be None when emission_probs is None")
                prior = tfd.Dirichlet(self.emission_prior_concentration)
                emission_probs = prior.sample(seed=key, sample_shape=(self.num_states, self.emission_dim))
            elif method.lower() == "kmeans":
                raise NotImplementedError("kmeans initialization is not yet implemented!")
            else:
                raise Exception("invalid initialization method: {}".format(method))
        else:
            assert emission_probs.shape == (self.num_states, self.emission_dim, self.num_classes)
            assert jnp.all(emission_probs >= 0)
            assert jnp.allclose(emission_probs.sum(axis=2), 1.0)

        # Add parameters to the dictionary
        params = ParamsCategoricalHMMEmissions(probs=emission_probs)
        props = ParamsCategoricalHMMEmissions(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        """Collect sufficient statistics for the emission distribution."""
        expected_states = posterior.smoothed_probs
        x = one_hot(emissions, self.num_classes)
        return dict(sum_x=jnp.einsum("tk,tdi->kdi", expected_states, x))

    def initialize_m_step_state(self, params, props):
        """Initialize the m-step state."""
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        """Perform the m-step for the emission distribution."""
        if props.probs.trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            probs = tfd.Dirichlet(self.emission_prior_concentration + emission_stats['sum_x']).mode()
            params = params._replace(probs=probs)
        return params, m_step_state
    
    def calc_sample_moment(self, 
                           emissions: Float[Array, "num_batches num_timesteps emission_dim"], 
                           order: Union[int,
                                        List[int]]):
        r"""Find the sample cross moments of order $n$. These are averaged over the
        full timeseries because the HMM is time homogeneous, so for example the following
        are assumed interchangeable:

        $$\mathbb{E}[x_1 \otimes x_2 \otimes x_3]$$

        $$\mathbb{E}[x_{t+1} \otimes x_{t+2} \dots x_{t+3}]$$
        """
        x = one_hot(jnp.squeeze(emissions, -1), num_classes=self.num_classes)
        B, T, _ = x.shape
        if isinstance(order, int):
            order = list(range(order))
        order_len = max(order)+1
        T_effective = T - order_len + 1
        if T_effective <= 0:
            raise ValueError
        
        einsum_args = []
        output_indices = []
        for i, j in enumerate(order):
            slice_j = x[:, j:T_effective+j, :]
            einsum_args.append(slice_j)
            einsum_args.append([0, 1, 2+j])
            output_indices.append(2+j)
            
        einsum_args.append(output_indices)
        sum_outer_products = jnp.einsum(*einsum_args)

        return sum_outer_products / (B * T)
    
    def calc_pos_sample_mean(self, emissions, pos):
        return jnp.mean(one_hot(jnp.squeeze(emissions, -1), num_classes=self.num_classes)[:,pos,:], axis=0)


class CategoricalHMM(HMM):
    r"""An HMM with conditionally independent categorical emissions.

    Let $y_t \in \{1,\ldots,C\}^N$ denote a vector of $N$ conditionally independent
    categorical emissions from $C$ classes at time $t$. In this model,the emission
    distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathrm{Cat}(y_{tn} \mid \theta_{z_t,n})$$
    $$p(\theta) = \prod_{k=1}^K \prod_{n=1}^N \mathrm{Dir}(\theta_{k,n}; \gamma 1_C)$$

    with $\theta_{k,n} \in \Delta_C$ for $k=1,\ldots,K$ and $n=1,\ldots,N$ are the
    *emission probabilities* and $\gamma$ is their prior concentration.

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param num_classes: number of multinomial classes $C$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_concentration: $\gamma$

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 num_classes: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_concentration=1.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = CategoricalHMMEmissions(num_states, emission_dim, num_classes, emission_prior_concentration=emission_prior_concentration)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_probs: Optional[Float[Array, "num_states emission_dim num_classes"]]=None
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_probs=emission_probs)
        return ParamsCategoricalHMM(**params), ParamsCategoricalHMM(**props)


    def get_view(self, 
                 batch_emissions: Float[Array, "num_batches num_timesteps emission_dim"], 
                 target: int, 
                 num_init: int = 100, 
                 num_iter: int = 1000, 
                 key: Array = jr.PRNGKey(0)
                 ) -> Array:
        r"""Return the sample conditional means from the requested view.

        Specifically, return the conditional mean of the emissions at time t+target
        given the hidden state at time t+1. 

        $$\mathbb{E}[x_{t+target} \mid y_{t+1}=h]$$

        Args:
            batch_emissions: the emission data.
            target: the requested timestep relative to the hidden state being conditioned on.
            num_init: number of random starting points should be used in the robust tensor power method.
            num_iter: number of iterations in the robust tensor power method.
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.

        Returns:
            Conditional mean vector $\mu$.
        """
        k = self.num_states
        sym_M2, sym_M3 = self.moment_view(batch_emissions, target, k)
        # find whitening matrix W
        eigvals, eigvecs = jnp.linalg.eigh(sym_M2)

        idx = jnp.argsort(jnp.abs(eigvals))[-k:]
        trunc_eigvals = eigvals[idx]
        trunc_eigvecs = eigvecs[:,idx]
        
        U = trunc_eigvecs
        D = jnp.diag(1/jnp.sqrt(trunc_eigvals))
        
        W = U @ D
        B = jnp.linalg.pinv(W.T)

        # tensor decomposition
        tilde_sym_M3 = multilinear_product(sym_M3, [W, W, W])
        rob_eigvecs, rob_eigvals = cp_decomp(tilde_sym_M3, L=num_init, N=num_iter, k=k, key=key)
        
        return jnp.diag(rob_eigvals) @ rob_eigvecs @ B.T


    def fit_moments(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        num_init: int=100,
        num_iter: int=1000,
        key: Array=jr.PRNGKey(0)
    ) -> ParameterSet:
        r"""Estimate the parameters using method of moments.

        Specifically, compute emission distribution and transition matrix from the second
        and third moments. Since the model is time homogeneous, you can take it over all
        consecutive 2 or 3 timesteps respectively. To recover the initial distribution, take
        the mean over the first timestep of each sequence using the known emission
        distribution to find the hidden state distribution.

        Then 

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: observations from data.
            num_init: number of random starting points should be used in the robust tensor power method.
            num_iter: number of iterations in the robust tensor power method.
            key: sufficient statistics from each sequence

        Returns:
            new parameters

        """
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        key_2, key_3 = jr.split(key, 2)
        mu_1 = self.get_view(batch_emissions, 1, 100,1000, key_2)
        mu_2 = self.get_view(batch_emissions, 2, 100,1000, key_3)
        k = self.num_states

        transition_params = mu_2 @ low_rank_pinv(mu_1, k)
        emission_params = mu_1
        
        initial_params = low_rank_pinv(emission_params.T, k) @ self.emission_component.calc_pos_sample_mean(batch_emissions, 0)
        params = params._replace(initial=initial_params, transitions=transition_params, emissions=emission_params)

        return params, 


    def moment_view(self, 
                    batch_emissions: Float[Array, "num_batches num_timesteps emission_dim"], 
                    target: int, 
                    k: int
                    ) -> Tuple[Array, Array]:
        r"""Perform the symmetrizing operation to get a particular view of the
        second and third order moments.

        Specifically, compute the second and third moments. Since the model is time 
        homogeneous, you can take it over all consecutive 2 or 3 timesteps respectively.

        Then 

        Args:
            batch_emissions: the emission data.
            target: the requested view.
            k: the number of hidden states.

        Returns:
            sym_M2: symmetrized second order moment corresponding to view of `target`.
            sym_M3: symmetrized third order moment corresponding to view of `target`.
        """
        
        if target == 0:
            source_1, source_2 = 1, 2
        elif target == 1:
            source_1, source_2 = 0, 2
        else:
            source_1, source_2 = 0, 1
        
        A = self.emission_component.calc_sample_moment(batch_emissions, [target, source_2])
        B = self.emission_component.calc_sample_moment(batch_emissions, [source_1, source_2])
        C = self.emission_component.calc_sample_moment(batch_emissions, [target, source_1])
        D = self.emission_component.calc_sample_moment(batch_emissions, [source_2, source_1])

        M2 = self.emission_component.calc_sample_moment(batch_emissions, [source_1, source_2])
        M3 = self.emission_component.calc_sample_moment(batch_emissions, 3)

        d = self.emission_component.num_classes
        
        sym_pre = jnp.transpose(A @ low_rank_pinv(B, k))
        sym_post = jnp.transpose(C @ low_rank_pinv(D, k))
        
        M3_args = [jnp.eye(d)]*3
        M3_args[source_1] = sym_pre
        M3_args[source_2] = sym_post
        
        sym_M2 = multilinear_product(M2, [sym_pre, sym_post])
        sym_M3 = multilinear_product(M3, M3_args)
        return sym_M2, sym_M3
