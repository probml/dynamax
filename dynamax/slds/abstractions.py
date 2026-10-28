r"""
Component interfaces and the driver of the switching linear dynamical system.

An SLDS has a discrete chain :math:`z_{1:T}` and a continuous chain :math:`x_{1:T}`:

.. math::
    z_0 \sim \mathrm{Cat}(\pi_0), \quad z_t \mid z_{t-1} \sim \mathrm{Cat}(P_t[z_{t-1}, :]), \quad
    x_0 \sim \mathcal{N}(m_0, \Sigma_0), \quad x_t \mid x_{t-1}, z_t \sim \mathcal{N}(F_{z_t} x_{t-1} + b_{z_t} + B_{z_t} u_t, Q_{z_t}), \quad
    y_t \mid x_t, z_t \sim \mathcal{N}(H_{z_t} x_t + d_{z_t} + D_{z_t} u_t, R_{z_t}), \qquad t = 0, \ldots, T-1.

Exact inference is intractable, so :class:`SLDS` fits the structured mean-field posterior
:math:`q(z) q(x)` of Laplace variational EM (Zoltowski, Pillow and Linderman, 2020), ported
from ``lindermanlab/ssm``: :math:`q(z)` is a Markov chain updated with the HMM smoother and
:math:`q(x)` a Gaussian with block-tridiagonal precision updated with a Newton step at its mode.
"""
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from fastprogress.fastprogress import progress_bar
from jax import jit, lax, vmap
from jax.tree_util import tree_map
from jaxtyping import Array, Float, Int

from dynamax.hidden_markov_model.inference import hmm_posterior_mode
from dynamax.hidden_markov_model.models.initial import ParamsStandardHMMInitialState, StandardHMMInitialState
from dynamax.linear_gaussian_ssm.block_tridiag import BlockTridiagPotentials, block_tridiag_mvn_sample
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.slds.inference import (
    ContinuousPosteriorSLDS, DiscretePosteriorSLDS, PosteriorSLDS,
    initialize_posterior, log_space_hmm_posterior_sample, slds_elbo, update_continuous_posterior,
    update_discrete_posterior)
from dynamax.ssm import SSM
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.utils import ensure_array_has_batch_dim


# ---- sufficient statistics ------------------------------------------------------------------
class DynamicsStats(NamedTuple):
    r"""Expected statistics of the dynamics regression :math:`x_t \approx \Theta_k \phi_t`,
    :math:`\phi_t = [x_{t-1}; u_t; 1]`, weighted by :math:`\mathbb{E}[z_t = k]` over :math:`t \geq 1`,
    plus the moments of :math:`x_0`."""
    Ex0: Float[Array, " state_dim"]
    Ex0x0T: Float[Array, "state_dim state_dim"]
    sum_w: Float[Array, " num_states"]
    sum_phiphiT: Float[Array, "num_states num_regressors num_regressors"]
    sum_phixT: Float[Array, "num_states num_regressors state_dim"]
    sum_xxT: Float[Array, "num_states state_dim state_dim"]


class EmissionStats(NamedTuple):
    r"""Expected statistics of the emission regression :math:`y_t \approx \Theta_k \psi_t`,
    :math:`\psi_t = [x_t; u_t; 1]`, weighted by :math:`\mathbb{E}[z_t = k]` (or 1 with a single subspace)."""
    sum_w: Float[Array, " num_subspaces"]
    sum_psipsiT: Float[Array, "num_subspaces num_regressors num_regressors"]
    sum_psiyT: Float[Array, "num_subspaces num_regressors emission_dim"]
    sum_yyT: Float[Array, "num_subspaces emission_dim emission_dim"]


class RecurrentTransStats(NamedTuple):
    """Statistics of recurrent transitions: pairwise expectations plus the samples of x and the
    inputs the M-step objective needs."""
    trans_probs: Float[Array, "num_timesteps_minus_1 num_states num_states"]
    x_samples: Float[Array, "num_samples num_timesteps state_dim"]
    inputs: Float[Array, "num_timesteps input_dim"]


class SuffStatsSLDS(NamedTuple):
    """Expected sufficient statistics of one sequence, as returned by :meth:`SLDS.e_step`."""
    initial: Float[Array, " num_states"]
    transitions: Any
    dynamics: DynamicsStats
    emissions: EmissionStats


# ---- component interfaces -------------------------------------------------------------------
class SLDSTransitions(ABC):
    r"""Discrete transitions :math:`p(z_{t+1} \mid z_t, x_t, u_{t+1})`.

    ``depends_on_x`` is a static class attribute: recurrent transitions need Monte Carlo
    samples of :math:`x` and a Newton (not closed-form) update of :math:`q(x)`.
    """
    depends_on_x: bool = False

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Parameters and properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Log prior of the parameters."""

    @abstractmethod
    def transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """(K, K) if stationary, else (T-1, K, K) with entry t linking z_t and z_{t+1}."""

    @abstractmethod
    def log_transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Row-normalized log transition matrices, same shape as :meth:`transition_matrices`."""

    @abstractmethod
    def transition_row(self, params: ParameterSet, state: Array, x: Array, next_input: Array) -> Array:
        """p(z_{t+1} = . | z_t = state, x_t = x, u_{t+1} = next_input), shape (K,); for sampling."""

    @abstractmethod
    def expected_log_trans_prob(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Scalar:
        """sum over t, i, j of E[z_t = i, z_{t+1} = j] log P_t(x)[i, j]."""

    @abstractmethod
    def neg_hessian_x(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Array:
        """Negative Hessian in x_t of the expected log transition term, (T-1, D, D), added to J_diag[:-1]."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, qz: DiscretePosteriorSLDS, x_samples: Array, inputs: Array) -> Any:
        """Per-sequence statistics for the M-step."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """State carried between M-steps (an optimizer state, or None)."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: Any, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """M-step on batched statistics (leading batch axis)."""


class SLDSDynamics(ABC):
    r"""Continuous dynamics :math:`p(x_t \mid x_{t-1}, z_t, u_t)` and the initial distribution of :math:`x_0`."""
    is_quadratic_in_x: bool = True

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Parameters and properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Log prior of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """(T, K): row 0 is the initial density for every state, rows t >= 1 the transition densities."""

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, qx: ContinuousPosteriorSLDS, inputs: Array) -> Array:
        """(T, K): the same, in expectation under q(x)."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """sum_t sum_k Ez[t, k] log p(x_t | x_{t-1}, z_t = k) at the point x."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, smoothed_probs: Array, inputs: Array) -> Tuple[Array, Array]:
        """(J_diag (T, D, D), J_lower (T-1, D, D)) of the negative Hessian of :meth:`expected_log_joint_x`."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> DynamicsStats:
        """Expected statistics under q(x)."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """State carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: DynamicsStats, m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParameterSet, Any]:
        """M-step on batched statistics."""


class SLDSEmissions(ABC):
    r"""Emissions :math:`p(y_t \mid x_t, z_t, u_t)`."""
    is_quadratic_in_x: bool = True
    single_subspace: bool = True

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        """Shape of one emission."""

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Parameters and properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Log prior of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, emissions: Array, x: Array, inputs: Array, mask=None) -> Array:
        """(T, Keff) log p(y_t | x_t, z_t = k, u_t)."""

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, emissions: Array, qx: ContinuousPosteriorSLDS, inputs: Array, mask=None) -> Array:
        """(T, Keff): the same, in expectation under q(x)."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array, mask=None) -> Scalar:
        """sum_t sum_k w[t, k] log p(y_t | x_t, z_t = k) at the point x."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array, mask=None) -> Array:
        """(T, D, D) negative Hessian of :meth:`expected_log_joint_x` in x_t."""

    @abstractmethod
    def invert(self, params: ParameterSet, emissions: Array, inputs: Array) -> Array:
        """A point estimate of x_{0:T-1} from the emissions, for the initial q(x)."""

    @abstractmethod
    def state_weights(self, smoothed_probs: Array) -> Array:
        """(T, Keff) weights of the per-subspace terms: ``smoothed_probs``, or a column of ones with a single subspace."""

    @abstractmethod
    def expected_emissions(self, params: ParameterSet, means: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """(T, N): sum_k w[t, k] E[y_t | x_t = means[t], z_t = k, u_t]; used by :meth:`SLDS.smooth`."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, emissions: Array, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array, mask=None) -> EmissionStats:
        """Expected statistics under q(x)."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """State carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: EmissionStats, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """M-step on batched statistics."""


class SLDSInitialState(StandardHMMInitialState):
    r"""Initial discrete distribution with ``ssm``'s M-step, :math:`\pi_0 \propto \max(\sum_b \mathbb{E}[z_{b,0}] + \beta - 1, 0) + 10^{-8}`
    (the floor scaled by the total count). ``StandardHMMInitialState``'s Dirichlet-mode update is NaN
    whenever a count is zero at concentration 1."""

    def __init__(self, num_states: int, initial_probs_concentration: Union[Scalar, Array] = 1.0):
        super().__init__(num_states, initial_probs_concentration)

    def log_prior(self, params: ParamsStandardHMMInitialState) -> Scalar:
        """Dirichlet log prior; zero for a single state, where the Dirichlet is undefined."""
        if self.num_states == 1:
            return 0.0
        return super().log_prior(params)

    def m_step(self, params, props, batch_stats, m_step_state):
        """Floored, normalized expected counts."""
        if props.probs.trainable:
            counts = jnp.maximum(batch_stats.sum(axis=0) + (self.initial_probs_concentration - 1.0), 0.0)
            counts = counts + 1e-8 * jnp.maximum(counts.sum(), 1.0)
            params = params._replace(probs=counts / counts.sum())
        return params, m_step_state


# ---- driver -------------------------------------------------------------------------------
class ParamsSLDS(NamedTuple):
    """Parameters of an SLDS: one entry per component."""
    initial: Any
    transitions: Any
    dynamics: Any
    emissions: Any


def _damp_leaf(old: Array, new: Array, prop: ParameterProperties, alpha: float) -> Array:
    """alpha * old + (1 - alpha) * new for one trainable leaf. Probability vectors (``SoftmaxCentered``
    constrainer) are mixed in log space and renormalized, as ``ssm`` mixes its log parameters, so exact
    zeros stay zero; other leaves are mixed in their unconstrained space."""
    if not prop.trainable:
        return new
    if isinstance(prop.constrainer, tfb.SoftmaxCentered):
        return jax.nn.softmax(alpha * jnp.log(old) + (1 - alpha) * jnp.log(new), axis=-1)
    if prop.constrainer is None:
        return alpha * old + (1 - alpha) * new
    mixed = alpha * prop.constrainer.inverse(old) + (1 - alpha) * prop.constrainer.inverse(new)
    return prop.constrainer.forward(mixed)


def _damp(params: ParamsSLDS, new_params: ParamsSLDS, props: ParamsSLDS, alpha: float) -> ParamsSLDS:
    """alpha * old + (1 - alpha) * new on the trainable leaves of initial, transitions and
    emissions (see :func:`_damp_leaf`); the closed-form dynamics are never damped (as ``ssm``)."""
    if alpha == 0.0:
        return new_params

    def _mix(component_old, component_new, component_props):
        is_leaf = lambda node: isinstance(node, ParameterProperties)
        return tree_map(lambda o, n, p: _damp_leaf(o, n, p, alpha), component_old, component_new, component_props, is_leaf=is_leaf)

    return ParamsSLDS(initial=_mix(params.initial, new_params.initial, props.initial),
                      transitions=_mix(params.transitions, new_params.transitions, props.transitions),
                      dynamics=new_params.dynamics,
                      emissions=_mix(params.emissions, new_params.emissions, props.emissions))


class SLDS(SSM):
    r"""Switching linear dynamical system fitted by Laplace variational EM.

    :param num_states: number of discrete states :math:`K`
    :param state_dim: continuous state dimension :math:`D`
    :param initial_component: :class:`SLDSInitialState`
    :param transition_component: an :class:`SLDSTransitions`
    :param dynamics_component: an :class:`SLDSDynamics`
    :param emission_component: an :class:`SLDSEmissions`
    :param num_inner_sweeps: coordinate-ascent sweeps over :math:`q(z), q(x)` per EM iteration
    :param num_samples: Monte Carlo samples of :math:`x` for recurrent transition terms
    :param cold_start_sweeps: sweeps run by :meth:`e_step` when no posterior is supplied
    :param newton_tolerance: Newton decrement tolerance of the :math:`q(x)` update
    :param newton_maxiter: Newton iteration cap of the :math:`q(x)` update
    :param damping: :math:`\alpha` of the parameter update :math:`\alpha\,\theta_\mathrm{old} + (1-\alpha)\,\theta_\mathrm{new}` (``ssm``: 0.5; 0 disables)
    :param perturb_unused_states: replace the dynamics of a state with less than one expected observation by a
        perturbed copy of a used state at each M-step (``ssm``); this can lower the objective, so switch it off
        for a strictly monotone ``fit_em`` at ``damping=0``
    :param initial_variance: variance of the initial :math:`q(x)` around the inverted emissions
    :param jitter: added to the diagonal of every block-tridiagonal Cholesky (a bias; 0 is exact)
    """

    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 initial_component: SLDSInitialState,
                 transition_component: SLDSTransitions,
                 dynamics_component: SLDSDynamics,
                 emission_component: SLDSEmissions,
                 *,
                 num_inner_sweeps: int = 1,
                 num_samples: int = 1,
                 cold_start_sweeps: int = 10,
                 newton_tolerance: float = 1e-4,
                 newton_maxiter: int = 100,
                 damping: float = 0.5,
                 perturb_unused_states: bool = True,
                 initial_variance: float = 0.01,
                 jitter: float = 0.0):
        self.num_states = num_states
        self.state_dim = state_dim
        self.initial_component = initial_component
        self.transition_component = transition_component
        self.dynamics_component = dynamics_component
        self.emission_component = emission_component
        self.num_inner_sweeps = num_inner_sweeps
        self.num_samples = num_samples
        self.cold_start_sweeps = cold_start_sweeps
        self.newton_tolerance = newton_tolerance
        self.newton_maxiter = newton_maxiter
        self.damping = damping
        self.perturb_unused_states = perturb_unused_states
        self.initial_variance = initial_variance
        self.jitter = jitter

    # ---- static structure -------------------------------------------------------------------
    @property
    def emission_shape(self) -> Tuple[int]:
        """Shape of one emission."""
        return self.emission_component.emission_shape

    @property
    def inputs_shape(self) -> Tuple[int]:
        """Shape of one input, ``(0,)`` without inputs."""
        return (self.input_dim,)

    @property
    def input_dim(self) -> int:
        """Input dimension (0 without inputs)."""
        return self.dynamics_component.input_dim

    @property
    def quadratic(self) -> bool:
        """True when the q(x) objective is exactly quadratic, so one Newton step is exact."""
        return (not self.transition_component.depends_on_x) \
            and self.dynamics_component.is_quadratic_in_x and self.emission_component.is_quadratic_in_x

    @property
    def _components(self):
        return self.transition_component, self.dynamics_component, self.emission_component

    def _full_inputs(self, inputs: Optional[Array], num_timesteps: int) -> Array:
        """Zero-column inputs when None."""
        return jnp.zeros((num_timesteps, self.input_dim)) if inputs is None else inputs

    # ---- SSM abstract methods ---------------------------------------------------------------
    def initial_distribution(self, params, inputs=None):
        raise NotImplementedError("SLDS defines sample() directly; the joint (z, x) distributions are not exposed.")

    def transition_distribution(self, params, state, inputs=None):
        raise NotImplementedError("SLDS defines sample() directly; the joint (z, x) distributions are not exposed.")

    def emission_distribution(self, params, state, inputs=None):
        raise NotImplementedError("SLDS defines sample() directly; the joint (z, x) distributions are not exposed.")

    def log_prob(self, params, states, emissions, inputs=None):
        raise NotImplementedError("The SLDS likelihood is intractable; use fit_em / approximate_posterior (ELBO).")

    def marginal_log_prob(self, params, emissions, inputs=None):
        raise NotImplementedError("The SLDS marginal likelihood is intractable; use approximate_posterior (ELBO).")

    def fit_sgd(self, *args, **kwargs):
        raise NotImplementedError("The SLDS marginal likelihood is intractable; use fit_em.")

    # ---- parameters -------------------------------------------------------------------------
    def initialize(self,
                   key: PRNGKeyT = jr.PRNGKey(0),
                   method: str = "prior",
                   initial_params: Optional[dict] = None,
                   transition_params: Optional[dict] = None,
                   dynamics_params: Optional[dict] = None,
                   emission_params: Optional[dict] = None) -> Tuple[ParamsSLDS, ParamsSLDS]:
        """Parameters and properties.

        ``method="prior"`` draws each component from its prior. The ``*_params`` dicts override
        individual arrays (e.g. ``dynamics_params=dict(weights=F)``).
        """
        k1, k2, k3, k4 = jr.split(key, 4)
        params_initial, props_initial = self.initial_component.initialize(k1, "prior", **(initial_params or {}))
        params_trans, props_trans = self.transition_component.initialize(k2, "prior", **(transition_params or {}))
        params_dyn, props_dyn = self.dynamics_component.initialize(k3, "prior", **(dynamics_params or {}))
        params_emis, props_emis = self.emission_component.initialize(k4, "prior", **(emission_params or {}))
        params = ParamsSLDS(params_initial, params_trans, params_dyn, params_emis)
        props = ParamsSLDS(props_initial, props_trans, props_dyn, props_emis)
        if method != "prior":
            raise ValueError(f"Unknown initialization method {method!r}")
        return params, props

    def log_prior(self, params: ParamsSLDS) -> Scalar:
        """Sum of the components' log priors."""
        return (self.initial_component.log_prior(params.initial)
                + self.transition_component.log_prior(params.transitions)
                + self.dynamics_component.log_prior(params.dynamics)
                + self.emission_component.log_prior(params.emissions))

    # ---- variational E-step -----------------------------------------------------------------
    def initialize_posterior(self, params: ParamsSLDS, emissions: Array, inputs: Optional[Array] = None) -> PosteriorSLDS:
        """A starting posterior: uniform q(z) and q(x) = N(inverted emissions, initial_variance I)."""
        inputs = self._full_inputs(inputs, emissions.shape[0])
        return initialize_posterior(self, params, emissions, inputs)

    def _sample_x(self, key: PRNGKeyT, qx: ContinuousPosteriorSLDS) -> Array:
        """num_samples draws from q(x), (S, T, D); zeros when no component needs them."""
        if not self.transition_component.depends_on_x:
            return jnp.zeros((self.num_samples,) + qx.smoothed_means.shape)
        potentials = BlockTridiagPotentials(qx.J_diag, qx.J_lower, qx.h)
        filtered = (qx.filtered_precisions, qx.filtered_linear_potentials)
        return vmap(lambda k: block_tridiag_mvn_sample(k, potentials, filtered, self.jitter))(jr.split(key, self.num_samples))

    def _sweep(self, params: ParamsSLDS, emissions: Array, inputs: Array, posterior: PosteriorSLDS, key: PRNGKeyT
               ) -> Tuple[PosteriorSLDS, Array, Array]:
        """One coordinate-ascent sweep: q(z) then q(x); returns the new posterior and two
        independent sets of samples from the new q(x), for the statistics and for the ELBO."""
        k_pre, k_post, k_elbo = jr.split(key, 3)
        x_pre = self._sample_x(k_pre, posterior.continuous)
        qz = update_discrete_posterior(self, params, posterior.continuous, x_pre, emissions, inputs)
        qx = update_continuous_posterior(self, params, qz, posterior.continuous, emissions, inputs)
        new_posterior = PosteriorSLDS(discrete=qz, continuous=qx)
        return new_posterior, self._sample_x(k_post, qx), self._sample_x(k_elbo, qx)

    def _collect_suff_stats(self, params: ParamsSLDS, emissions: Array, inputs: Array, posterior: PosteriorSLDS, x_samples: Array) -> SuffStatsSLDS:
        """Expected sufficient statistics of one sequence."""
        qz, qx = posterior.discrete, posterior.continuous
        return SuffStatsSLDS(
            initial=qz.smoothed_probs[0],
            transitions=self.transition_component.collect_suff_stats(params.transitions, qz, x_samples, inputs),
            dynamics=self.dynamics_component.collect_suff_stats(params.dynamics, qz.smoothed_probs, qx, inputs),
            emissions=self.emission_component.collect_suff_stats(params.emissions, emissions, qz.smoothed_probs, qx, inputs))

    def _elbo(self, params: ParamsSLDS, emissions: Array, inputs: Array, posterior: PosteriorSLDS, x_samples: Array) -> Scalar:
        """Evidence lower bound of one sequence."""
        return slds_elbo(self, params, posterior, emissions, inputs, x_samples)

    def e_step(self, params: ParamsSLDS, emissions: Array, inputs: Optional[Array] = None,
               key: PRNGKeyT = jr.PRNGKey(0), posterior: Optional[PosteriorSLDS] = None
               ) -> Tuple[SuffStatsSLDS, Scalar]:
        """Expected sufficient statistics and ELBO of one sequence.

        Without ``posterior`` the variational posterior is initialized and ``cold_start_sweeps``
        sweeps are run (``SSM.fit_em`` then performs cold-start variational EM, which is valid but
        slower than :meth:`fit_em`, which warm-starts the posterior across iterations). ``key`` only
        matters for recurrent transitions (Monte Carlo samples of x); ``SSM.fit_em`` reuses the default
        every iteration, :meth:`fit_em` threads fresh keys.
        """
        inputs = self._full_inputs(inputs, emissions.shape[0])
        num_sweeps = self.num_inner_sweeps if posterior is not None else self.cold_start_sweeps
        if posterior is None:
            posterior = self.initialize_posterior(params, emissions, inputs)

        def _body(i, carry):
            q, _, _ = carry
            return self._sweep(params, emissions, inputs, q, jr.fold_in(key, i))

        shape = (self.num_samples, emissions.shape[0], self.state_dim)
        posterior, x_post, x_elbo = lax.fori_loop(0, num_sweeps, _body, (posterior, jnp.zeros(shape), jnp.zeros(shape)))
        stats = self._collect_suff_stats(params, emissions, inputs, posterior, x_post)
        return stats, self._elbo(params, emissions, inputs, posterior, x_elbo)

    # ---- M-step -----------------------------------------------------------------------------
    def initialize_m_step_state(self, params: ParamsSLDS, props: ParamsSLDS):
        """Per-component M-step states."""
        return (self.initial_component.initialize_m_step_state(params.initial, props.initial),
                self.transition_component.initialize_m_step_state(params.transitions, props.transitions),
                self.dynamics_component.initialize_m_step_state(params.dynamics, props.dynamics),
                self.emission_component.initialize_m_step_state(params.emissions, props.emissions))

    def m_step(self, params: ParamsSLDS, props: ParamsSLDS, batch_stats: SuffStatsSLDS, m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParamsSLDS, Any]:
        """M-step of every component on batched statistics; ``key`` enables ``ssm``'s perturbation
        of unused states in the dynamics."""
        s_init, s_trans, s_dyn, s_emis = m_step_state
        p_init, s_init = self.initial_component.m_step(params.initial, props.initial, batch_stats.initial, s_init)
        p_trans, s_trans = self.transition_component.m_step(params.transitions, props.transitions, batch_stats.transitions, s_trans)
        p_dyn, s_dyn = self.dynamics_component.m_step(params.dynamics, props.dynamics, batch_stats.dynamics, s_dyn, key=key)
        p_emis, s_emis = self.emission_component.m_step(params.emissions, props.emissions, batch_stats.emissions, s_emis)
        return ParamsSLDS(p_init, p_trans, p_dyn, p_emis), (s_init, s_trans, s_dyn, s_emis)

    # ---- fitting ----------------------------------------------------------------------------
    def _batch(self, emissions, inputs):
        """Emissions and inputs with a leading batch axis; zero-column inputs when None."""
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        if inputs is None:
            batch_inputs = jnp.zeros(batch_emissions.shape[:2] + (self.input_dim,))
        else:
            batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)
        return batch_emissions, batch_inputs

    def _run(self, params, props, batch_emissions, batch_inputs, num_iters, key, posterior, learn, verbose, print_every):
        """Shared loop of fit_em (learn=True) and approximate_posterior (learn=False)."""
        num_sequences, num_timesteps = batch_emissions.shape[:2]
        sample_shape = (num_sequences, self.num_samples, num_timesteps, self.state_dim)
        if posterior is None:
            posterior = vmap(self.initialize_posterior, (None, 0, 0))(params, batch_emissions, batch_inputs)
        m_step_state = self.initialize_m_step_state(params, props) if learn else None

        def _step(carry, _):
            params, m_step_state, posterior, key = carry
            key, k_sweep, k_m = jr.split(key, 3)

            def _inner(i, inner_carry):
                q, _, _ = inner_carry
                keys = jr.split(jr.fold_in(k_sweep, i), num_sequences)
                return vmap(self._sweep, (None, 0, 0, 0, 0))(params, batch_emissions, batch_inputs, q, keys)

            posterior, x_post, x_elbo = lax.fori_loop(
                0, self.num_inner_sweeps, _inner, (posterior, jnp.zeros(sample_shape), jnp.zeros(sample_shape)))
            elbo = vmap(self._elbo, (None, 0, 0, 0, 0))(params, batch_emissions, batch_inputs, posterior, x_elbo).sum()
            if learn:
                objective = self.log_prior(params) + elbo
                stats = vmap(self._collect_suff_stats, (None, 0, 0, 0, 0))(params, batch_emissions, batch_inputs, posterior, x_post)
                new_params, m_step_state = self.m_step(params, props, stats, m_step_state,
                                                      key=k_m if self.perturb_unused_states else None)
                params = _damp(params, new_params, props, self.damping)
            else:
                objective = elbo
            return (params, m_step_state, posterior, key), objective

        @partial(jit, static_argnums=1)
        def _run_steps(carry, num_steps):
            return lax.scan(_step, carry, None, length=num_steps)

        carry = (params, m_step_state, posterior, key)
        if num_iters <= 0:
            return params, jnp.array([]), posterior
        if not verbose:
            (params, _, posterior, _), objectives = _run_steps(carry, num_iters)
            return params, objectives, posterior
        if not isinstance(print_every, Integral) or print_every < 1:
            raise ValueError(f"print_every must be a positive integer, got {print_every!r}")
        objectives = []
        pbar = progress_bar(range(num_iters))
        pbar.update(0)
        for start in range(0, num_iters, print_every):
            num_steps = min(print_every, num_iters - start)
            carry, chunk = _run_steps(carry, num_steps)
            objectives.append(chunk)
            pbar.update(start + num_steps)
        params, _, posterior, _ = carry
        return params, jnp.concatenate(objectives), posterior

    def fit_em(self,
               params: ParamsSLDS,
               props: ParamsSLDS,
               emissions: Array,
               inputs: Optional[Array] = None,
               num_iters: int = 50,
               key: PRNGKeyT = jr.PRNGKey(0),
               posterior: Optional[PosteriorSLDS] = None,
               return_posterior: bool = False,
               verbose: bool = True,
               print_every: int = 1):
        r"""Laplace variational EM.

        Each iteration runs ``num_inner_sweeps`` coordinate-ascent sweeps over :math:`q(z), q(x)`
        (warm-started from the previous iteration), records
        ``log_prior(params) + sum_b ELBO_b`` before the M-step, then updates the parameters with
        damping ``damping``.

        Args:
            params, props: parameters and properties, from :meth:`initialize`.
            emissions: one sequence (T, N) or a batch (B, T, N).
            inputs: optional inputs, (T, M) or (B, T, M).
            num_iters: EM iterations.
            key: PRNG key for the Monte Carlo samples and the unused-state perturbation.
            posterior: optional posterior to warm-start from (batched).
            return_posterior: also return the batched posterior of the last E-step, which was fitted
                under the parameters *before* the final M-step (as ``ssm``); refresh it with
                :meth:`approximate_posterior` (``posterior=``, a few iterations) for the returned parameters.
            verbose, print_every: progress bar, as :meth:`SSM.fit_em`.

        Returns:
            ``(params, log_probs)``, or ``(params, log_probs, posterior)`` with ``return_posterior``.
        """
        batch_emissions, batch_inputs = self._batch(emissions, inputs)
        params, log_probs, posterior = self._run(params, props, batch_emissions, batch_inputs, num_iters, key,
                                                 posterior, True, verbose, print_every)
        return (params, log_probs, posterior) if return_posterior else (params, log_probs)

    def approximate_posterior(self,
                              params: ParamsSLDS,
                              emissions: Array,
                              inputs: Optional[Array] = None,
                              num_iters: int = 50,
                              key: PRNGKeyT = jr.PRNGKey(0),
                              posterior: Optional[PosteriorSLDS] = None,
                              verbose: bool = False,
                              print_every: int = 1) -> Tuple[PosteriorSLDS, Array]:
        """Variational posterior for fixed parameters: ``num_iters`` iterations of ``num_inner_sweeps``
        coordinate-ascent sweeps each, no M-step.

        Returns the batched posterior (leading batch axis even for one sequence) and the ELBO
        after each iteration.
        """
        batch_emissions, batch_inputs = self._batch(emissions, inputs)
        _, elbos, posterior = self._run(params, None, batch_emissions, batch_inputs, num_iters, key,
                                        posterior, False, verbose, print_every)
        return posterior, elbos

    # ---- summaries of a posterior -----------------------------------------------------------
    def _model_potentials(self, params: ParamsSLDS, emissions: Array, x: Array, inputs: Array):
        """(initial probs, transition matrices, log likelihoods) of the model's discrete chain at a known x."""
        inputs = self._full_inputs(inputs, emissions.shape[0])
        ll = self.dynamics_component.log_likelihoods(params.dynamics, x, inputs)
        if not self.emission_component.single_subspace:
            ll = ll + self.emission_component.log_likelihoods(params.emissions, emissions, x, inputs)
        return params.initial.probs, self.transition_component.transition_matrices(params.transitions, x, inputs), ll

    def most_likely_states(self, params: ParamsSLDS, emissions: Array, posterior: PosteriorSLDS,
                           inputs: Optional[Array] = None) -> Int[Array, " num_timesteps"]:
        """Viterbi path of the model's discrete chain with x fixed at the posterior mean (``ssm``)."""
        return hmm_posterior_mode(*self._model_potentials(params, emissions, posterior.continuous.smoothed_means, inputs))

    def smooth(self, params: ParamsSLDS, posterior: PosteriorSLDS, inputs: Optional[Array] = None) -> Array:
        """Posterior-expected emissions, sum_k Ez[t, k] (H_k mean_t + D_k u_t + d_k)."""
        means = posterior.continuous.smoothed_means
        inputs = self._full_inputs(inputs, means.shape[0])
        return self.emission_component.expected_emissions(params.emissions, means, posterior.discrete.smoothed_probs, inputs)

    def sample_posterior(self, key: PRNGKeyT, params: ParamsSLDS, posterior: PosteriorSLDS, num_samples: int = 1
                         ) -> Tuple[Int[Array, "num_samples num_timesteps"], Float[Array, "num_samples num_timesteps state_dim"]]:
        """Joint samples of (z, x) from the variational posterior q(z) q(x)."""
        qz, qx = posterior.discrete, posterior.continuous
        k_z, k_x = jr.split(key)
        states = vmap(lambda k: log_space_hmm_posterior_sample(k, qz.initial_probs, qz.log_transition_matrices, qz.log_likelihoods))(jr.split(k_z, num_samples))
        potentials = BlockTridiagPotentials(qx.J_diag, qx.J_lower, qx.h)
        filtered = (qx.filtered_precisions, qx.filtered_linear_potentials)
        xs = vmap(lambda k: block_tridiag_mvn_sample(k, potentials, filtered, self.jitter))(jr.split(k_x, num_samples))
        return states, xs

    # ---- generative model -------------------------------------------------------------------
    def sample(self, params: ParamsSLDS, key: PRNGKeyT, num_timesteps: int, inputs: Optional[Array] = None
               ) -> Tuple[Int[Array, " num_timesteps"], Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]]:
        """Sample discrete states, continuous states and emissions."""
        inputs = self._full_inputs(inputs, num_timesteps)
        dyn, emis = self.dynamics_component, self.emission_component
        key_z0, key_x0, key_y0, key_rest = jr.split(key, 4)
        z0 = tfd.Categorical(probs=params.initial.probs).sample(seed=key_z0)
        x0 = params.dynamics.initial_mean + jnp.linalg.cholesky(params.dynamics.initial_cov) @ jr.normal(key_x0, (self.state_dim,))
        y0 = emis.sample_emission(key_y0, params.emissions, z0, x0, inputs[0])

        def _step(carry, args):
            z_prev, x_prev = carry
            k, u_next = args
            k_z, k_x, k_y = jr.split(k, 3)
            row = self.transition_component.transition_row(params.transitions, z_prev, x_prev, u_next)
            z = tfd.Categorical(probs=row).sample(seed=k_z)
            x = dyn.sample_transition(k_x, params.dynamics, z, x_prev, u_next)
            y = emis.sample_emission(k_y, params.emissions, z, x, u_next)
            return (z, x), (z, x, y)

        _, (zs, xs, ys) = lax.scan(_step, (z0, x0), (jr.split(key_rest, num_timesteps - 1), inputs[1:]))
        return (jnp.concatenate([z0[None], zs]), jnp.concatenate([x0[None], xs]), jnp.concatenate([y0[None], ys]))
