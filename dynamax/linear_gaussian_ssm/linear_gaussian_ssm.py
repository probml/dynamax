from functools import partial
import jax
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import tree_map

import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from jaxtyping import Array, Float, PyTree, Bool, Int, Num
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic, Mapping, Callable
import chex
from dataclasses import dataclass


from dynamax.abstractions import SSM
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMMoment, PosteriorLGSSMFiltered, PosteriorLGSSMSmoothed
from dynamax.parameters import ParameterProperties
from dynamax.utils import PSDToRealBijector

ParamsLGSSM = Dict

ParamPropsLGSSM = Dict

SuffStatsLGSSM = Any

_zeros_if_none = lambda x, shp: x if x is not None else jnp.zeros(shp)

class LinearGaussianSSM(SSM):
    """
    Linear Gaussian State Space Model.

    The model is defined as follows:

    p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
    p(y_t | z_t) = N(y_t | H_t z_t + D_t u_t + d_t, R_t)
    p(z_1) = N(z_1 | m, S)

    where

    z_t = hidden variables of size `state_dim`,
    y_t = observed variables of size `emission_dim`
    u_t = input covariates of size `input_dim` (defaults to 0)
    """
    def __init__(self,
                 state_dim,
                 emission_dim,
                 input_dim=0,
                 has_dynamics_bias=True,
                 has_emissions_bias=True):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initialize(
        self,
        key: jr.PRNGKey =jr.PRNGKey(0),
        initial_mean=None,
        initial_covariance=None,
        dynamics_weights=None,
        dynamics_bias=None,
        dynamics_input_weights=None,
        dynamics_covariance=None,
        emission_weights=None,
        emission_bias=None,
        emission_input_weights=None,
        emission_covariance=None
    ) -> Tuple[ParamsLGSSM, ParamPropsLGSSM]:
        """Initialize the model parameters and their corresponding properties."""

        # Arbitrary default values, for demo purposes.
        _initial_mean = jnp.zeros(self.state_dim)
        _initial_covariance = jnp.eye(self.state_dim)
        _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
        _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)
        _emission_weights = jr.normal(key, (self.emission_dim, self.state_dim))
        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # Create nested dictionary of params
        params = dict(
            initial=dict(mean=default(initial_mean, _initial_mean),
                         cov=default(initial_covariance, _initial_covariance)),
            dynamics=dict(weights=default(dynamics_weights, _dynamics_weights),
                          bias=default(dynamics_bias, _dynamics_bias),
                          input_weights=default(dynamics_input_weights, _dynamics_input_weights),
                          cov=default(dynamics_covariance, _dynamics_covariance)),
            emissions=dict(weights=default(emission_weights, _emission_weights),
                           bias=default(emission_bias, _emission_bias),
                           input_weights=default(emission_input_weights, _emission_input_weights),
                           cov=default(emission_covariance, _emission_covariance))
        )

        # The keys of param_props must match those of params!
        param_props = dict(
            initial=dict(mean=ParameterProperties(),
                         cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            dynamics=dict(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            emissions=dict(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        )
        return params, param_props

    def initial_distribution(
        self,
        params: ParamsLGSSM,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        return MVN(params["initial"]["mean"], params["initial"]["cov"])

    def transition_distribution(
        self,
        params: ParamsLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params["dynamics"]["weights"] @ state + params["dynamics"]["input_weights"] @ inputs
        if self.has_dynamics_bias:
            mean += params["dynamics"]["bias"]
        return MVN(mean, params["dynamics"]["cov"])

    def emission_distribution(
        self,
        params: ParamsLGSSM,
        state: Float[Array, "state_dim"], 
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params["emissions"]["weights"] @ state + params["emissions"]["input_weights"] @ inputs
        if self.has_emissions_bias:
            mean += params["emissions"]["bias"]
        return MVN(mean, params["emissions"]["cov"])

    def to_inference_args(
        self,
        params: ParamsLGSSM
    ) -> ParamsLGSSMMoment:
        """Convert params dict to inference container replacing Nones if necessary."""
        dyn_bias = _zeros_if_none(params["dynamics"]["bias"], self.state_dim)
        ems_bias = _zeros_if_none(params["emissions"]["bias"], self.emission_dim)
        return ParamsLGSSMMoment(initial_mean=params["initial"]["mean"],
                           initial_covariance=params["initial"]["cov"],
                           dynamics_weights=params["dynamics"]["weights"],
                           dynamics_input_weights=params["dynamics"]["input_weights"],
                           dynamics_bias=dyn_bias,
                           dynamics_covariance=params["dynamics"]["cov"],
                           emission_weights=params["emissions"]["weights"],
                           emission_input_weights=params["emissions"]["input_weights"],
                           emission_bias=ems_bias,
                           emission_covariance=params["emissions"]["cov"])

    def from_inference_args(
        self,
        params: ParamsLGSSMMoment
    ) -> ParamsLGSSM:
        """Convert params from inference container to dict."""
        return dict(
            initial=dict(mean=params.initial_mean,
                         cov=params.initial_covariance),
            dynamics=dict(weights=params.dynamics_weights,
                          bias=params.dynamics_bias,
                          input_weights=params.dynamics_input_weights,
                          cov=params.dynamics_covariance),
            emissions=dict(weights=params.emission_weights,
                           bias=params.emission_bias,
                           input_weights=params.emission_input_weights,
                           cov=params.emission_covariance)
            )
       

    def log_prior(
        self,
        params: ParamsLGSSM
    ) -> float:
        """Return the log prior probability of model parameters."""
        return 0.0

    def marginal_log_prob(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> float:
        """Compute log marginal likelihood of observations."""
        filtered_posterior = lgssm_filter(self.to_inference_args(params), emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"], 
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorLGSSMFiltered:
        """Compute filtering tfd.Distribution."""
        return lgssm_filter(self.to_inference_args(params), emissions, inputs)

    def smoother(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"], 
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorLGSSMSmoothed:
        """Compute smoothing tfd.Distribution."""
        return lgssm_smoother(self.to_inference_args(params), emissions, inputs)

    def posterior_sample(
        self,
        key: jr.PRNGKey, 
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"], 
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Float[Array, "ntime state_dim"]:
        return lgssm_posterior_sample(key, self.to_inference_args(params), emissions, inputs)

    def posterior_predictive(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        """Compute marginal posterior predictive for each observation.

        Returns:
            means: (T,D) array of E[Y(t,d) | Y(1:T)]
            stds: (T,D) array std[Y(t,d) | Y(1:T)]
        """
        posterior = self.smoother(params, emissions, inputs)
        H = params['emissions']['weights']
        b = params['emissions']['bias']
        R = params['emissions']['cov']
        emission_dim = R.shape[0]
        smoothed_emissions = posterior.smoothed_means @ H.T + b
        smoothed_emissions_cov = H @ posterior.smoothed_covariances @ H.T + R
        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    # Expectation-maximization (EM) code
    def e_step(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "nseq ntime emission_dim"],
        inputs: Optional[Float[Array, "nseq ntime input_dim"]]=None
    ) -> Tuple[SuffStatsLGSSM, float]:
        """The E-step computes sums of expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Run the smoother to get posterior expectations
        posterior = lgssm_smoother(self.to_inference_args(params), emissions, inputs)

        # shorthand
        Ex = posterior.smoothed_means
        Exp = posterior.smoothed_means[:-1]
        Exn = posterior.smoothed_means[1:]
        Vx = posterior.smoothed_covariances
        Vxp = posterior.smoothed_covariances[:-1]
        Vxn = posterior.smoothed_covariances[1:]
        Expxn = posterior.smoothed_cross_covariances

        # Append bias to the inputs
        inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        u = inputs
        y = emissions

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = posterior.smoothed_means[0]
        Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                                num_timesteps - 1)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
        sum_zzT = sum_zzT.at[:self.state_dim, :self.state_dim].add(Vx.sum(0))
        sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        if not self.has_emissions_bias:
            emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

        return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik


    def m_step(
        self,
        params: ParamsLGSSM,
        props: ParamPropsLGSSM,
        batch_stats: SuffStatsLGSSM
    ) -> ParamsLGSSM:
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = jnp.linalg.solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
        m = sum_x0 / N

        FB, Q = fit_linear_regression(*dynamics_stats)
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], None)

        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
            else (HD[:, self.state_dim:], None)

        return dict(
            initial=dict(mean=m, cov=S),
            dynamics=dict(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=dict(weights=H, bias=d, input_weights=D, cov=R)
        )
