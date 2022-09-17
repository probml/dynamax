from functools import partial

from jax import vmap
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import register_pytree_node_class, tree_map
from ssm_jax.abstractions import SSM
from ssm_jax.distributions import InverseWishart, MatrixNormalPrecision as MN
from ssm_jax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, LGSSMParams
from ssm_jax.parameters import ParameterProperties
from ssm_jax.utils import PSDToRealBijector
import  tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

tfd = tfp.distributions
tfb = tfp.bijectors


@register_pytree_node_class
class LinearGaussianSSM(SSM):
    """
    Linear Gaussian State Space Model is defined as follows:
    p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
    p(y_t | z_t) = N(y_t | H_t z_t + D_t u_t + d_t, R_t)
    p(z_1) = N(z_1 | m, S)
    where z_t = hidden, y_t = observed, u_t = inputs,
    dynamics_matrix = F
    dynamics_covariance = Q
    emission_matrix = H
    emissions_covariance = R
    initial_mean = mu_{1|0}
    initial_covariance = Sigma_{1|0}
    Optional parameters (default to 0)
    dynamics_input_matrix = B
    dynamics_bias = b
    emission_input_matrix = D
    emission_bias = d
    """

    def __init__(self,
                 state_dim,
                 emission_dim,
                 input_dim=0,
                 has_dynamics_bias=True,
                 has_emissions_bias=True,
                 **kw_priors):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias

        # Initialize prior distributions
        def default_prior(arg, default):
            return kw_priors[arg] if arg in kw_priors else default

        self.initial_mean_prior = default_prior(
            'initial_mean_prior',
            MVN(loc=jnp.zeros(self.state_dim), covariance_matrix=jnp.eye(self.state_dim)))
        self.initial_covariance_prior = default_prior(
            'initial_covariance_prior',
            InverseWishart(df=self.state_dim + 0.1, scale=jnp.eye(self.state_dim)))

        self.dynamics_matrix_prior = default_prior(
            'dynamics_matrix_prior',
            MN(loc=jnp.eye(state_dim),
               row_covariance=jnp.eye(self.state_dim),
               col_precision=jnp.eye(self.state_dim)))
        self.dynamics_input_weights_prior = default_prior(
            'dynamics_input_weights_prior',
            MN(loc=jnp.zeros((state_dim, input_dim)),
               row_covariance=jnp.eye(self.state_dim),
               col_precision=jnp.eye(input_dim)))
        self.dynamics_covariance_prior = default_prior(
            'dynamics_covariance_prior',
            InverseWishart(df=self.state_dim + 0.1, scale=jnp.eye(self.state_dim)))

        if has_dynamics_bias:
            self.dynamics_bias_prior = MVN(loc=jnp.zeros(state_dim), covariance_matrix=jnp.eye(self.state_dim))

        self.emission_matrix_prior = default_prior(
            'emission_matrix_prior',
            MN(loc=jnp.zeros((emission_dim, state_dim)),
               row_covariance=jnp.eye(self.emission_dim),
               col_precision=jnp.eye(self.state_dim)))
        self.emission_input_weights_prior = default_prior(
            'emission_input_weights_prior',
            MN(loc=jnp.zeros((emission_dim, input_dim)),
               row_covariance=jnp.eye(self.emission_dim),
               col_precision=jnp.eye(input_dim)))
        self.emission_covariance_prior = default_prior(
            'emission_covariance_prior',
            InverseWishart(df=self.emission_dim + 0.1, scale=jnp.eye(self.emission_dim)))

        if has_emissions_bias:
            self.emission_bias_prior = MVN(loc=jnp.zeros(emission_dim),
                                           covariance_matrix=jnp.eye(self.emission_dim))

    def random_initialization(self, key):
        m = jnp.zeros(self.state_dim)
        S = jnp.eye(self.state_dim)
        # TODO: Sample a random rotation matrix
        F = 0.99 * jnp.eye(self.state_dim)
        B = jnp.zeros((self.state_dim, self.input_dim))
        b = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        Q = 0.1 * jnp.eye(self.state_dim)
        H = jr.normal(key, (self.emission_dim, self.state_dim))
        D = jnp.zeros((self.emission_dim, self.input_dim))
        d = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        R = 0.1 * jnp.eye(self.emission_dim)

        params = dict(
            initial=dict(mean=m, cov=S),
            dynamics=dict(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=dict(weights=H, bias=d, input_weights=D, cov=R)
        )

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

    def initial_distribution(self, params, **covariates):
        return MVN(params["initial"]["mean"], params["initial"]["cov"])

    def transition_distribution(self, params, state, **covariates):
        input = covariates['inputs'] if 'inputs' in covariates else jnp.zeros(self.input_dim)
        return MVN(
            params["dynamics"]["weights"] @ state + params["dynamics"]["input_weights"] @ input +
            params["dynamics"]["bias"], params["dynamics"]["cov"])

    def emission_distribution(self, params, state, **covariates):
        input = covariates['inputs'] if 'inputs' in covariates else jnp.zeros(self.input_dim)
        return MVN(
            params["emissions"]["weights"] @ state + params["emissions"]["input_weights"] @ input +
            params["emissions"]["bias"], params["emissions"]["cov"])

    def _make_inference_args(self, params):
        return LGSSMParams(initial_mean=params["initial"]["mean"],
                           initial_covariance=params["initial"]["cov"],
                           dynamics_matrix=params["dynamics"]["weights"],
                           dynamics_input_weights=params["dynamics"]["input_weights"],
                           dynamics_bias=params["dynamics"]["bias"],
                           dynamics_covariance=params["dynamics"]["cov"],
                           emission_matrix=params["emissions"]["weights"],
                           emission_input_weights=params["emissions"]["input_weights"],
                           emission_bias=params["emissions"]["bias"],
                           emission_covariance=params["emissions"]["cov"])

    def log_prior(self, params):
        """Return the log prior probability of any model parameters.
        Returns:
            lp (Scalar): log prior probability.
        """
        # log prior of the initial state
        lp = self.initial_mean_prior.log_prob(params["initial"]["mean"])
        lp += self.initial_covariance_prior.log_prob(params["initial"]["cov"])

        # log prior of the dynamics model
        lp += self.dynamics_matrix_prior.log_prob(params["dynamics"]["weights"])
        lp += self.dynamics_input_weights_prior.log_prob(params["dynamics"]["input_weights"])
        lp += self.dynamics_covariance_prior.log_prob(params["dynamics"]["cov"])

        # log prior of the emission model
        lp += self.emission_matrix_prior.log_prob(params["emissions"]["weights"])
        lp += self.emission_input_weights_prior.log_prob(params["emissions"]["input_weights"])
        lp += self.emission_covariance_prior.log_prob(params["emissions"]["cov"])

        # log prior of bias (if needed)
        if self.has_dynamics_bias:
            lp += self.dynamics_bias_prior.log_prob(params["dynamics"]["bias"])
        if self.has_emissions_bias:
            lp += self.emission_bias_prior.log_prob(params["emissions"]["bias"])

        return lp

    def marginal_log_prob(self, params, emissions, inputs=None):
        """Compute log marginal likelihood of observations."""
        filtered_posterior = lgssm_filter(self._make_inference_args(params), emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, params, emissions, inputs=None):
        """Compute filtering distribution."""
        return lgssm_filter(self._make_inference_args(params), emissions, inputs)

    def smoother(self, params, emissions, inputs=None):
        """Compute smoothing distribution."""
        return lgssm_smoother(self._make_inference_args(params), emissions, inputs)

    # Expectation-maximization (EM) code
    def e_step(self, params, batch_emissions, batch_inputs=None):
        """The E-step computes sums of expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        num_batches, num_timesteps = batch_emissions.shape[:2]
        if batch_inputs is None:
            batch_inputs = jnp.zeros((num_batches, num_timesteps, 0))

        def _single_e_step(emissions, inputs):
            # Run the smoother to get posterior expectations
            posterior = lgssm_smoother(self._make_inference_args(params), emissions, inputs)

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

            # expected sufficient statistics for the initial distribution
            Ex0 = posterior.smoothed_means[0]
            Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
            init_stats = (Ex0, Ex0x0T, 1)

            # expected sufficient statistics for the dynamics distribution
            # let zp[t] = [x[t], u[t]] for t = 0...T-2
            # let xn[t] = x[t+1]          for t = 0...T-2
            sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
            sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
            sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
            sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
            if not self._db_indicator:
                dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                                  num_timesteps - 1)

            # more expected sufficient statistics for the emissions
            # let z[t] = [x[t], u[t]] for t = 0...T-1
            sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
            sum_zzT = sum_zzT.at[:self.state_dim, :self.state_dim].add(Vx.sum(0))
            sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
            sum_yyT = emissions.T @ emissions
            emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
            if not self.has_emission_bias:
                emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

            return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

        # TODO: what's the best way to vectorize/parallelize this?
        return vmap(_single_e_step)(batch_emissions, batch_inputs)

    def m_step(self, curr_params, param_props, batch_stats):

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
