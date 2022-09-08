from functools import partial

import blackjax
from jax import jit, vmap
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import register_pytree_node_class, tree_map
from ssm_jax.abstractions import SSM, Parameter
from ssm_jax.distributions import InverseWishart, MatrixNormalPrecision as MN
from ssm_jax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, LGSSMParams
from ssm_jax.utils import PSDToRealBijector
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tqdm.auto import trange


def _get_shape(x, dim):
    return x.shape[1:] if x.ndim == dim + 1 else x.shape


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
                 dynamics_matrix,
                 dynamics_covariance,
                 emission_matrix,
                 emission_covariance,
                 initial_mean=None,
                 initial_covariance=None,
                 dynamics_input_weights=None,
                 dynamics_bias=None,
                 emission_input_weights=None,
                 emission_bias=None,
                 **kw_priors):
        self.emission_dim, self.state_dim = _get_shape(emission_matrix, 2)
        dynamics_input_dim = dynamics_input_weights.shape[
            1] if dynamics_input_weights is not None else 0
        emission_input_dim = emission_input_weights.shape[
            1] if emission_input_weights is not None else 0
        self.input_dim = max(dynamics_input_dim, emission_input_dim)
        self._db_indicator = dynamics_bias is not None
        self._eb_indicator = emission_bias is not None

        # Set optional args to default value if not given
        def default(x, v):
            return x if x is not None else v

        initial_mean = default(initial_mean, jnp.zeros(self.state_dim))
        initial_covariance = default(initial_covariance, jnp.eye(self.state_dim))
        dynamics_input_weights = default(dynamics_input_weights,
                                         jnp.zeros((self.state_dim, self.input_dim)))
        dynamics_bias = default(dynamics_bias, jnp.zeros(self.state_dim))
        emission_input_weights = default(emission_input_weights,
                                         jnp.zeros((self.emission_dim, self.input_dim)))
        emission_bias = default(emission_bias, jnp.zeros(self.emission_dim))

        # Save args
        self._dynamics_matrix = Parameter(dynamics_matrix)
        self._dynamics_covariance = Parameter(dynamics_covariance, bijector=PSDToRealBijector)
        self._emission_matrix = Parameter(emission_matrix)
        self._emission_covariance = Parameter(emission_covariance, bijector=PSDToRealBijector)
        self._initial_mean = Parameter(initial_mean)
        self._initial_covariance = Parameter(initial_covariance, bijector=PSDToRealBijector)
        self._dynamics_input_weights = Parameter(dynamics_input_weights)
        self._dynamics_bias = Parameter(dynamics_bias)
        self._emission_input_weights = Parameter(emission_input_weights)
        self._emission_bias = Parameter(emission_bias)

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
            MN(loc=dynamics_matrix,
               row_covariance=jnp.eye(self.state_dim),
               col_precision=jnp.eye(self.state_dim)))
        self.dynamics_input_weights_prior = default_prior(
            'dynamics_input_weights_prior',
            MN(loc=dynamics_input_weights,
               row_covariance=jnp.eye(self.state_dim),
               col_precision=jnp.eye(dynamics_input_dim)))
        self.dynamics_covariance_prior = default_prior(
            'dynamics_covariance_prior',
            InverseWishart(df=self.state_dim + 0.1, scale=jnp.eye(self.state_dim)))
        self.dynamics_bias_prior = MVN(loc=dynamics_bias, covariance_matrix=jnp.eye(self.state_dim))

        self.emission_matrix_prior = default_prior(
            'emission_matrix_prior',
            MN(loc=emission_matrix,
               row_covariance=jnp.eye(self.emission_dim),
               col_precision=jnp.eye(self.state_dim)))
        self.emission_input_weights_prior = default_prior(
            'emission_input_weights_prior',
            MN(loc=emission_input_weights,
               row_covariance=jnp.eye(self.emission_dim),
               col_precision=jnp.eye(emission_input_dim)))
        self.emission_covariance_prior = default_prior(
            'emission_covariance_prior',
            InverseWishart(df=self.emission_dim + 0.1, scale=jnp.eye(self.emission_dim)))
        self.emission_bias_prior = MVN(loc=emission_bias,
                                       covariance_matrix=jnp.eye(self.emission_dim))

        # Check shapes
        assert self._initial_mean.value.shape == (self.state_dim,)
        assert self._initial_covariance.value.shape == (self.state_dim, self.state_dim)
        assert self._dynamics_matrix.value.shape[-2:] == (self.state_dim, self.state_dim)
        assert self._dynamics_input_weights.value.shape[-2:] == (self.state_dim, self.input_dim)
        assert self._dynamics_bias.value.shape[-1:] == (self.state_dim,)
        assert self._dynamics_covariance.value.shape == (self.state_dim, self.state_dim)
        assert self._emission_input_weights.value.shape[-2:] == (self.emission_dim, self.input_dim)
        assert self._emission_bias.value.shape[-1:] == (self.emission_dim,)
        assert self._emission_covariance.value.shape == (self.emission_dim, self.emission_dim)

    @classmethod
    def random_initialization(cls, key, state_dim, emission_dim, input_dim=0):
        m = jnp.zeros(state_dim)
        S = jnp.eye(state_dim)
        # TODO: Sample a random rotation matrix
        F = 0.99 * jnp.eye(state_dim)
        B = jnp.zeros((state_dim, input_dim))
        Q = 0.1 * jnp.eye(state_dim)
        H = jr.normal(key, (emission_dim, state_dim))
        D = jnp.zeros((emission_dim, input_dim))
        R = 0.1 * jnp.eye(emission_dim)
        return cls(dynamics_matrix=F,
                   dynamics_covariance=Q,
                   emission_matrix=H,
                   emission_covariance=R,
                   initial_mean=m,
                   initial_covariance=S,
                   dynamics_input_weights=B,
                   emission_input_weights=D)

    # Properties to get various parameters of the model
    # Parameters of initial state
    @property
    def initial_mean(self):
        return self._initial_mean

    @property
    def initial_covariance(self):
        return self._initial_covariance

    # Parameters of dynamics model
    @property
    def dynamics_matrix(self):
        return self._dynamics_matrix

    @property
    def dynamics_input_weights(self):
        return self._dynamics_input_weights

    @property
    def dynamics_bias(self):
        return self._dynamics_bias

    @property
    def dynamics_covariance(self):
        return self._dynamics_covariance

    # Parameters of emission model
    @property
    def emission_matrix(self):
        return self._emission_matrix

    @property
    def emission_input_weights(self):
        return self._emission_input_weights

    @property
    def emission_bias(self):
        return self._emission_bias

    @property
    def emission_covariance(self):
        return self._emission_covariance

    def initial_distribution(self, **covariates):
        return MVN(self.initial_mean.value, self.initial_covariance.value)

    def transition_distribution(self, state, **covariates):
        input = covariates['inputs'] if 'inputs' in covariates else jnp.zeros(self.input_dim)
        return MVN(
            self.dynamics_matrix.value @ state + self.dynamics_input_weights.value @ input +
            self.dynamics_bias.value, self.dynamics_covariance.value)

    def emission_distribution(self, state, **covariates):
        input = covariates['inputs'] if 'inputs' in covariates else jnp.zeros(self.input_dim)
        return MVN(
            self.emission_matrix.value @ state + self.emission_input_weights.value @ input +
            self.emission_bias.value, self.emission_covariance.value)

    @property
    def params(self):
        return LGSSMParams(initial_mean=self.initial_mean.value,
                           initial_covariance=self.initial_covariance.value,
                           dynamics_matrix=self.dynamics_matrix.value,
                           dynamics_input_weights=self.dynamics_input_weights.value,
                           dynamics_bias=self.dynamics_bias.value,
                           dynamics_covariance=self.dynamics_covariance.value,
                           emission_matrix=self.emission_matrix.value,
                           emission_input_weights=self.emission_input_weights.value,
                           emission_bias=self.emission_bias.value,
                           emission_covariance=self.emission_covariance.value)

    @params.setter
    def params(self, new_param):
        assert isinstance(new_param, LGSSMParams)
        self._initial_mean.value = new_param.initial_mean
        self._initial_covariance.value = new_param.initial_covariance
        self._dynamics_matrix.value = new_param.dynamics_matrix
        self._dynamics_input_weights.value = new_param.dynamics_input_weights
        self._dynamics_bias.value = new_param.dynamics_bias
        self._dynamics_covariance.value = new_param.dynamics_covariance
        self._emission_matrix.value = new_param.emission_matrix
        self._emission_input_weights.value = new_param.emission_input_weights
        self._emission_bias.value = new_param.emission_bias
        self._emission_covariance.value = new_param.emission_covariance

    def log_prior(self):
        """Return the log prior probability of any model parameters.
        Returns:
            lp (Scalar): log prior probability.
        """
        # log prior of the initial state
        lp = self.initial_mean_prior.log_prob(self.initial_mean.value)
        lp += self.initial_covariance_prior.log_prob(self.initial_covariance.value)

        # log prior of the dynamics model
        lp += self.dynamics_matrix_prior.log_prob(self.dynamics_matrix.value)
        lp += self.dynamics_input_weights_prior.log_prob(self.dynamics_input_weights.value)
        lp += self.dynamics_covariance_prior.log_prob(self.dynamics_covariance.value)

        # log prior of the emission model
        lp += self.emission_matrix_prior.log_prob(self.emission_matrix.value)
        lp += self.emission_input_weights_prior.log_prob(self.emission_input_weights.value)
        lp += self.emission_covariance_prior.log_prob(self.emission_covariance.value)

        # log prior of bias (if needed)
        if self._db_indicator:
            lp += self.dynamics_bias_prior.log_prob(self.dynamics_bias.value)
        if self._eb_indicator:
            lp += self.emission_bias_prior.log_prob(self.emission_bias.value)

        return lp

    def marginal_log_prob(self, emissions, inputs=None):
        """Compute log marginal likelihood of observations."""
        filtered_posterior = lgssm_filter(self.params, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, emissions, inputs=None):
        """Compute filtering distribution."""
        return lgssm_filter(self.params, emissions, inputs)

    def smoother(self, emissions, inputs=None):
        """Compute smoothing distribution."""
        return lgssm_smoother(self.params, emissions, inputs)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params):
        initial_mean = unconstrained_params[0]
        initial_covariance = PSDToRealBijector.inverse(unconstrained_params[1])
        dynamics_matrix = unconstrained_params[2]
        dynamics_input_weights = unconstrained_params[3]
        dynamics_bias = unconstrained_params[4]
        dynamics_covariance = PSDToRealBijector.inverse(unconstrained_params[5])
        emission_matrix = unconstrained_params[6]
        emission_input_weights = unconstrained_params[7]
        emission_bias = unconstrained_params[8]
        emission_covariance = PSDToRealBijector.inverse(unconstrained_params[9])
        return cls(dynamics_matrix=dynamics_matrix,
                   dynamics_covariance=dynamics_covariance,
                   emission_matrix=emission_matrix,
                   emission_covariance=emission_covariance,
                   initial_mean=initial_mean,
                   initial_covariance=initial_covariance,
                   dynamics_input_weights=dynamics_input_weights,
                   dynamics_bias=dynamics_bias,
                   emission_input_weights=emission_input_weights,
                   emission_bias=emission_bias)

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions, batch_inputs=None):
        """The E-step computes sums of expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        num_batches, num_timesteps = batch_emissions.shape[:2]
        if batch_inputs is None:
            batch_inputs = jnp.zeros((num_batches, num_timesteps, 0))

        def _single_e_step(emissions, inputs):
            # Run the smoother to get posterior expectations
            posterior = lgssm_smoother(self.params, emissions, inputs)

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
            if not self._eb_indicator:
                emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

            return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

        # TODO: what's the best way to vectorize/parallelize this?
        return vmap(_single_e_step)(batch_emissions, batch_inputs)

    def m_step(self, batch_stats):

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
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self._db_indicator \
            else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self._eb_indicator \
            else (HD[:, self.state_dim:], jnp.zeros(self.emission_dim))

        return LGSSMParams(initial_mean=m,
                           initial_covariance=S,
                           dynamics_matrix=F,
                           dynamics_input_weights=B,
                           dynamics_bias=b,
                           dynamics_covariance=Q,
                           emission_matrix=H,
                           emission_input_weights=D,
                           emission_bias=d,
                           emission_covariance=R)

    def fit_em(self, batch_emissions, batch_inputs=None, num_iters=50):
        """Fit this HMM with Expectation-Maximization (EM).
        """

        @jit
        def em_step(_params):
            self.params = _params
            posterior_stats, marginal_loglikes = self.e_step(batch_emissions, batch_inputs)
            new_param = self.m_step(posterior_stats)
            return new_param, marginal_loglikes.sum()

        log_probs = []
        _params = self.params

        for _ in trange(num_iters):
            _params, marginal_loglik = em_step(_params)
            log_probs.append(marginal_loglik)

        self.params = _params
        return jnp.array(log_probs)

    def fit_hmc(self,
                key,
                sample_size,
                batch_emissions,
                batch_inputs=None,
                warmup_steps=500,
                num_integration_steps=30):
        """Sample parameters of the model using HMC."""

        # The log likelihood that the HMC samples from
        def _logprob(hmc_position, batch_emissions=batch_emissions, batch_inputs=batch_inputs):
            self.unconstrained_params = hmc_position
            batch_lls = vmap(self.marginal_log_prob)(batch_emissions, batch_inputs)
            lp = self.log_prior() + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptation
        hmc_initial_position = self.unconstrained_params
        warmup = blackjax.window_adaptation(blackjax.hmc,
                                            _logprob,
                                            num_steps=warmup_steps,
                                            num_integration_steps=num_integration_steps)
        hmc_initial_state, hmc_kernel, _ = warmup.run(key, hmc_initial_position)
        hmc_kernel = jit(hmc_kernel)

        @jit
        def one_step(current_state, rng_key):
            next_state, _ = hmc_kernel(rng_key, current_state)
            return next_state

        # Start sampling
        keys = iter(jr.split(key, sample_size))
        param_samples = []
        current_state = hmc_initial_state
        for _ in trange(sample_size):
            current_state = one_step(current_state, next(keys))
            param_samples.append(current_state.position)

        # Return list of full parameters, each is an instance of LGSSMParams
        return self._to_complete_parameters(param_samples)

    def _to_complete_parameters(self, unconstrained_params):
        """Transform samples of subset of unconstrained params to samples of complete params,
        each is an instance of LGSSMParams
        """
        items = sorted(self.__dict__.items())
        names = [key for key, prm in items if isinstance(prm, Parameter) and not prm.is_frozen]

        @jit
        def join_params(unconstrained_parameter):
            current_params = self.params
            for i in range(len(unconstrained_parameter)):
                value = getattr(self, names[i]).bijector.inverse(unconstrained_parameter[i])
                setattr(current_params, names[i][1:], value)
            return current_params

        return [join_params(param) for param in unconstrained_params]
