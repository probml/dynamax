from fastprogress.fastprogress import progress_bar
from functools import partial
import jax.random as jr
from jax import jit
from jax import numpy as jnp
from jax.tree_util import tree_map
from dynamax.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.distributions import NormalInverseWishart as NIW
from dynamax.distributions import mniw_posterior_update, niw_posterior_update
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMMoment, lgssm_posterior_sample
from dynamax.linear_gaussian_ssm.linear_gaussian_ssm import LinearGaussianSSM



class LinearGaussianConjugateSSM(LinearGaussianSSM):
    """
    Linear Gaussian State Space Model with conjugate priors for the model parameters.
    The parameters are the same as LG-SSM.
    The priors are as follows:
    p(m, S) = NIW(loc, mean_concentration, df, scale) # normal inverse wishart
    p([F, B, b], Q) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart
    p([H, D, d], R) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart
    """
    def __init__(self,
                 state_dim,
                 emission_dim,
                 input_dim=0,
                 has_dynamics_bias=True,
                 has_emissions_bias=True,
                 **kw_priors):
        super().__init__(state_dim, emission_dim, input_dim, has_dynamics_bias, has_emissions_bias)

        # Initialize prior distributions
        def default_prior(arg, default):
            return kw_priors[arg] if arg in kw_priors else default

        self.initial_prior = default_prior(
            'initial_prior',
            NIW(loc=jnp.zeros(self.state_dim),
                mean_concentration=1.,
                df=self.state_dim + 0.1,
                scale=jnp.eye(self.state_dim)))

        self.dynamics_prior = default_prior(
            'dynamics_prior',
            MNIW(loc=jnp.zeros((self.state_dim, self.state_dim + self.input_dim + self.has_dynamics_bias)),
                 col_precision=jnp.eye(self.state_dim + self.input_dim + self.has_dynamics_bias),
                 df=self.state_dim + 0.1,
                 scale=jnp.eye(self.state_dim)))

        self.emission_prior = default_prior(
            'emission_prior',
            MNIW(loc=jnp.zeros((self.emission_dim, self.state_dim + self.input_dim + self.has_emissions_bias)),
                 col_precision=jnp.eye(self.state_dim + self.input_dim + self.has_emissions_bias),
                 df=self.emission_dim + 0.1,
                 scale=jnp.eye(self.emission_dim)))

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return dict(inputs=(self.input_dim,)) if self.input_dim > 0 else dict()

    def log_prior(self, params):
        """Return the log prior probability of any model parameters.
        Returns:
            lp (Scalar): log prior probability.
        """
        lp = self.initial_prior.log_prob((params['initial']['cov'], params['initial']['mean']))

        # dynamics
        dynamics_bias = params['dynamics']['bias'] if self.has_dynamics_bias else jnp.zeros((self.state_dim, 0))
        dynamics_matrix = jnp.column_stack((params['dynamics']['weights'],
                                            params['dynamics']['input_weights'],
                                            dynamics_bias))
        lp += self.dynamics_prior.log_prob((params['dynamics']['cov'], dynamics_matrix))

        emission_bias = params['emissions']['bias'] if self.has_emissions_bias else jnp.zeros((self.emission_dim, 0))
        emission_matrix = jnp.column_stack((params['emissions']['weights'],
                                            params['emissions']['input_weights'],
                                            emission_bias))
        lp += self.emission_prior.log_prob((params['emissions']['cov'], emission_matrix))
        return lp

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MAP estimation jointly
        initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
        S, m = initial_posterior.mode()

        dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
        Q, FB = dynamics_posterior.mode()
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

        emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
        R, HD = emission_posterior.mode()
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
            else (HD[:, self.state_dim:], jnp.zeros(self.emission_dim))

        params = dict(
            initial=dict(mean=m, cov=S),
            dynamics=dict(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=dict(weights=H, bias=d, input_weights=D, cov=R)
        )
        return params, m_step_state

    def fit_blocked_gibbs(self, key, sample_size, emissions, inputs=None):
        """Estimation using blocked-Gibbs sampler."""
        num_timesteps = len(emissions)

        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        def sufficient_stats_from_sample(states):
            """Convert samples of states to sufficient statistics."""
            inputs_joint = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
            # Let xn[t] = x[t+1]          for t = 0...T-2
            x, xp, xn = states, states[:-1], states[1:]
            u, up = inputs_joint, inputs_joint[:-1]
            y = emissions

            init_stats = (x[0], jnp.outer(x[0], x[0]), 1)

            # Quantities for the dynamics distribution
            # Let zp[t] = [x[t], u[t]] for t = 0...T-2
            sum_zpzpT = jnp.block([[xp.T @ xp, xp.T @ up], [up.T @ xp, up.T @ up]])
            sum_zpxnT = jnp.block([[xp.T @ xn], [up.T @ xn]])
            sum_xnxnT = xn.T @ xn
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
            if not self._db_indicator:
                dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                                  num_timesteps - 1)

            # Quantities for the emissions
            # Let z[t] = [x[t], u[t]] for t = 0...T-1
            sum_zzT = jnp.block([[x.T @ x, x.T @ u], [u.T @ x, u.T @ u]])
            sum_zyT = jnp.block([[x.T @ y], [u.T @ y]])
            sum_yyT = y.T @ y
            emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
            if not self._db_indicator:
                emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

            return init_stats, dynamics_stats, emission_stats

        def lgssm_params_sample(rng, stats):
            """Sample parameters of the model.
            """
            init_stats, dynamics_stats, emission_stats = stats
            rngs = iter(jr.split(rng, 3))

            # Sample the initial params
            initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
            S, m = initial_posterior.sample(seed=next(rngs))

            # Sample the dynamics params
            dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
            Q, FB = dynamics_posterior.sample(seed=next(rngs))
            F = FB[:, :self.state_dim]
            B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self._db_indicator \
                else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

            # Sample the emission params
            emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
            R, HD = emission_posterior.sample(seed=next(rngs))
            H = HD[:, :self.state_dim]
            D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self._eb_indicator \
                else (HD[:, self.state_dim:], jnp.zeros(self.emission_dim))

            return ParamsLGSSMMoment(initial_mean=m,
                               initial_covariance=S,
                               dynamics_matrix=F,
                               dynamics_input_weights=B,
                               dynamics_bias=b,
                               dynamics_covariance=Q,
                               emission_matrix=H,
                               emission_input_weights=D,
                               emission_bias=d,
                               emission_covariance=R)

        @jit
        def one_sample(_params, rng):
            rngs = jr.split(rng, 2)
            # Sample latent states
            self._make_inference_args = _params
            l_prior = self.log_prior()
            ll, states = lgssm_posterior_sample(rngs[0], self._make_inference_args, emissions, inputs)
            log_probs = l_prior + ll
            # Sample parameters
            _stats = sufficient_stats_from_sample(states)
            new_param = lgssm_params_sample(rngs[1], _stats)
            return new_param, log_probs

        log_probs = []
        sample_of_params = []
        keys = iter(jr.split(key, sample_size))
        current_params = self._make_inference_args
        for _ in progress_bar(sample_size):
            sample_of_params.append(current_params)
            current_params, loglik = one_sample(current_params, next(keys))
            log_probs.append(loglik)
        return log_probs, sample_of_params
