from functools import partial

import jax.random as jr
from jax import jit
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class, tree_map
from ssm_jax.distributions import MatrixNormalInverseWishart as MNIW
from ssm_jax.distributions import NormalInverseWishart as NIW
from ssm_jax.distributions import mniw_posterior_update, niw_posterior_update
from ssm_jax.linear_gaussian_ssm.inference import lgssm_posterior_sample, LGSSMParams
from ssm_jax.linear_gaussian_ssm.models.linear_gaussian_ssm import LinearGaussianSSM
from tqdm.auto import trange


@register_pytree_node_class
class LinearGaussianConjugateSSM(LinearGaussianSSM):
    """
    Linear Gaussian State Space Model with conjugate priors for model parameters,
    p(m, S) = NIW(loc, mean_concentration, df, scale)
    p([F, B, b], Q) = MNIW(loc, col_precision, df, scale)
    p([H, D, d], R) = MNIW(loc, col_precision, df, scale)
    where:

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
        super().__init__(dynamics_matrix, dynamics_covariance, emission_matrix, emission_covariance,
                         initial_mean, initial_covariance, dynamics_input_weights, dynamics_bias,
                         emission_input_weights, emission_bias)

        # Initialize prior distributions
        def default_prior(arg, default):
            return kw_priors[arg] if arg in kw_priors else default

        self.initial_prior = default_prior(
            'initial_prior',
            NIW(loc=self.initial_mean.value,
                mean_concentration=1.,
                df=self.state_dim + 0.1,
                scale=jnp.eye(self.state_dim)))

        loc_d = self._join_matrix(self.dynamics_matrix.value, self.dynamics_input_weights.value,
                                  self.dynamics_bias.value, self._db_indicator)
        self.dynamics_prior = default_prior(
            'dynamics_prior',
            MNIW(loc=loc_d,
                 col_precision=jnp.eye(loc_d.shape[1]),
                 df=self.state_dim + 0.1,
                 scale=jnp.eye(self.state_dim)))

        loc_e = self._join_matrix(self.emission_matrix.value, self.emission_input_weights.value,
                                  self.emission_bias.value, self._eb_indicator)
        self.emission_prior = default_prior(
            'emission_prior',
            MNIW(loc=loc_e,
                 col_precision=jnp.eye(loc_e.shape[1]),
                 df=self.emission_dim + 0.1,
                 scale=jnp.eye(self.emission_dim)))

    def log_prior(self):
        """Return the log prior probability of any model parameters.
        Returns:
            lp (Scalar): log prior probability.
        """
        d_matrix = self._join_matrix(self.dynamics_matrix.value, self.dynamics_input_weights.value,
                                     self.dynamics_bias.value, self._db_indicator)
        e_matrix = self._join_matrix(self.emission_matrix.value, self.emission_input_weights.value,
                                     self.emission_bias.value, self._eb_indicator)
        # Compute log probs
        lp = self.initial_prior.log_prob((self.initial_covariance.value, self.initial_mean.value))
        lp += self.dynamics_prior.log_prob((self.dynamics_covariance.value, d_matrix))
        lp += self.emission_prior.log_prob((self.emission_covariance.value, e_matrix))
        return lp

    def map_step(self, batch_stats):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MAP estimation jointly
        initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
        S, m = initial_posterior.mode()

        dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
        Q, FB = dynamics_posterior.mode()
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self._db_indicator \
            else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

        emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
        R, HD = emission_posterior.mode()
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

    def fit_em(self, batch_emissions, batch_inputs=None, num_iters=50, method='MLE'):
        """Fit this HMM with Expectation-Maximization (EM)."""
        if method == 'MLE':
            return super().fit_em(batch_emissions, batch_inputs, num_iters)
        elif method == 'MAP':

            @jit
            def emap_step(_params):
                self.params = _params
                posterior_stats, marginal_loglikes = self.e_step(batch_emissions, batch_inputs)
                new_param = self.map_step(posterior_stats)
                return new_param, marginal_loglikes.sum()

            log_probs = []
            _params = self.params
            for _ in trange(num_iters):
                _params, marginal_loglik = emap_step(_params)
                log_probs.append(marginal_loglik)

            self.params = _params
            return jnp.array(log_probs)

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

        @jit
        def one_sample(_params, rng):
            rngs = jr.split(rng, 2)
            # Sample latent states
            self.params = _params
            l_prior = self.log_prior()
            ll, states = lgssm_posterior_sample(rngs[0], self.params, emissions, inputs)
            log_probs = l_prior + ll
            # Sample parameters
            _stats = sufficient_stats_from_sample(states)
            new_param = lgssm_params_sample(rngs[1], _stats)
            return new_param, log_probs

        log_probs = []
        sample_of_params = []
        keys = iter(jr.split(key, sample_size))
        current_params = self.params
        for _ in trange(sample_size):
            sample_of_params.append(current_params)
            current_params, loglik = one_sample(current_params, next(keys))
            log_probs.append(loglik)
        return log_probs, sample_of_params

    def _join_matrix(self, F, B, b, bias_indicator):
        """Join the transition matrix and weight matrix so that they can be inferred jointly.

        If bias_indicator is True,
        the bias vector b is added to the last column of the [F, B] matrix,
        otherwise,
        there is NO bias term, and [F, B] are jointly estiamted.
        """
        if not bias_indicator:
            return jnp.concatenate((F, B), axis=1)
        else:
            return jnp.concatenate((F, B, b[:, None]), axis=1)
