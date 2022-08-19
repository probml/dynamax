from functools import partial

from jax import numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
from jax.tree_util import register_pytree_node_class

from ssm_jax.linear_gaussian_ssm.models import LinearGaussianSSM
from ssm_jax.linear_gaussian_ssm.inference import lgssm_smoother, LGSSMParams



@register_pytree_node_class
class LinearGaussianSSMMLE(LinearGaussianSSM):
    """
    Linear Gaussian State Space Model with EM learning algorithm.
    """

    def __init__(
        self,
        dynamics_matrix,
        dynamics_covariance,
        emission_matrix,
        emission_covariance,
        initial_mean=None,
        initial_covariance=None,
        dynamics_input_weights=None,
        dynamics_bias_indicator=False,
        emission_input_weights=None,
        emission_bias_indicator=False
    ):
        super().__init__(dynamics_matrix, dynamics_covariance, 
                         emission_matrix, emission_covariance,
                         initial_mean,    initial_covariance,
                         dynamics_input_weights, dynamics_bias_indicator, 
                         emission_input_weights, emission_bias_indicator)
        
    ### Expectation-maximization (EM) code
    def e_step(self, batch_emissions, batch_inputs=None):
        """The E-step computes sums of expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        num_batches, num_timesteps = batch_emissions.shape[:2]
        if batch_inputs is None:
            batch_inputs = jnp.zeros((num_batches, num_timesteps, 0))

        def _single_e_step(emissions, inputs):
            # Run the smoother to get posterior expectations
            posterior = lgssm_smoother(LGSSMParams(), emissions, inputs)

            # shorthand
            Ex = posterior.smoothed_means
            Exp = posterior.smoothed_means[:-1]
            Exn = posterior.smoothed_means[1:]
            Vx = posterior.smoothed_covariances
            Vxp = posterior.smoothed_covariances[:-1]
            Vxn = posterior.smoothed_covariances[1:]
            Expxn = posterior.smoothed_cross_covariances
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
            sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up],
                                   [ up.T @ Exp,  up.T @ up]])
            sum_zpzpT = sum_zpzpT.at[: self.state_dim, : self.state_dim].add(Vxp.sum(0))
            sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
            sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)

            # more expected sufficient statistics for the emissions
            # let z[t] = [x[t], u[t]] for t = 0...T-1
            sum_zzT = jnp.block([[Ex.T @ Ex,  Ex.T @ u],
                                 [ u.T @ Ex,   u.T @ u]])
            sum_zzT = sum_zzT.at[: self.state_dim, : self.state_dim].add(Vx.sum(0))
            sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
            sum_yyT = emissions.T @ emissions
            emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)

            return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

        # TODO: what's the best way to vectorize/parallelize this?
        return vmap(_single_e_step)(batch_emissions, batch_inputs)
    
    @classmethod
    def m_step(cls, batch_stats, dynamics_bias_indicator, emission_bias_indicator):
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = jnp.linalg.solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # initial distribution
        sum_x0, sum_x0x0T, N = init_stats
        dim = sum_x0.shape[0]
        S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
        m = sum_x0 / N

        # dynamics distribution
        FB, Q = fit_linear_regression(*dynamics_stats)
        F = FB[:, :dim]
        B = FB[:, dim:-1] if dynamics_bias_indicator else FB[:,dim:] 

        # emission distribution
        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :dim]
        D = HD[:, dim:-1] if emission_bias_indicator else HD[:,dim:]
        
        return cls(initial_mean = m,
                   initial_covariance = S,
                   dynamics_matrix = F,
                   dynamics_input_weights = B,
                   dynamics_covariance = Q,
                   emission_matrix = H,
                   emission_input_weights = D,
                   emission_covariance = R,
                   dynamics_bias_indicator=dynamics_bias_indicator,
                   emission_bias_indicator=emission_bias_indicator)