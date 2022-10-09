from jax import value_and_grad
import jax.numpy as jnp
import jax.ranodm as jr
from jax.scipy.optimize import minimize


def fit_advi(self, key, emissions, inputs, M):
    """ADVI with fixed sample and optimized with second order method
    Mean field VI
    """
    standard_normal = MVN()
    samples = standard_normal(key, sample_shape=M)
    
    def unnorm_log_pos(trainable_unc_params):
        params = from_unconstrained(trainable_unc_params, fixed_params, self.param_props)
        log_det_jac = log_det_jac_constrain(trainable_unc_params, fixed_params, self.param_props)
        log_pri = self.log_prior(params) + log_det_jac
        batch_lls = self.marginal_log_prob(emissions, inputs, params)
        lp = log_pri + batch_lls.sum()
        return lp
    
    def log_prob_q():
        return 
        
    def elbo(z, mu, w):
        unc_params = [mu, w] + z
        return unnorm_log_pos(unc_params) + log_prob_q(unc_params)
    
    loss_fn = vmap(elbo)(samples).sum()
    
    result = minimize(with_grad, start_params, method=method_name, jac=True)
    
    
