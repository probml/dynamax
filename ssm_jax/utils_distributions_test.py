import jax.numpy as jnp
import jax.random as jr

import os
os.chdir('/home/xinglong/git_local/ssm-jax/ssm_jax')
from utils_distributions import NormalInverseWishart as NIW, InverseWishart as IW, MatrixNormal as MN, MatrixNormalInverseWishart as MNIW

key = jr.PRNGKey(0)


# Test the NormalInverseWishart distribution
niw_params = (jnp.zeros(3), 1, 3, jnp.eye(3))
niw = NIW(*niw_params)
niw.mode
niw_samples = niw.sample(seed=key, sample_shape=(2,))
niw_sample_probs = niw.prob(niw_samples)
niw_sample_log_probs = niw.log_prob(niw_samples)

def manual_niw_log_prob(mu, Sigma, loc, mean_precision, df, scale):
    """
    Evaluate the NIW log prob using scipy.stats functions
    """
    from scipy.stats import invwishart
    from scipy.stats import multivariate_normal as mvn

    lp_iw = invwishart.logpdf(jnp.transpose(Sigma, (1,2,0)), df, scale)
    lp_n = jnp.array([mvn.logpdf(x, loc, sigma/mean_precision) for x, sigma in zip(mu, Sigma)])
    return lp_iw + lp_n

assert jnp.allclose(niw.log_prob(niw_samples), 
                    manual_niw_log_prob(niw_samples['mu'], niw_samples['Sigma'], *niw_params))


# Test the InverseWishart distribution 
iw_params = (3, jnp.eye(3))
iw = IW(*iw_params)
iw.mode
iw_samples = iw.sample(seed=key, sample_shape=2)
iw_sample_probs = iw.prob(iw_samples)
iw_sample_log_probs = iw.log_prob(iw_samples)

def manual_iw_log_prob(Sigma, df, scale):
    """
    Evaluate the IW log prob using scipy.stats functions
    """
    from scipy.stats import invwishart
    
    lp = invwishart.logpdf(jnp.transpose(Sigma, (1,2,0)), df, scale)
    return lp

assert jnp.allclose(iw.log_prob(iw_samples), manual_iw_log_prob(iw_samples, *iw_params))


# Test the MatrixNormal distribution
mn_params = (jnp.ones((2, 3)), jnp.eye(2), jnp.eye(3))
mn = MN(*mn_params)
mn.mode
mn_samples = mn.sample(seed=key, sample_shape=(2,))
mn_probs = mn.prob(mn_samples)
mn_log_probs = mn.log_prob(mn_samples)

def manual_mn_log_prob(X, loc, row_cov, col_precision):
    """
    Evaluate the MN log prob using scipy.stats functions
    """
    from scipy.stats import matrix_normal
    
    lp = matrix_normal.logpdf(X, mean=loc, rowcov=row_cov, colcov=jnp.linalg.inv(col_precision))
    return lp

assert jnp.allclose(mn.log_prob(mn_samples), manual_mn_log_prob(mn_samples, *mn_params))


# Test the MatrixNormalInverseWishart distribution
mniw_params = (jnp.ones((2, 3)), jnp.eye(3), 3, jnp.eye(2))
mniw = MNIW(*mniw_params)
mniw.mode
mniw_samples = mniw.sample(seed=key, sample_shape=2)
mniw_probs = mniw.prob(mniw_samples)
mniw_log_probs = mniw.log_prob(mniw_samples)

def manual_mniw_log_prob(Matrix, Sigma, loc, col_precision, df, scale):
    """
    Evaluate the MNIW log prob using scipy.stats functions
    """
    from scipy.stats import invwishart, matrix_normal

    lp_iw = invwishart.logpdf(jnp.transpose(Sigma, (1,2,0)), df, scale)
    lp_mn = jnp.array([matrix_normal.logpdf(m, loc, sigma, jnp.linalg.inv(col_precision)) for m, sigma in zip(Matrix, Sigma)])
    return lp_iw + lp_mn

assert jnp.allclose(mniw.log_prob(mniw_samples), 
                    manual_mniw_log_prob(mniw_samples['Matrix'], mniw_samples['Sigma'], *mniw_params))