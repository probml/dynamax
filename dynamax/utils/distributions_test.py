import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import tree_map
from jax.scipy.stats import norm
from scipy.stats import invgamma
from tensorflow_probability.substrates import jax as tfp

from dynamax.utils.distributions import InverseWishart
from dynamax.utils.distributions import MatrixNormalInverseWishart
from dynamax.utils.distributions import MatrixNormalPrecision
from dynamax.utils.distributions import NormalInverseGamma
from dynamax.utils.distributions import NormalInverseWishart

tfd = tfp.distributions
tfb = tfp.bijectors


def test_inverse_wishart_mode(df=7.0, dim=3, scale_factor=3.0):
    scale = scale_factor * jnp.eye(dim)
    iw = InverseWishart(df, scale)
    assert jnp.allclose(iw.mode(), scale / (df + dim + 1))


def test_inverse_wishart_log_prob(df=7.0, dim=3, scale_factor=3.0, n_samples=10):
    scale = scale_factor * jnp.eye(dim)
    iw = InverseWishart(df, scale)
    samples = iw.sample(seed=jr.PRNGKey(0), sample_shape=(n_samples,))
    assert samples.shape == (n_samples, dim, dim)
    lps = iw.log_prob(samples)
    assert lps.shape == (n_samples,)
    assert jnp.all(jnp.isfinite(lps))


def test_inverse_wishart_sample(df=7.0, dim=3, scale_factor=3.0, n_samples=10000, num_std=6):
    """Test that the sample mean is within a (large) interval around the true mean.
    To determine the interval to 6 times the standard deviation of the Monte
    Carlo estimator.

    Note: This also tests the variance implementation indirectly. If there's a bug in
    variance it will affect the MC error.
    """
    scale = scale_factor * jnp.eye(dim)
    iw = InverseWishart(df, scale)
    samples = iw.sample(seed=jr.PRNGKey(0), sample_shape=(n_samples,))
    assert samples.shape == (n_samples, dim, dim)

    mc_std = jnp.sqrt(iw.variance() / n_samples)
    assert jnp.allclose(samples.mean(axis=0), iw.mean(), atol=num_std * mc_std)


def test_normal_inverse_wishart_mode(loc=0., mean_conc=1.0, df=7.0, dim=3, scale_factor=3.0):
    loc = loc * jnp.ones(dim)
    scale = scale_factor * jnp.eye(dim)
    niw = NormalInverseWishart(loc, mean_conc, df, scale)
    Sigma, mu = niw.mode()
    assert jnp.allclose(mu, loc)
    assert jnp.allclose(Sigma, scale / (df + dim + 2))


def test_normal_inverse_wishart_mode_batch(loc=0., mean_conc=1.0, df=7.0, dim=3, scale_factor=3.0, batch_size=10):
    loc = loc * jnp.ones(dim)
    scale = scale_factor * jnp.eye(dim)
    niw = NormalInverseWishart(loc[None, ...].repeat(batch_size, axis=0), mean_conc, df, scale[None,
                                                                                               ...].repeat(batch_size,
                                                                                                           axis=0))
    Sigma, mu = niw.mode()
    assert Sigma.shape == (batch_size, dim, dim)
    assert mu.shape == (batch_size, dim)
    assert jnp.allclose(mu, loc)
    assert jnp.allclose(Sigma, scale / (df + dim + 2))


def test_normal_inverse_wishart_log_prob(loc=0., mean_conc=1.0, df=7.0, dim=3, scale_factor=3.0, n_samples=10):
    loc = loc * jnp.ones(dim)
    scale = scale_factor * jnp.eye(dim)
    niw = NormalInverseWishart(loc, mean_conc, df, scale)
    Sigma_samples, mu_samples = niw.sample(seed=jr.PRNGKey(0), sample_shape=(n_samples,))
    assert mu_samples.shape == (n_samples, dim)
    assert Sigma_samples.shape == (n_samples, dim, dim)
    lps = niw.log_prob((Sigma_samples, mu_samples))
    assert lps.shape == (n_samples,)
    assert jnp.all(jnp.isfinite(lps))


def test_matrix_normal_log_prob(loc=jnp.ones((2, 3)), row_cov=jnp.eye(2), col_precision=jnp.eye(3), n_samples=2):
    """
    Evaluate the MN log prob using scipy.stats functions
    """
    from scipy.stats import matrix_normal

    mn = MatrixNormalPrecision(loc, row_cov, col_precision)
    mn_samples = mn.sample(seed=jr.PRNGKey(0), sample_shape=n_samples)
    mn_probs = mn.prob(mn_samples)
    mn_log_probs = mn.log_prob(mn_samples)
    lps = matrix_normal.logpdf(mn_samples, mean=loc, rowcov=row_cov, colcov=jnp.linalg.inv(col_precision))
    assert jnp.allclose(jnp.array(mn_log_probs), lps)
    assert jnp.allclose(jnp.array(mn_probs), jnp.exp(lps))


def test_matrix_normal_inverse_wishart_log_prob(
        loc=jnp.ones((2, 3)), col_precision=jnp.eye(3), df=3, scale=jnp.eye(2), n_samples=2):
    """
    Evaluate the MNIW log prob using scipy.stats functions
    """
    from scipy.stats import invwishart
    from scipy.stats import matrix_normal

    mniw = MatrixNormalInverseWishart(loc, col_precision, df, scale)
    Sigma_samples, Matrix_samples = mniw.sample(seed=jr.PRNGKey(0), sample_shape=n_samples)
    mniw_probs = mniw.prob((Sigma_samples, Matrix_samples))
    mniw_log_probs = mniw.log_prob((Sigma_samples, Matrix_samples))

    lp_iw = invwishart.logpdf(jnp.transpose(Sigma_samples, (1, 2, 0)), df, scale)
    lp_mn = jnp.array([matrix_normal.logpdf(m, loc, sigma, jnp.linalg.inv(col_precision)) \
                       for m, sigma in zip(Matrix_samples, Sigma_samples)])

    assert jnp.allclose(mniw_log_probs, lp_iw + lp_mn)
    assert jnp.allclose(mniw_probs, jnp.exp(lp_iw + lp_mn))


def test_normal_inverse_gamma_vs_normal_inv_wishart(
        key=jr.PRNGKey(0), loc=0.0, mean_conc=1.0, concentration=7.0, scale=3.0):

    nig = NormalInverseGamma(loc, mean_conc, concentration, scale)

    loc = loc * jnp.ones((1, 1))
    scale = scale * jnp.ones((1, 1))
    niw = NormalInverseWishart(loc, mean_conc, 2 * concentration, 2 * scale)

    niw_sigma_samples, niw_mu_samples = niw.sample(seed=key, sample_shape=(1,))

    nig_log_probs = nig.log_prob((jnp.squeeze(niw_sigma_samples, axis=-1), niw_mu_samples))
    nig_probs = nig.prob((jnp.squeeze(niw_sigma_samples, axis=-1), niw_mu_samples))

    niw_log_probs = niw.log_prob((niw_sigma_samples, niw_mu_samples))
    niw_probs = niw.prob((niw_sigma_samples, niw_mu_samples))

    assert jnp.allclose(nig_log_probs, niw_log_probs, atol=1e-3)
    assert jnp.allclose(nig_probs, niw_probs, atol=1e-3)
    assert all(tree_map(lambda x, y: jnp.allclose(jnp.array(x), jnp.array(y)), niw.mode(), nig.mode()))


def test_normal_inverse_gamma_log_prob(
        key=jr.PRNGKey(0), loc=0.0, mean_conc=1.0, concentration=7.0, scale=3.0, n_samples=10):

    nig = NormalInverseGamma(loc, mean_conc, concentration, scale)

    variance, mean = nig.sample(seed=key, sample_shape=(n_samples,))

    log_prob = tfd.Normal(loc, jnp.sqrt(variance / mean_conc)).log_prob(mean)
    log_prob += tfd.InverseGamma(concentration, scale).log_prob(variance)

    scipy_log_prob = norm.logpdf(mean, loc, jnp.sqrt(variance / mean_conc))
    scipy_log_prob += invgamma.logpdf(jnp.array(variance), concentration, scale=scale)

    nig_log_prob = nig.log_prob((variance, mean))

    assert jnp.allclose(nig_log_prob, log_prob)
    assert jnp.allclose(nig_log_prob, scipy_log_prob, atol=1e-3)
