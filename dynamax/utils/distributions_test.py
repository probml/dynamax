"""
Tests for the distributions module.
"""
import jax.numpy as jnp
import jax.random as jr

from jax.tree_util import tree_map
from jax.scipy.stats import norm
import pytest
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
    """
    Test that the mode of the InverseWishart distribution is correct.
    """
    scale = scale_factor * jnp.eye(dim)
    iw = InverseWishart(df, scale)
    assert jnp.allclose(iw.mode(), scale / (df + dim + 1))


def test_inverse_wishart_log_prob(df=7.0, dim=3, scale_factor=3.0, n_samples=10):
    """
    Test that the log_prob of samples from the InverseWishart distribution is finite.
    """
    scale = scale_factor * jnp.eye(dim)
    iw = InverseWishart(df, scale)
    samples = iw.sample(seed=jr.PRNGKey(0), sample_shape=(n_samples,))
    assert samples.shape == (n_samples, dim, dim)
    lps = iw.log_prob(samples)
    assert lps.shape == (n_samples,)
    assert jnp.all(jnp.isfinite(lps))


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_inverse_wishart_sample(dim, df=7.0, scale_factor=3.0, n_samples=10000, num_std=6):
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

def test_inverse_wishart_variance_vectorization():
    """Test the vectorization of marginal variance calculation."""
    p = 3
    # Vectorize over two inverse-Wishart distributions.
    Ψ1 = jnp.array([[ 3.915627  ,  0.05046278, -0.7466818 ],
       [ 0.05046278,  5.368932  , -0.57225686],
       [-0.7466818 , -0.57225686,  3.162733  ]], dtype=jnp.float32)
    𝜈1 = 7.5  # >p + 3
    Ψ2 = jnp.array([[ 4.3864717, -0.8869951, -0.9199044],
       [-0.8869951,  5.07704  , -0.8494578],
       [-0.9199044, -0.8494578,  4.1288185]], dtype=jnp.float32)
    𝜈2 = 8.0  # >p + 3
    assert all(jnp.linalg.eigvals(Ψ1) > 0)
    assert all(jnp.linalg.eigvals(Ψ2) > 0)

    # Make a (2, 1) batch shape to test vectorization over >1 leading axis.
    𝜈 = jnp.stack([𝜈1, 𝜈2]) # Shape: (2,)
    𝜈 = 𝜈[:, None]  # Shape: (2, 1)
    Ψ = jnp.stack([Ψ1, Ψ2])  # Shape: (2, 3, 3)
    Ψ = Ψ[:, None]  # Shape: (2, 1, 3, 3)
    variance_composite = InverseWishart(df=𝜈, scale=Ψ).variance()
    assert variance_composite.shape[-2:] == (p, p)

    # Verify that vectorized and non-vectorized calculation gives the same variance.
    variance1 = InverseWishart(df=𝜈1, scale=Ψ1).variance()
    variance2 = InverseWishart(df=𝜈2, scale=Ψ2).variance()
    jnp.allclose(variance_composite[0, 0], variance1)
    jnp.allclose(variance_composite[1, 0], variance2)

def test_inverse_wishart_sample_non_diagonal_scale(n_samples: int = 10_000, num_std=3):
    """Test sample mean of an inverse-Wishart distr. w/ non-diagonal scale matrix."""
    k = 2
    𝜈 = 5.5  # 𝜈 > k
    Ψ = jnp.array([[20.712932, 25.124634],
        [25.124634, 32.814785]], dtype=jnp.float32)  # k x k
    Ψ_diag = jnp.diagonal(Ψ)
    assert all(jnp.linalg.eigvals(Ψ) > 0)  # Is positive definite.

    iw = InverseWishart(df=𝜈, scale=Ψ)
    Σs = iw.sample(sample_shape=n_samples, seed=jr.key(42))
    actual_Σ_avg = jnp.mean(Σs, axis=0)

    # Closed form expression of mean.
    true_Σ_avg = Ψ / (𝜈 - k - 1)
    # Closed form expression of variance.
    numerator = (𝜈 - k + 1) * Ψ**2 + (𝜈 - k - 1) * jnp.outer(Ψ_diag, Ψ_diag)
    denominator = (𝜈 - k) * (𝜈 - k - 1)**2 * (𝜈 - k - 3)
    true_Σ_var = numerator / denominator

    mc_std = jnp.sqrt(true_Σ_var / n_samples)
    assert jnp.allclose(actual_Σ_avg, true_Σ_avg, atol=num_std * mc_std)


def test_normal_inverse_wishart_mode(loc=0., mean_conc=1.0, df=7.0, dim=3, scale_factor=3.0):
    """
    Test that the mode of the NormalInverseWishart distribution is correct.
    """
    loc = loc * jnp.ones(dim)
    scale = scale_factor * jnp.eye(dim)
    niw = NormalInverseWishart(loc, mean_conc, df, scale)
    Sigma, mu = niw.mode()
    assert jnp.allclose(mu, loc)
    assert jnp.allclose(Sigma, scale / (df + dim + 2))


def test_normal_inverse_wishart_mode_batch(loc=0., mean_conc=1.0, df=7.0, dim=3, scale_factor=3.0, batch_size=10):
    """
    Test that the mode of the NormalInverseWishart distribution is correct.
    """
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
    """
    Test that the log_prob of samples from the NormalInverseWishart distribution is finite.
    """
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


def test_normal_inverse_gamma_vs_normal_inv_wishart(key=jr.PRNGKey(0), loc=0.0, mean_conc=1.0, concentration=7.0, scale=3.0):
    """
    Test that the NormalInverseGamma and NormalInverseWishart distributions are equivalent.
    """
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


def test_normal_inverse_gamma_log_prob(key=jr.PRNGKey(0), loc=0.0, mean_conc=1.0, concentration=7.0, scale=3.0, n_samples=10):
    """
    Test that the log_prob of samples from the NormalInverseGamma distribution is finite.
    """
    nig = NormalInverseGamma(loc, mean_conc, concentration, scale)

    variance, mean = nig.sample(seed=key, sample_shape=(n_samples,))

    log_prob = tfd.Normal(loc, jnp.sqrt(variance / mean_conc)).log_prob(mean)
    log_prob += tfd.InverseGamma(concentration, scale).log_prob(variance)

    scipy_log_prob = norm.logpdf(mean, loc, jnp.sqrt(variance / mean_conc))
    scipy_log_prob += invgamma.logpdf(jnp.array(variance), concentration, scale=scale)

    nig_log_prob = nig.log_prob((variance, mean))

    assert jnp.allclose(nig_log_prob, log_prob)
    assert jnp.allclose(nig_log_prob, scipy_log_prob, atol=1e-3)
