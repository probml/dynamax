"""
Demo of a Conditional Moment Generating Function (CMGF) estimator for 
online estimation of a logistic regression.
"""
import jax
from jax import numpy as jnp
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

from dynamax.generalized_gaussian_ssm import conditional_moments_gaussian_filter, EKFIntegrals, ParamsGGSSM

def fill_diagonal(A, elts):
    """
    Fill the diagonal of a matrix with elements from a vector.
    """
    # Taken from https://github.com/google/jax/issues/2680
    elts = jnp.ravel(elts)
    i, j = jnp.diag_indices(min(A.shape[-2:]))
    return A.at[..., i, j].set(elts)

class CMGFEstimator(BaseEstimator, ClassifierMixin):
    """
    Conditional Moment Generating Function (CMGF) estimator for online estimation of a logistic regression.
    """
    def __init__(self,  mean=None, cov=None):
        self.mean = mean
        self.cov = cov

    def fit(self, X, y):
        """
        Fit the model to the data in online fasion using CMGF.
        """
        X_bias = jnp.concatenate([jnp.ones((len(X), 1)), X], axis=1)
        # Encode output as one-hot-encoded vectors with first column dropped,
        # i.e., [0, ..., 0] correspondes to 1st class
        # This is done to prevent the "Dummy Variable Trap".
        enc = OneHotEncoder(drop='first')
        y_oh = jnp.array(enc.fit_transform(y.reshape(-1, 1)).toarray())
        input_dim = X_bias.shape[-1]
        num_classes = y_oh.shape[-1] + 1
        self.classes_ = jnp.arange(num_classes)
        weight_dim = input_dim * num_classes
        
        initial_mean, initial_covariance = jnp.zeros(weight_dim), jnp.eye(weight_dim)
        dynamics_function = lambda w, x: w
        dynamics_covariance = jnp.zeros((weight_dim, weight_dim))
        emission_mean_function = lambda w, x: jax.nn.softmax(x @ w.reshape(input_dim, -1))[1:]
        
        def emission_var_function(w, x):
            """Compute the variance of the emission distribution."""
            ps = jnp.atleast_2d(emission_mean_function(w, x))
            return fill_diagonal(ps.T @ -ps, ps * (1-ps))
        
        cmgf_params = ParamsGGSSM(
            initial_mean = initial_mean,
            initial_covariance = initial_covariance,
            dynamics_function = dynamics_function,
            dynamics_covariance = dynamics_covariance,
            emission_mean_function = emission_mean_function,
            emission_cov_function = emission_var_function
        )
        post = conditional_moments_gaussian_filter(cmgf_params, EKFIntegrals(), y_oh, inputs = X_bias)
        post_means, post_covs = post.filtered_means, post.filtered_covariances
        self.mean, self.cov = post_means[-1], post_covs[-1]
        return self
    
    def predict(self, X):
        """Predict the outputs for a new input"""
        return jnp.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Predict the class probabilities for a new input"""
        X = jnp.array(X)
        X_bias = jnp.concatenate([jnp.ones((len(X), 1)), X], axis=1)
        return jax.nn.softmax(X_bias @ self.mean.reshape(X_bias.shape[-1], -1))
