import time
from typing import Sequence
from functools import partial

from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves
from jax import jit, lax
from jax.scipy.linalg import block_diag
from ssm_jax.cond_moments_gaussian_filter.containers import *
from ssm_jax.cond_moments_gaussian_filter.inference import *


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class CMGFBinaryMLPEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, params_, mlp_model_dims_, decouple_=False):
        self.params_ = params_
        self.mlp_model_dims_ = mlp_model_dims_
        self.classes_ = jnp.arange(2)
        self.decouple_ = decouple_
        _, self.mean_, _, self.apply_fn_ = get_mlp_flattened_params(self.mlp_model_dims_)
        eps = 1e-4
        self.sigmoid_fn_ = lambda w, x: jnp.clip(jax.nn.sigmoid(self.apply_fn_(w, x)), eps, 1-eps) # Clip to prevent divergence
        self.cov_ = jnp.eye(self.mean_.size)
    
    def fit(self, X, y):
        X, y = jnp.array(X), jnp.array(y)
        state_dim = self.mean_.size
        
        cmgf_params = self.params_(
            initial_mean = self.mean_,
            initial_covariance = self.cov_,
            dynamics_function = lambda w, _: w,
            dynamics_covariance = jnp.zeros((state_dim, state_dim,)),
            emission_mean_function = lambda w, x: self.sigmoid_fn_(w, x),
            emission_cov_function = lambda w, x: self.sigmoid_fn_(w, x) * (1 - self.sigmoid_fn_(w, x))
        )
        decouple_params = generate_decouple_params(self.mlp_model_dims_)

        # Run CMGF to train MLP
        if self.decouple_:
            post = decoupled_extended_conditional_moments_gaussian_filter(cmgf_params, decouple_params, y, inputs=X)
        else:
            post = conditional_moments_gaussian_filter(cmgf_params, y, inputs = X)

        post_means, post_covs = post.filtered_means, post.filtered_covariances
        self.mean_, self.cov_ = post_means[-1], post_covs[-1]
        return self
    
    def predict(self, X):
        return jnp.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        prob_one = self.sigmoid_fn_(self.mean_, jnp.array(X))
        return jnp.array([1-prob_one, prob_one]).T[0]


def get_mlp_flattened_params(model_dims, key=0):
    """_summary_

    Args:
        model_dims (_type_): _description_
        key (int, optional): _description_. Defaults to 0.

    Returns:
        model: _description_
        flat_params:
        unflatten_fn:
        apply_fn:
    """    
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    # Define MLP model
    input_dim, features = model_dims[0], model_dims[1:]
    model = MLP(features)
    dummy_input = jnp.ones((input_dim,))

    # Initialize parameters using dummy input
    params = model.init(key, dummy_input)
    flat_params, unflatten_fn = ravel_pytree(params)

    # Define apply function
    def apply(flat_params, x, model, unflatten_fn):
        return model.apply(unflatten_fn(flat_params), jnp.atleast_1d(x))

    apply_fn = partial(apply, model=model, unflatten_fn=unflatten_fn)

    return model, flat_params, unflatten_fn, apply_fn


def decouple_flat_params(model_dims):
    """_summary_

    Args:
        model_dims (_type_): _description_

    Returns:
        decouple_fn:
        recouple_fn:
        decouple_cov_fn:
        recouple_cov_fn: 
        decouple_jac_fn: 
    """    
    assert len(model_dims) > 1
    decoupled_params_idx = []
    curr_idx = 0
    for layer in range(1, len(model_dims)):
        # Number of parameter elements corresponding to current layer
        num_prev, num_curr = model_dims[layer-1], model_dims[layer] # Number of nodes in prev, curr layer
        num_bias_params = num_curr
        num_weight_params = num_prev * num_curr
        num_params_curr_layer = num_bias_params + num_weight_params
        
        # Range of indices in flattened params array corresponding to current layer
        idx_range = jnp.arange(curr_idx, curr_idx + num_params_curr_layer)
        
        # Append list of indices for each node in current layer
        decoupled_params_idx += [jnp.array([idx_range[i + num_curr * j] for j in range(num_prev + 1)]) for i in range(num_curr)]
        
        curr_idx += num_params_curr_layer

    # Function to decouple parameters by node
    decouple_fn = jit(lambda params: {i: params[idx] for i, idx in enumerate(decoupled_params_idx)})

    # Function to decouple Jacobians by node
    decouple_jac_fn = jit(lambda jac: {i: jac[:,idx] for i, idx in enumerate(decoupled_params_idx)})

    # Set to tuple to avoid non-hashable type error when using lists/DeviceArrays
    params_sizes = tuple([0] + [len(node_params) for node_params in decoupled_params_idx])
    diag_idx = jnp.cumsum(jnp.array(params_sizes))

    # Function to separate parameter covariance matrix by node
    @partial(jit, static_argnums=(2,))
    def decouple_cov_fn(cov, diag_idx, params_sizes):
        return {i: lax.dynamic_slice(cov, (diag_idx[i], diag_idx[i]), (params_sizes[i+1], params_sizes[i+1])) for i in range(0, len(diag_idx)-1)}
    decouple_cov_fn = partial(decouple_cov_fn, diag_idx=diag_idx, params_sizes=params_sizes)
    recouple_cov_fn = jit(lambda covs: block_diag(*tree_leaves(covs)))
    
    # Function to recouple decoupled params list
    @jit
    def recouple_fn(decoupled_params):
        decoupled_params_list = tree_leaves(decoupled_params)
        recoupled_params_list = []
        curr_idx = 0
        for layer in range(1, len(model_dims)):
            # Flatten params sublist corresponding to each layer
            recoupled_params_list.append(jnp.ravel(jnp.array(decoupled_params_list[curr_idx:curr_idx + model_dims[layer]]), order='F'))
            curr_idx += model_dims[layer]
            
        return jnp.concatenate(recoupled_params_list)
    
    return decouple_fn, recouple_fn, decouple_cov_fn, recouple_cov_fn, decouple_jac_fn


def generate_decouple_params(model_dims):
    """_summary_

    Args:
        model_dims (_type_): _description_

    Returns:
        decouple_params: _description_
    """    
    decouple_params = DecoupleParams(
        decouple_fn = decouple_flat_params(model_dims)[0],
        recouple_fn = decouple_flat_params(model_dims)[1],
        decouple_cov_fn = decouple_flat_params(model_dims)[2],
        recouple_cov_fn = decouple_flat_params(model_dims)[3],
        decouple_jac_fn = decouple_flat_params(model_dims)[4]
    )
    return decouple_params


def compute_cv_avg_score(estimator, num_points, num_features, n_splits=5, scoring='neg_log_loss'):
    # Generate dataset
    X, y = make_classification(n_samples=num_points, n_features=num_features, 
                               n_informative=num_features, n_redundant=0, random_state=0)

    # Setup pipeline
    scaler = preprocessing.StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', estimator)])

    # K-fold cross validation accuracy
    cv = KFold(n_splits=n_splits)
    scores = cross_val_score(pipeline, X, y.astype('float64'), cv=cv, scoring=scoring)

    return scores.mean()


def compare_performance(estimators, input_dim, data_size_grid):
    scores, times = {est: [] for est in estimators.keys()}, {est: [] for est in estimators.keys()}
    for num_points in data_size_grid:
        print(f'{num_points} data points.')
        for est_name, estimator in estimators.items():
            # Reinitialize estimator
            estimator = clone(estimator)

            # Measure running time to train
            start = time.time()
            log_likelihood = compute_cv_avg_score(estimator, num_points, input_dim)
            elapsed_time = time.time() - start
            
            print(f'{est_name} average per-sample loglikelihood = {log_likelihood}')
            print(f'{est_name} took {elapsed_time} seconds to train.\n')

            # Store results
            scores[est_name].append(log_likelihood)
            times[est_name].append(elapsed_time)
    
    return scores, times


if __name__ == "__main__":
    # Define MLP architecture
    input_dim, hidden_dims, output_dim = 2, [3, 3, 3, 3, 3, 3], 1
    model_dims = [input_dim, *hidden_dims, output_dim]

    cmgf_est = CMGFBinaryMLPEstimator(EKFParams, model_dims)
    d_cmgf_est = CMGFBinaryMLPEstimator(EKFParams, model_dims, True)
    
    estimators = {'CMGF-EKF': cmgf_est, 'Decoupled CMGF-EKF': d_cmgf_est}
    scores, times = compare_performance(estimators, input_dim, [1000, 10000, 50000, 100000, 500000])
