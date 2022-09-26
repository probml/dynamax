import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import einops
import matplotlib
from functools import partial
from collections import namedtuple

import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

import jax
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, grad, jit

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


#We use a variant of the example from eqn 2 from the NURD paper,
#in which y and z are now binary, with possible values -1,+1.
#p(x|y,z) = N(x | [y-bz, y+bz], diag(1.5, 0.5))  \\
#p(y) = Ber(0.5) \\
#p(z|y) = \rho \text{ if $z=y$}, 1-\rho \text{ otherwise} 
#
#NURD paper:
# A. M. Puli, L. H. Zhang, E. K. Oermann, and R. Ranganath, 
# “Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations,” 
#  ICLR, May 2022 [Online]. Available: https://openreview.net/forum?id=12RoR2o32T. 


nclasses = 2
nfactors = 2 # nuisance factors
nmix = nclasses * nfactors # mixture components

### Likelihood

def yz_to_mix(y, z):
    m = jnp.ravel_multi_index((jnp.array([y]), jnp.array([z])), (nclasses, nfactors))
    return m[0]

def mix_to_yz(m):
    yz = jnp.unravel_index(m, (nclasses, nfactors))
    return yz[0], yz[1]

def class_cond_density(y, z, b, sf):  # p(x|y,z)
    # convert from (0,1) to (-1,1)
    ysigned = 2.0*y-1
    zsigned = 2.0*z-1
    mu = jnp.array([ysigned - b*zsigned, ysigned + b*zsigned])
    Sigma = sf*jnp.diag(jnp.array([1.5, 0.5]))
    dist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=Sigma)
    #dist = multivariate_normal(mean=mu, cov=Sigma, seed=seed)
    return dist

def make_xgrid(npoints = 100):
    npoints = npoints * 1j
    xyrange = jnp.array([[-3, 3], [-3, 3]])
    mesh = jnp.mgrid[xyrange[0, 0] : xyrange[0, 1] : npoints, xyrange[1, 0] : xyrange[1, 1] : npoints]
    x1, x2 = mesh[0], mesh[1]
    points = jnp.vstack([jnp.ravel(x1), jnp.ravel(x2)]).T
    return points, x1, x2



class LikModel: # p(x|m)
    def __init__(self, b=1, sf=1):
        self.b = b
        self.sf = sf
        self.dist = [None] * nmix
        for y in [0,1]:
            for z in [0,1]:
                m = yz_to_mix(y, z)
                self.dist[m] = class_cond_density(y, z, self.b, self.sf)

    def sample(self, key, m, nsamples):
        xs = self.dist[m].sample(seed=key, sample_shape=(nsamples,))  
        return xs

    def prob(self, m, xs):
        # return p(n) = p(xs(n) | m)
        return self.dist[m].prob(xs) # tfd

    def plot_class_cond_dist(self):
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        points, x1, x2 = make_xgrid(100)
        for y in [0,1]:
            for z in [0,1]:
                m = yz_to_mix(y, z)
                ax = axs[y, z]
                p = self.prob(m, points).reshape(x1.shape[0], x2.shape[0])
                contour = ax.contourf(x1, x2, p)
                # cbar = fig.colorbar(contour, ax=ax)
                ax.set_title('y={:d}, z={:d}'.format(y, z))


### Prior

def make_mix_prior(rho):
    p_class =  np.array([0.5, 0.5]) # uniform prior  on p(y) = (-1, +1)
    p_factor_given_class = np.zeros((2, 2))  # p(z|c) = p_factor(c,z) so each row sums to 1
    p_factor_given_class[0, :] = [1 - rho, rho]
    p_factor_given_class[1, :] = [rho, 1 - rho]
    p_mix = np.zeros((nclasses, nfactors))  # (c,z)
    for c in range(nclasses):
        for z in range(nfactors):
            p_mix[c, z] = p_class[c] * p_factor_given_class[c, z]
    p_mix = einops.rearrange(p_mix, 'y z -> (y z)')
    return p_mix



class PriorModel:
    def __init__(self, rho):
        self.rho = rho
        self.prior_probs = make_mix_prior(rho)

    def sample(self, key, nsamples):
        labels =  jr.categorical(key, logits=jnp.log(self.prior_probs), shape=(nsamples,))
        return labels

    def prob(self, m):
        return self.prior_probs[m]


### Posterior

def normalize_probs(probs):
    # make each row sum to 1. Also works for multi-dim distributions eg 3d
    #S = einops.rearrange(probs, "i ... -> i (...)").sum(axis=-1)
    S = einops.reduce(probs, "i ... -> i", "sum")
    probs = np.einsum("i...,i->i...", probs, 1 / S)
    return probs

def compute_mix_post_bayes_rule(xs, mix_prior_probs, mix_lik_fn):  
    ndata = len(xs)
    nmix = 4
    mix_post = np.zeros((ndata, nmix))
    for m in range(nmix):
        mix_post[:, m] = mix_prior_probs[m] * mix_lik_fn(m, xs)
    #mix_post = normalize_probs(mix_post)
    norm = mix_post.sum(axis=1)
    mix_post = mix_post / jnp.expand_dims(norm, axis=1) 
    return mix_post

def compute_class_post(mix_post):
    mix_post = einops.rearrange(mix_post, 'n (y z) -> n y z', y=nclasses, z=nfactors)
    class_post = einops.reduce(mix_post, 'n y z -> n y', 'sum') 
    return class_post

def plot_class_post_2d(xs, post, fig=None, ax=None, ttl=None):
    x = xs[:,0]
    y = xs[:,1]
    z = post[:,0]
    if fig is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=100)
    norm = matplotlib.colors.Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm = norm), ax=ax)
    if ttl is not None:
      ax.set_title(ttl)

def plot_mix_post(mix_post, nmax=100, ttl=None):
  ntrain, nmix = mix_post.shape
  # select a subset of points to plot
  n = np.min([ntrain, nmax])
  ndx = np.round(np.linspace(1, ntrain-1, num=n)).astype(int)
  colors = ['r', 'g', 'b', 'k']
  plt.figure()
  for m in range(nmix):
    y, z = mix_to_yz(m)
    plt.plot(mix_post[ndx,m], colors[m], label='m={:d},y={:d},z={:d}'.format(m,y,z))
  plt.legend()
  if ttl is not None:
    plt.title(ttl)

def plot_mix_post_pair(mix_post1, mix_post2, nmax=100):
  ntrain, nmix = mix_post1.shape
  colors = ['r', 'g', 'b', 'k']
  fig, axs = plt.subplots(2, 2, figsize=(10,10))
  axs = axs.reshape(-1)
  # select a subset of points to plot
  n = np.min([ntrain, nmax])
  ndx = np.round(np.linspace(1, ntrain-1, num=n)).astype(int)
  for m in range(nmix):
    ax = axs[m]
    ax.plot(mix_post1[ndx,m], color=colors[m], ls='-', label='m={:d}, method 1'.format(m))
    ax.plot(mix_post2[ndx,m], color=colors[m], ls=':', label='m={:d}, method 2'.format(m))
    ax.legend()

class JointModel:
    def __init__(self, lik_model, prior_model):
        self.lik_model = lik_model
        self.prior_model = prior_model
        self.lik_fn = lambda m, xs: self.lik_model.prob(m, xs)
        self.prior = prior_model.prior_probs

    def predict(self, xs):
        mix_post = compute_mix_post_bayes_rule(xs, self.prior, self.lik_fn)
        class_post = compute_class_post(mix_post)
        return mix_post, class_post

    def predict_target(self, xs):
        return self.predict(xs)

    def sample(self, key, nsamples):
        labels = self.prior_model.sample(key, nsamples)
        ndim = 2
        X = np.zeros((nsamples, ndim))
        nmix = 4
        for m in range(nmix):
            ndx = np.nonzero(labels==m)[0]
            xs = self.lik_model.sample(key, m, len(ndx))  
            X[ndx, :] = xs
        return X, labels



### Training

def est_empirical_mix_dist(labels):
  nmix = 4
  ndata = len(labels)
  counts = np.zeros(nmix)
  for m in range(nmix):
    counts[m] = np.sum(labels==m)
  assert np.allclose(np.sum(counts), ndata)
  return counts / ndata

def em(Xtarget, init_dist, lik_fn, niter=10, pseudo_counts=np.zeros(4), verbose=False):
  target_dist = init_dist
  ndata = Xtarget.shape[0]
  nmix = 4
  post = np.zeros((ndata, nmix))
  for t in range(niter):
    if verbose: print(f'EM iter {t}')
    # E step
    for m in range(nmix):
      post[:, m] = lik_fn(m, Xtarget) * target_dist[m] 
    post = normalize_probs(post)
    # M step
    ncounts = np.zeros(nmix)
    for m in range(nmix):
      ncounts[m] = np.sum(post[:, m]) + pseudo_counts[m]
    target_dist = ncounts / np.sum(ncounts)
  return target_dist



class Estimator:
    def __init__(self, max_classifier_iter=500):
        self.prior_source = None
        self.prior_target = None
        self.lik_fn = lambda m, xs: self.likelihood_from_classifier(m, xs)
        self.max_classifier_iter = max_classifier_iter
        self.classifier = Pipeline([
            ('standardscaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)), 
            ('logreg', LogisticRegression(random_state=0, max_iter=self.max_classifier_iter))])

    def fit_source(self, X, mix_labels, jm=None, key=None):
        self.classifier.fit(X, mix_labels)
        probs = self.classifier.predict_proba(X) # (n,m)
        self.prior_source = np.mean(probs, axis=0)

    def fit_target(self, X, jm=None): 
        pass # details depend on the method

    def likelihood_from_classifier(self, m, xs):
        # return p(n) = p_s(x(n) | m) = p_s(m|x) p_s(x) / p_s(m) propto p_s(m|x) / p_s(m)
        n = xs.shape[0]
        probs = self.classifier.predict_proba(xs)[:,m] / self.prior_source[m]
        return probs

    def predict_source(self, xs):
        mix_post = self.classifier.predict_proba(xs)
        class_post = compute_class_post(mix_post)
        return mix_post, class_post

    def predict_target(self, xs):
        mix_post = compute_mix_post_bayes_rule(xs, self.prior_target, self.lik_fn)
        class_post = compute_class_post(mix_post)
        return mix_post, class_post

class Estimator_EM(Estimator):
    def __init__(self,  max_classifier_iter=500, max_em_iter=5, prior_strength=0.01):
        super().__init__(max_classifier_iter)
        self.max_em_iter = max_em_iter
        self.prior_strength = prior_strength

    def fit_target(self, X, jm=None): 
        nmix = 4
        self.prior_target = em(X, self.prior_source, self.lik_fn,
            self.max_em_iter, self.prior_strength * np.ones(nmix))


class Estimator_True_Prior(Estimator):
    def __init__(self):
        super().__init__()
    
    def fit_target(self, X, jm_target): # cheat by setting prior target to truth instead of using EM
        self.prior_target = jm_target.prior


class Estimator_Unadapted(Estimator):
    def __init__(self):
        super().__init__()
    
    def predict_target(self, xs): # use source model
        return self.predict_source(xs)


class Estimator_Unadapted_UnifSource(Estimator):
    def __init__(self):
        super().__init__()
    
    def fit_source(self, X_source, mix_labels_source, jm_source, key):
        # generate a new set of training data from a uniform version of the source
        pm_source = PriorModel(0.5)
        lm = jm_source.lik_model
        jm_source = JointModel(lm, pm_source)
        X_train_unif, mix_train_unif = jm_source.sample(key,  500)
        self.classifier.fit(X_train_unif, mix_train_unif)
        probs = self.classifier.predict_proba(X_train_unif) # (n,m)
        self.prior_source = np.mean(probs, axis=0)

    def predict_target(self, xs): # use source model
        return self.predict_source(xs)

#### Evaluation

def evaluate_class_post(class_post_true, class_post_pred):
  # we use mean squared error for simplicity
  return jnp.mean(jnp.power(class_post_true[:,0] - class_post_pred[:,0], 2))

def evaluate_shift(key, lm, rho_source, rho_targets, model,
                    ntrain_source=500, ntrain_target=100):
    pm_source = PriorModel(rho_source)
    jm_source = JointModel(lm, pm_source)
    X_train_source, mix_train_source = jm_source.sample(key,  ntrain_source)
    # we pass in the true jm_sourcet as a backdoor for oracles
    model.fit_source(X_train_source, mix_train_source, jm_source, key)

    xs, x1, x2 = make_xgrid(npoints = 100)
    ndomains = len(rho_targets)
    loss_per_domain = np.zeros(ndomains)
    for i in range(ndomains):
        rho = rho_targets[i]
        pm_target = PriorModel(rho = rho)
        jm_target = JointModel(lm, pm_target)
        X_train_target, mix_train_target = jm_target.sample(key,  ntrain_target)

        mix_post_true, class_post_true = jm_target.predict_target(X_train_target)

        # we pass in the true jm_target as a backdoor for oracles
        model.fit_target(X_train_target, jm_target) 
        mix_post_pred, class_post_pred = model.predict_target(X_train_target)

        err = evaluate_class_post(class_post_true, class_post_pred)
        loss_per_domain[i] = err
    return loss_per_domain

def run_expt(rho_source=0.3, ntrials=3, sf=3):
    rho_targets = np.linspace(0.1, 0.9, num=9)
    lm = LikModel(b=1, sf=sf)

    key = jr.PRNGKey(420)
    keys = jr.split(key, ntrials)
    ndomains = len(rho_targets)

    methods = {
        'EM': Estimator_EM(),
        'TruePrior': Estimator_True_Prior(),
        'SourceUnadapted': Estimator_Unadapted(),
        'UnifSourceUnadapted': Estimator_Unadapted_UnifSource()
    }

    nmethods = len(methods)
    losses = {}
    losses_mean = {}
    losses_std = {}
    for name, estimator in methods.items():
        print(name)
        losses_per_trial =  np.zeros((ndomains, ntrials))
        for i in range(ntrials):
            losses_per_trial[:, i] =  evaluate_shift(keys[i], lm, rho_source, rho_targets,  estimator)
        losses[name] = losses_per_trial
        losses_mean[name] = np.mean(losses_per_trial, axis=1)
        losses_std[name] = np.std(losses_per_trial, axis=1)


    plt.figure()
    for name in methods.keys():
        plt.errorbar(rho_targets, losses_mean[name], yerr=losses_std[name], marker='o', label=name)
    plt.xlabel('correlation')
    plt.ylabel('L1 error of class 1')
    plt.title(r'Source $\rho={:0.1f}, sf={:0.1f}$'.format(rho_source, lm.sf))
    plt.legend()
    plt.axvline(x=rho_source);