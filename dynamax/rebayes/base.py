from abc import ABC
from abc import abstractmethod
import time

import jax.numpy as jnp
from jax import jacrev, jacfwd, vmap
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
import chex
# from jax_tqdm import scan_tqdm #TODO: Figure out why this fails in GH Workflow and add back in

_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
MVN = tfd.MultivariateNormalFullCovariance


FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]

FnStateToEmission2 = Callable[[Float[Array, "state_dim"]], Float[Array, "emission_dim emission_dim"]]
FnStateAndInputToEmission2 = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "emission_dim emission_dim"]]
EmissionDistFn = Callable[ [Float[Array, "state_dim"], Float[Array, "state_dim state_dim"]], tfd.Distribution]

CovMat = Union[float, Float[Array, "dim"], Float[Array, "dim dim"]]

class GaussianBroken(NamedTuple):
    mean: Float[Array, "state_dim"]
    cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "state_dim"]]

@chex.dataclass
class Gaussian:
    mean: chex.Array
    cov: chex.Array

Belief = Gaussian # Can be over-ridden by other representations (e.g., MCMC samples or memory buffer)

class RebayesParams(NamedTuple):
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: CovMat
    dynamics_weights: Float[Array, "state_dim state_dim"]
    dynamics_covariance: CovMat
    #emission_function: FnStateAndInputToEmission
    #emission_covariance: CovMat
    emission_mean_function: FnStateAndInputToEmission
    emission_cov_function: FnStateAndInputToEmission2
    emission_dist: EmissionDistFn = lambda mean, cov: MVN(loc=mean, covariance_matrix=cov) 
    #emission_dist=lambda mu, Sigma: tfd.Poisson(log_rate = jnp.log(mu))



class Rebayes(ABC):
    def __init__(
        self,
        params: RebayesParams,
    ):
        self.params = params
        #self.emission_mean_function = lambda z, u: self.emission_function(z, u)
        #self.emission_cov_function = lambda z, u: self.params.emission_covariance

    def init_bel(self):
        return Gaussian(mean=self.params.initial_mean, cov=self.params.initial_covariance)

    def predict_state(
        self,
        bel: Gaussian,
        u: Float[Array, "input_dim"]
    ) -> Gaussian:
        """Given bel(t-1|t-1) = p(z(t-1) | D(1:t-1)), return bel(t|t-1) = p(z(t) | u(t), D(1:t-1)).
        This is cheap, since the dyanmics model is linear-Gaussian.
        """
        m, P = bel.mean, bel.cov 
        F = self.params.dynamics_weights
        Q = self.params.dynamics_covariance
        pred_mean = F @ m
        pred_cov = F @ P @ F.T + Q
        return Gaussian(mean=pred_mean, cov=pred_cov)

    def predict_obs(
        self,
        bel: Gaussian,
        u: Float[Array, "input_dim"]
    ) -> Gaussian: # TODO: replace output with emission_dist
        """Given bel(t|t-1) = p(z(t) | D(1:t-1)), return obs(t|t-1) = p(y(t) | u(t), D(1:t-1))"""
        prior_mean, prior_cov = bel.mean, bel.cov # p(z(t) | y(1:t-1))
        # Partially apply fn to the input u so it just depends on hidden state z
        m_Y = lambda z: self.params.emission_mean_function(z, u)
        Cov_Y = lambda z: self.params.emission_cov_function(z, u)

        yhat = jnp.atleast_1d(m_Y(prior_mean))
        R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)

        Sigma_obs = H @ prior_cov @ H.T + R
        return Gaussian(mean=yhat, cov=Sigma_obs)

    @abstractmethod
    def update_state(
        self,
        bel: Gaussian,
        u: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> Gaussian:
        """Return bel(t|t) = p(z(t) | u(t), y(t), D(1:t-1)) using bel(t|t-1)"""
        raise NotImplementedError



    def scan(
        self,
        X: Float[Array, "ntime input_dim"],
        Y: Float[Array, "ntime emission_dim"],
        callback=None
    ) -> Tuple[Gaussian, Any]:
        """Apply filtering to entire sequence of data. Return final belief state and outputs from callback."""
        num_timesteps = X.shape[0]
        # @scan_tqdm(num_timesteps)
        def step(bel, t):
            pred_obs = self.predict_obs(bel, X[t])
            bel = self.predict_state(bel, X[t])
            bel = self.update_state(bel, X[t], Y[t])
            out = None
            if callback is not None:
                out = callback(pred_obs, bel, t, X[t], Y[t])
            return bel, out

        carry = self.init_bel()
        bel, outputs = scan(step, carry, jnp.arange(num_timesteps))
        return bel, outputs
    