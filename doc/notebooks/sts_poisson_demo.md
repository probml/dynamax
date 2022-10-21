---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3.8.14 64-bit
  language: python
  name: python3
---

# Structural time series (STS) model with Poisson likelihood

```{code-cell} ipython3
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

import dynamax.structural_time_series.models.structural_time_series as sts

import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
import time
```

## Synthetic Data

```{code-cell} ipython3
num_timesteps = 300
num_timesteps_training = 250
num_timesteps_forecast = num_timesteps - num_timesteps_training

np.random.seed(1)
observed_counts = np.round(3 + np.random.lognormal(np.log(np.linspace(
    num_timesteps, 5, num=num_timesteps)), 0.2, size=num_timesteps))
observed_counts = observed_counts.astype(np.float32)
observed_counts_training = observed_counts[:num_timesteps_training]
plt.figure(figsize=(12, 6))
plt.plot(observed_counts_training, lw=2)
plt.grid()
```

## Implementation in [TFP](https://www.tensorflow.org/probability/examples/STS_approximate_inference_for_models_with_non_Gaussian_observations)

+++

Instead of operating on the observed time series, the model operates on the series of Poisson rate
parameters that govern the observations, which is transformed to real values Softplus transformation
$y = \log(1 + \exp(x))$.

To use approximate inference for a non-Gaussian observation model, it encodes the STS model as a
TFP JointDistribution. The random variables in this joint distribution are the parameters of the
STS model, the time series of latent Poisson rates, and the observed counts.

It uses HMC (specifically, NUTS) to sample from the joint posterior over model parameters and latent
rates. This will be significantly slower than fitting a standard STS model with HMC, since in addition
to the model's (relatively small number of) parameters it also has to infer the entire series of
Poisson rates.

```{code-cell} ipython3
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

# Build STS model with only LocalLinearTrend component.
def build_model(approximate_unconstrained_rates):
    trend = tfp.sts.LocalLinearTrend(
        observed_time_series=approximate_unconstrained_rates)
    return tfp.sts.Sum([trend], observed_time_series=approximate_unconstrained_rates)

positive_bijector = tfb.Softplus()

approximate_unconstrained_rates = positive_bijector.inverse(
    tf.convert_to_tensor(observed_counts_training) + 0.01)

sts_model = build_model(approximate_unconstrained_rates)

# To use Poisson likelihood, it encodes the STS model as a TFP JointDistribution
def sts_with_poisson_likelihood_model():
    param_vals = []
    for param in sts_model.parameters:
        param_val = yield param.prior
        param_vals.append(param_val)
        
    unconstrained_rates = yield sts_model.make_state_space_model(
        num_timesteps_training, param_vals)
    rate = positive_bijector.forward(unconstrained_rates[..., 0])
    observed_counts = yield tfd.Poisson(rate, name='observed_counts')
    
model = tfd.JointDistributionCoroutineAutoBatched(sts_with_poisson_likelihood_model)
pinned_model = model.experimental_pin(observed_counts=observed_counts_training)
constraining_bijector = pinned_model.experimental_default_event_space_bijector()

# Inference with HMC
num_results = int(200)
num_burnin_steps = int(100)

sampler = tfp.mcmc.TransformedTransitionKernel(
    tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=pinned_model.unnormalized_log_prob,
        step_size=0.1),
    bijector=constraining_bijector)

adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=sampler,
    num_adaptation_steps=int(0.8 * num_burnin_steps),
    target_accept_prob=0.75)

initial_state = constraining_bijector.forward(
    type(pinned_model.event_shape)(
        *(tf.random.normal(part_shape)
          for part_shape in constraining_bijector.inverse_event_shape(
              pinned_model.event_shape))))

@tf.function(autograph=False, jit_compile=True)
def do_sampling():
    return tfp.mcmc.sample_chain(
        kernel=adaptive_sampler,
        current_state=initial_state,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        trace_fn=None)

t0 = time.time()
samples = do_sampling()
t1 = time.time()
print(f"Inference ran in {t1-t0:.3f}s")

# Posterior Samples
param_samples = samples[:-1]
unconstrained_rate_samples = samples[-1][..., 0]
rate_samples = positive_bijector.forward(unconstrained_rate_samples)

plt.figure(figsize=(12, 6))
mean_lower, mean_upper = np.percentile(rate_samples, [10, 90], axis=0)
pred_lower, pred_upper = np.percentile(np.random.poisson(rate_samples), [10, 90], axis=0)

_ = plt.plot(observed_counts_training, ls='--', marker='o', label='observed', alpha=0.7)
_ = plt.plot(np.mean(rate_samples, axis=0), label='rate', color='green', ls='dashed', lw=2, alpha=0.7)
_ = plt.fill_between(np.arange(0, num_timesteps_training), mean_lower, mean_upper, color='green', alpha=0.2)
_ = plt.fill_between(np.arange(0, num_timesteps_training), pred_lower, pred_upper, color='grey', label='counts', alpha=0.2)
plt.xlabel('Day')
plt.ylabel('Daily Sample Size')
plt.title('Posterior Mean')
plt.legend()
plt.grid()

# Forecasting
def sample_forecasted_counts(sts_model, posterior_latent_rates,
                             posterior_params, num_timesteps_forecast,
                             num_sampled_forecasts):
    
    unconstrained_rates_forecast_dist = tfp.sts.forecast(
        sts_model,
        observed_time_series=unconstrained_rate_samples,
        parameter_samples=posterior_params,
        num_steps_forecast=num_timesteps_forecast
        )
    rates_forecast_dist = tfd.TransformedDistribution(
        unconstrained_rates_forecast_dist, positive_bijector
        )
    sampled_latent_rates = rates_forecast_dist.sample(num_sampled_forecasts)
    sampled_forecast_counts = tfd.Poisson(rate=sampled_latent_rates).sample()

    return sampled_forecast_counts, sampled_latent_rates
    
forecast_samples, rate_samples = sample_forecasted_counts(
    sts_model, posterior_latent_rates=unconstrained_rate_samples,
    posterior_params=param_samples, num_timesteps_forecast=num_timesteps_forecast, num_sampled_forecasts=100)

forecast_samples = np.squeeze(forecast_samples)

def plot_forecast_helper(data, forecast_samples, CI=90):
    """Plot the observed time series alongside the forecast."""
    plt.figure(figsize=(12, 6))
    forecast_median = np.median(forecast_samples, axis=0)

    num_steps = len(data)
    num_steps_forecast = forecast_median.shape[-1]
    num_steps_training = num_steps - num_timesteps_forecast

    plt.plot(np.arange(num_steps), data, lw=2, linestyle='--', marker='o', label='Observed Data', alpha=0.7)

    forecast_steps = np.arange(num_steps_training, num_steps)

    CI_interval = [(100 - CI)/2, 100 - (100 - CI)/2]
    lower, upper = np.percentile(forecast_samples, CI_interval, axis=0)

    plt.plot(forecast_steps, forecast_median, lw=2, ls='--', marker='o', color='orange',
             label=str(CI) + '% Forecast Interval', alpha=0.7)
    plt.fill_between(forecast_steps, lower, upper, color='orange', alpha=0.2)

    plt.xlim([0, num_steps])
    ymin, ymax = min(np.min(forecast_samples), np.min(data)),\
                 max(np.max(forecast_samples), np.max(data))
    yrange = ymax-ymin
    plt.title("{}".format('Observed time series with ' + str(num_steps_forecast) + ' Day Forecast'))
    plt.xlabel('Day')
    plt.ylabel('Daily Sample Size')
    plt.legend()
    plt.grid()
    
plot_forecast_helper(observed_counts, forecast_samples, CI=80)
```

## Implementation via CMGF

```{code-cell} ipython3
observed_counts = jnp.array(observed_counts.reshape((num_timesteps, 1)))
observed_counts_training = observed_counts[:num_timesteps_training]
```

### Build the STS model

```{code-cell} ipython3
# The model includes the local linear trend component
from dynamax.distributions import InverseWishart as IW

trend = sts.LocalLinearTrend(observed_time_series=jnp.log(observed_counts_training))
                            #  level_covariance_prior=IW(df=1, scale=jnp.eye(1)),
                            #  slope_covariance_prior=IW(df=1, scale=jnp.eye(1)))
model = sts.StructuralTimeSeries([trend],
                                 observation_distribution_family='Poisson',
                                 observed_time_series=observed_counts_training)
```

### Model fitting

```{code-cell} ipython3
# Fit the model using HMC
observed_time_series = observed_counts_training
key = jr.PRNGKey(0)
sample_size = 200

parameter_samples = model.fit_hmc(key, sample_size, observed_time_series,
                                  inputs=None, warmup_steps=200, num_integration_steps=10)
```

### Forecast

```{code-cell} ipython3
forecasts = model.forecast(jr.PRNGKey(0), observed_time_series, parameter_samples, num_timesteps_forecast)
forecast_means = jnp.median(jnp.squeeze(forecasts['means']), axis=0)
CI = 90
CI_interval = jnp.array([(100 - CI)/2, 100 - (100 - CI)/2])
lower, upper = jnp.percentile(jnp.squeeze(forecasts['means']), CI_interval, axis=0)
forecast_scale = jnp.std(jnp.squeeze(forecasts['means']), axis=0)

# Plot the prediction
time_steps = jnp.arange(num_timesteps)
time_steps_forecast = time_steps[-num_timesteps_forecast:]

fig = plt.figure(figsize=(8, 4))
plt.plot(time_steps, observed_counts, lw=2, linestyle='--', marker='o',
         alpha=0.7, label='Observed Data')
plt.plot(time_steps_forecast, forecast_means, lw=2, linestyle='--', marker='o',
         color='orange', alpha=0.7, label='Forecast')
plt.fill_between(time_steps_forecast, lower, upper, color='orange', alpha=0.3)
# plt.fill_between(time_steps_forecast, forecast_means-2*forecast_scale, forecast_means+2*forecast_scale, color='orange', alpha=0.3)
plt.grid()
plt.legend()
```
