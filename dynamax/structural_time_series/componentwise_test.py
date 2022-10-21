import jax.numpy as jnp
import jax.random as jr
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

import dynamax.structural_time_series.models.structural_time_series as sts


def test_local_linear(key=10, num_timesteps=100):
    # Synthetic data using STS model with only local_linear_trend component
    F = jnp.array([[1, 1],
                   [0, 1]])
    H = jnp.array([[1, 0]])
    dynamic_noise = MVN()
    
    initial_state = 0
    
    def _step():
        return state, ()
    
    _ state, obs = lax.scan()
    
    # Model fit using MLE
    params = 
    
    assert True

def test_seasonal():
    return

def test_linear_regression():
    assert True



co2_by_month = jnp.array(co2_by_month[:, None])
num_forecast_steps = 12 * 10  # Forecast the final ten years, given previous data
co2_by_month_training_data = co2_by_month[:-num_forecast_steps]


def test_sts_fit_hmc(key=jr.PRNGKey(0),
                     observed_time_series=co2_by_month_training_data,
                     future_observations=co2_by_month[-num_forecast_steps:],
                     sample_size=100):
    # Define a STS model
    trend = sts.LocalLinearTrend(observed_time_series=co2_by_month_training_data)
    seasonal = sts.Seasonal(num_seasons=12, observed_time_series=co2_by_month_training_data)
    model = sts.StructuralTimeSeries([trend, seasonal],
                                     observation_distribution_family='Gaussian',
                                     observed_time_series=co2_by_month_training_data)

    # Fit the model using HMC
    parameter_samples = model.fit_hmc(key, sample_size, observed_time_series,
                                      inputs=None, warmup_steps=500, num_integration_steps=30)
    predicts = model.forecast(key, observed_time_series, parameter_samples, num_forecast_steps)

    pred_means = predicts['means']
    pred_scales = jnp.sqrt(jnp.squeeze(predicts['covariances']))
    pred_error = jnp.abs(pred_means - future_observations)

    assert jnp.mean(pred_error) < 10 # 3
    assert jnp.mean(pred_scales) < 10 # 5
