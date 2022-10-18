import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt

import dynamax.structural_time_series.models.structural_time_series as sts


class CausalImpact():
    """A wrapper of help functions of the causal impact
    """
    def __init__(self, sts_model, intervention_time, predict, causal_impact, observed_timeseries):
        """
        Args:
            sts_model    : an object of the StructualTimeSeries class
            causal_impact: a dict returned by the function 'causal impact'
        """
        self.intervention_time = intervention_time
        self.sts_model = sts_model
        self.predict_point = predict['pointwise']
        self.predict_interval = predict['interval']
        self.impact_point = causal_impact['pointwise']
        self.impact_cumulat = causal_impact['cumulative']
        self.timeseries = observed_timeseries

    def plot(self):
        """Plot the causal impact
        """
        x = jnp.arange(self.timeseries.shape[0])
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(9, 6), sharex=True, layout='constrained')

        # Plot the original obvervation and the counterfactual predict
        ax1.plot(x, self.timeseries, color='black', lw=2, label='Observation')
        ax1.plot(x, self.predict_point, linestyle='dashed', color='blue', lw=2, label='Prediction')
        ax1.fill_between(x, self.predict_interval[0], self.predict_interval[1],
                         color='blue', alpha=0.2)
        ax1.axvline(x=self.intervention_time, linestyle='dashed', color='gray', lw=2)
        ax1.set_title('Original time series')

        # Plot the pointwise causal impact
        ax2.plot(x, self.impact_point[0], linestyle='dashed', color='blue')
        ax2.fill_between(x, self.impact_point[1][0], self.impact_point[1][1],
                         color='blue', alpha=0.2)
        ax2.axvline(x=self.intervention_time, linestyle='dashed', color='gray', lw=2)
        ax2.set_title('Poinwise causal impact')

        # Plot the cumulative causal impact
        ax3.plot(x, self.impact_cumulat[0], linestyle='dashed', color='blue')
        ax3.fill_between(x, self.impact_cumulat[1][0], self.impact_cumulat[1][1],
                         color='blue', alpha=0.2)
        ax3.axvline(x=self.intervention_time, linestyle='dashed', color='gray', lw=2)
        ax3.set_title('Cumulative causal impact')

        plt.show()

    def summary(self):
        msg = "The posterior causl effect"
        print(msg)


def causal_impact(observed_timeseries,
                  intervention_time,
                  distribution_family,
                  inputs=None,
                  sts_model=None,
                  confidence_level=0.95,
                  key=jr.PRNGKey(0),
                  sample_size=200):
    """Inferring the causal impact of an intervention on a time series,
    given the observed timeseries before and after the intervention.

    The causal effect is obtained by conditioned on, and only on, the observations,
    with parameters and latent states integrated out.

    Returns:
        An object of the CausalImpact class
    """
    assert distribution_family in ['Gaussian', 'Poisson']
    if sts_model is not None:
        assert distribution_family == sts_model.obs_family

    key1, key2, key3 = jr.split(key, 3)
    num_timesteps, dim_obs = observed_timeseries.shape

    # Split the data into pre-intervention period and post-intervention period
    timeseries_pre = observed_timeseries[:intervention_time]
    timeseries_pos = observed_timeseries[intervention_time:]

    if inputs is not None:
        dim_inputs = inputs.shape[-1]
        # The number of time steps of input must equal to that of observed time_series.
        inputs_pre = inputs[:intervention_time]
        inputs_pos = inputs[intervention_time:]

    # Construct a STS model with only local linear trend by default
    if sts_model is None:
        local_linear_trend = sts.LocalLinearTrend(observed_timeseries=observed_timeseries)
        if inputs is None:
            sts_model = sts.StructuralTimeSeries(components=[local_linear_trend],
                                                 observed_timeseries=observed_timeseries,
                                                 observation_distribution_family=distribution_family)
        else:
            linear_regression = sts.LinearRegression(weights_shape=(dim_obs, dim_inputs))
            sts_model = sts.StructuralTimeSeries(components=[local_linear_trend, linear_regression],
                                                 observed_timeseries=observed_timeseries,
                                                 observation_distribution_family=distribution_family)

    # Fit the STS model, sample from the past and forecast.
    if inputs is not None:
        # Model fitting
        print('Fit the model using HMC...')
        params_posterior_samples = sts_model.fit_hmc(key1, sample_size, timeseries_pre, inputs_pre)
        print("Model fitting completed.")
        # Sample from the past and forecast
        samples_pre = sts_model.posterior_sample(
            key2, timeseries_pre, params_posterior_samples, inputs_pre)
        samples_pos = sts_model.forecast(
            key3, timeseries_pre, params_posterior_samples, timeseries_pos.shape[0],
            inputs_pre, inputs_pos)
    else:
        # Model fitting
        print('Fit the model using HMC...')
        params_posterior_samples = sts_model.fit_hmc(key1, sample_size, timeseries_pre)
        print("Model fitting completed.")
        # Sample from the past and forecast
        samples_pre = sts_model.posterior_sample(key2, timeseries_pre, params_posterior_samples)
        samples_pos = sts_model.forecast(
            key3, timeseries_pre, params_posterior_samples, timeseries_pos.shape[0])

    # forecast_means = jnp.concatenate((samples_pre['means'], samples_pos['means']), axis=1).squeeze()
    forecast_observations = jnp.concatenate(
        (samples_pre['observations'], samples_pos['observations']), axis=1).squeeze()

    confidence_bounds = jnp.quantile(
        forecast_observations,
        jnp.array([0.5 - confidence_level/2., 0.5 + confidence_level/2.]),
        axis=0)
    predict_point = forecast_observations.mean(axis=0)
    predict_interval_upper = confidence_bounds[0]
    predict_interval_lower = confidence_bounds[1]

    cum_predict_point = jnp.cumsum(predict_point)
    cum_confidence_bounds = jnp.quantile(
        forecast_observations.cumsum(axis=1),
        jnp.array([0.5-confidence_level/2., 0.5+confidence_level/2.]),
        axis=0
        )
    cum_predict_interval_upper = cum_confidence_bounds[0]
    cum_predict_interval_lower = cum_confidence_bounds[1]

    # Evaluate the causal impact
    impact_point = observed_timeseries.squeeze() - predict_point
    impact_interval_lower = observed_timeseries.squeeze() - predict_interval_upper
    impact_interval_upper = observed_timeseries.squeeze() - predict_interval_lower

    cum_timeseries = jnp.cumsum(observed_timeseries.squeeze())
    cum_impact_point = cum_timeseries - cum_predict_point
    cum_impact_interval_lower = cum_timeseries - cum_predict_interval_upper
    cum_impact_interval_upper = cum_timeseries - cum_predict_interval_lower

    impact = {'pointwise': (impact_point, (impact_interval_lower, impact_interval_upper)),
              'cumulative': (cum_impact_point, (cum_impact_interval_lower, cum_impact_interval_upper))}

    predict = {'pointwise': predict_point,
               'interval': confidence_bounds}

    return CausalImpact(sts_model, intervention_time, predict, impact, observed_timeseries)
