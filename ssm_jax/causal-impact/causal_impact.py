import jax.numpy as jnp

import matplotlib.pyplot as plt

import ssm_jax.structural_time_series.models.structural_time_series as sts


class CausalImpact():
    """A wrapper of help functions of the causal impact
    """
    def __init__(self, sts_model, predict, causal_impact, observed_timeseries):
        """
        Args:
            sts_model    : an object of the StructualTimeSeries class
            causal_impact: a dict returned by the function 'causal impact'
        """
        self.sts_model = sts_model
        self.predict_point = predict[0]
        self.predict_interval = predict[1]
        self.impact_point = causal_impact['pointwise']
        self.impact_cumulat = causal_impact['cumulative']
        self.timeseries = observed_timeseries

    def plot(self):
        """Plot the causal impact
        """
        x = jnp.arange(self.timeseries.shape[0])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 2.7),
                                            sharex=True, layout='constrained')

        # Plot the original obvervation and the counterfactual predict
        ax1.plot(x, self.timeseries)
        ax1.plot(x, self.predict_point)
        ax1.fill_btween(x, self.predict_interval[0], self.predict_interval[1])
        ax1.set_title('Original time series')

        # Plot the pointwise causal impact
        ax2.plot(x, self.impact_point[0])
        ax2.plot(x, self.impact_point[1], self.impact_point[2])
        ax2.set_title('Poinwise causal impact')

        # Plot the cumulative causal impact
        ax3.plot(x, self.impact_cumulat[0])
        ax3.plot(x, self.impact_cumulat[1], self.impact_cumulat[2])
        ax3.set_title('Cumulative causal impact')

        plt.show()

    def summary(self):
        msg = "The posterior causl effect"

        print(msg)


def causal_impact(observed_timeseries, pre_period, post_period,
                  sts_model=None, input=None, confidence_level=0.95):
    """Inferring the causal impact of an intervention on a time series,
    given the observed timeseries before and after the intervention.

    The causal effect is obtained by conditioned on, and only on, the observations,
    with parameters and latent states integrated out.

    Args:
        observed_timeseries (_type_): _description_
        pre_period (_type_): _description_
        post_period (_type_): _description_
        sts_model (_type_, optional): _description_. Defaults to None.
        input (_type_, optional): _description_. Defaults to None.

    Returns:
        An object of the CausalImpact class
    """
    num_timesteps = observed_timeseries.shape[0]
    assert num_timesteps == post_period[1] - post_period[0] + pre_period[1] - pre_period[0],\
        "The length of observed time series must equal to the sum of the length of \
         the pre-intervention period and the length of the post intervention period."

    # Construct a STS model with only local linear trend by default
    if sts_model is None:
        local_linear_trend = sts.LocalLinearTrend(observed_timeseries=observed_timeseries)
        sts_model = sts.StructuralTimeSeries(components=local_linear_trend,
                                             observed_timeseries=observed_timeseries)

    # Split the data into pre-intervention period and post-intervention period
    timeseries_pre = observed_timeseries[pre_period[0]:pre_period[1]]
    timeseries_pos = observed_timeseries[post_period[0]:post_period[1]]

    if input is not None:
        assert input.shape[0] == observed_timeseries.shape[0], \
            "The number of time steps of input must equal to that of observed time_series."
        input_pre = input[pre_period[0]:pre_period[1]]
        input_pos = input[post_period[0]:post_period[1]]

    # Posterior inference of the parameters
    params_posterior_samples = sts_model.hmc(timeseries_pre, input_pre)

    # Predict the counterfactual observation, as well as the pre-intervention sample
    samples_pre = sts_model.posterior_sample(params_posterior_samples, input_pre)
    samples_pos = sts_model.forecast(params_posterior_samples, timeseries_pos, input_pos)
    samples = jnp.concatenate((samples_pre, samples_pos), aixs=1)

    confidence_bounds = jnp.quantile(samples['observations'],
                                     (0.5 - confidence_level/2., 0.5 + confidence_level/2.),
                                     axis=0)
    predict_point = samples['state'].mean(axis=0)
    predict_interval_upper = confidence_bounds[0]
    predict_interval_lower = confidence_bounds[1]

    cum_predict_point = jnp.cumsum(predict_point)
    cum_confidence_bounds = jnp.quantile(samples['observations'].cumsum(axis=1),
                                         (0.5 - confidence_level/2., 0.5 + confidence_level/2.),
                                         axis=0)
    cum_predict_interval_upper = cum_confidence_bounds[0]
    cum_predict_interval_lower = cum_confidence_bounds[1]

    # Evaluate the causal impact
    impact_point = observed_timeseries - predict_point
    impact_interval_lower = observed_timeseries - predict_interval_upper
    impact_interval_upper = observed_timeseries - predict_interval_lower

    cum_timeseries = jnp.cumsum(observed_timeseries)
    cum_impact_point = cum_timeseries - cum_predict_point
    cum_impact_interval_lower = cum_timeseries - cum_predict_interval_upper
    cum_impact_interval_upper = cum_timeseries - cum_predict_interval_lower

    impact = {'pointwise': (impact_point, (impact_interval_lower, impact_interval_upper)),
              'cumulative': (cum_impact_point, (cum_impact_interval_lower, cum_impact_interval_upper))}

    predict = (predict_point, confidence_bounds)

    return CausalImpact(sts_model, predict, impact, observed_timeseries)
