# ReBayes = Recursive Bayesian inference for laten states

We provide code for online (recursive) Bayesian inference in a state space model
with Gaussian random walk dynamics (optionally stationary)
and nonlinear observations, where the emission distribution can be non-Gaussian (e.g., Poisson, categorical).
For approximate inference, we use the extended Kalman filter, generalized so that
it can work with conditional first and second moments of the observation distribution,
i.e., E[Y|z] and Cov[Y|z].

