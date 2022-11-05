# Generalized Gaussian State Space Models

A GG-SSM is an SSM with nonlinear Gaussian dynamics, and nonlinear observations,
where the emission distribution can be non-Gaussian (e.g., Poisson, categorical).
To support approximateinference in this model, we can use the condititional moments
Gaussian filtering, which is a form of the generalized Gaussian filter where
the observation model is represented in terms of conditional first and second-order
moments, i.e., E[Y|z] and Cov[Y|z].

