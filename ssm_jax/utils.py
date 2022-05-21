import tensorflow_probability.substrates.jax.bijectors as tfb

# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation
PSDToRealBijector = tfb.Chain([
    tfb.Invert(tfb.FillTriangular()),
    tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
    tfb.Invert(tfb.CholeskyOuterProduct()),
])