import tensorflow_probability.substrates.jax.bijectors as tfb

# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation
class PSDToRealBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
            tfb.Invert(tfb.FillTriangular()),
            tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
            tfb.Invert(tfb.CholeskyOuterProduct()),
        ]
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)


class RealToPSDBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
            tfb.CholeskyOuterProduct(),
            tfb.TransformDiagonal(tfb.Exp()),
            tfb.FillTriangular(),
        ]
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)
