from . import _version
__version__ = _version.get_versions()['version']

# Catch expected warnings from TFP
import dynamax.warnings

# Default to float32 matrix multiplication on TPUs and GPUs
import jax
jax.config.update('jax_default_matmul_precision', 'float32')

from . import _version
__version__ = _version.get_versions()['version']
