import dynamax.warnings

from jaxtyping import install_import_hook
with install_import_hook("dynamax", ("beartype", "beartype")):
    from .linear_gaussian_ssm.inference import ParamsLGSSMInitial