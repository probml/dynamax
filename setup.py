#!/usr/bin/env python
from distutils.core import setup
import setuptools

setup(
    name="ssm_jax",
    version="0.1",
    description="JAX code for state space modeling and inference",
    url="https://github.com/probml/ssm-jax",
    install_requires=[
        "jax",
        "jaxlib",
        "optax",
        "chex",
        "distrax",
        "tensorflow_probability",
        "filterpy"
    ],
    packages=setuptools.find_packages(),
)
