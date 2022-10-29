#!/usr/bin/env python
from distutils.core import setup
import setuptools

with open("requirements.txt") as f:
    requirements = list(map(lambda x: x.strip(), f.read().strip().splitlines()))

setup(
    name="dynamax",
    version="0.1",
    description="JAX code for state space modeling and inference",
    url="https://github.com/probml/dynamax",
    install_requires=requirements,
    packages=setuptools.find_packages(),
)
