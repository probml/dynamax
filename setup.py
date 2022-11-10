#!/usr/bin/env python
import sys
from setuptools import setup
import versioneer

# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# Should match pyproject.toml
SETUP_REQUIRES = ['setuptools >= 30.3.0']
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []

if __name__ == '__main__':
    setup(name='dynamax',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          setup_requires=SETUP_REQUIRES,
          )

# #!/usr/bin/env python
# from distutils.core import setup
# import setuptools

# with open("requirements.txt") as f:
#     requirements = list(map(lambda x: x.strip(), f.read().strip().splitlines()))

# setup(
#     name="dynamax",
#     version="0.1",
#     description="JAX code for state space modeling and inference",
#     url="https://github.com/probml/dynamax",
#     install_requires=requirements,
#     packages=setuptools.find_packages(),
# )

