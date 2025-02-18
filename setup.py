#!/usr/bin/env python
import setuptools
import versioneer

if __name__ == '__main__':
    setuptools.setup(name='dynamax',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass()
          )
