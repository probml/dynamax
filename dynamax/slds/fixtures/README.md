We test the agreement of SLDS and rSLDS between Dynamax and [`lindermanlab/ssm`](https://github.com/lindermanlab/ssm). `export_ssm_fixtures.py` creates the SSM reference outputs (`ssm_fixtures.npz` and `ssm_benchmark.npz`) that the tests (`ssm_oracle_test.py` and `benchmark_test.py`) compare against.

To create the SSM reference outputs:

```bash
conda create -n dyna_dev_ssm python=3.10 -y && conda activate dyna_dev_ssm
pip install -e '.[dev]'
# ssm is not on PyPI and its setup.py imports numpy and Cython at build time: cython first, no build isolation
pip install cython
pip install --no-build-isolation "git+https://github.com/lindermanlab/ssm.git@eb6c8aa"
python dynamax/slds/fixtures/export_ssm_fixtures.py               # writes ssm_fixtures.npz
python dynamax/slds/fixtures/export_ssm_fixtures.py --benchmark   # writes ssm_benchmark.npz
```
