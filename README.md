# fnnls

[![PyPI Version](https://img.shields.io/pypi/v/fnnls.svg)](https://pypi.org/project/fnnls/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/fnnls.svg)](https://pypi.org/project/fnnls/)

Fast Nonnegative Least Squares

---

## Installation

To install fnnls, run this command in your terminal:

```bash
$ pip install -U fnnls
```

This is the preferred method to install fnnls, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Quick Start
```python
>>> from fnnls import fnnls
>>> A = np.abs(np.random.rand(5,10)) 
>>> b = np.abs(np.random.rand(5))
>>> fnnls(A,b)
[array([0.08743951, 0.        , 0.        , 0.        , 0.00271862,
       0.        , 0.        , 0.01767932, 0.        , 0.28705341]), 0.07496726413383449]

```

## Citing
If you use our work in an academic setting, please cite our paper:



## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
