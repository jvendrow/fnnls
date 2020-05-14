# fnnls

[![PyPI Version](https://img.shields.io/pypi/v/fnnls.svg)](https://pypi.org/project/fnnls/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/fnnls.svg)](https://pypi.org/project/fnnls/)

An implementation of the Fast Nonnegative Least Squares (fnnls) algorithm presented in the paper "A fast non‐negativity‐constrained least squares algorithm" by Bro and Jung:

Bro, Rasmus, and Sijmen De Jong. "A fast non‐negativity‐constrained least squares algorithm." _Journal of Chemometrics: A Journal of the Chemometrics Society_ 11, no. 5 (1997): 393-401.

Given a matrix Z and vector x, this algorithm aims to find the optimal d to minimize || x - Zd || subject to d >= 0.

The fnnls algorithm is most comparable to the Lawson and Hanson algorithm for nonegative least squares published in 1974, which is the standard algorithm used by the SciPy library. **In practice, the algorithm converges to the nonnegative least square solution faster than the SciPy implementation of the Lawson and Hanson algorithm for tall and large matrices.** 

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

## Authors
* Joshua Vendrow
* Jamie Haddock

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
