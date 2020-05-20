# fnnls

[![PyPI Version](https://img.shields.io/pypi/v/fnnls.svg)](https://pypi.org/project/fnnls/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/fnnls.svg)](https://pypi.org/project/fnnls/)

An implementation of the Fast Nonnegative Least Squares (fnnls) algorithm presented in the paper "A fast non‐negativity‐constrained least squares algorithm" by Bro and De Jung:

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

## Usage

**Quick Start**

It's easy to directly solve a nonnegative least squares task with `fnnls`: 
```python
>>> import numpy as np
>>> from fnnls import fnnls
>>> np.random.seed(1)
>>> Z = np.abs(np.random.rand(5,10)) 
>>> x = np.abs(np.random.rand(5))
>>> fnnls(Z,x)
[array([0.  , 0.  , 0.  , 0.49507457, 0.  ,
0.  , 0.10518829, 0.  , 0.  , 0.  ]), 0.29527550874513586]

```
**Custom least squared functions:**

The fast nonegative least square algorithm solves an unconstrained least squares task for every iteration. By default, we use numpy.linalg.lstsq. 

We provide more custom functions to solve the least squares algorithm, and give users the ability to define and use their own functions. 
```python
>>> import numpy as np
>>> from fnnls import fnnls
>>> from fnnls import RK #the Randomized Karzmarz method
>>> np.random.seed(1)
>>> RK1 = lambda Z, x: RK(Z,x,random_state=1) #Set random state
>>> Z = np.abs(np.random.rand(5,10)) 
>>> x = np.abs(np.random.rand(5))
>>> fnnls(Z,x,lstsq=RK1)
[array([0.  , 0.  , 0.  , 0.22992788, 0.  ,
0.19111572, 0.15289165, 0.10472243, 0.  , 0.  ]), 0.3198075779021192]
```
Note that to set a random state above for RK, we had to define a new function RK1.

**Initializing the Passive Set**

The fast nonnegative least squares algorithm is a combinatorial algorithm that continually updates a passive set P to indicate the support (non-zero elements) of the solution at the current iteration. Often, it is possible to have knowledge of an estimate for the support of the solution, which can improve the efficiency of the algorithm. We allow users to choose to input an estimate for the support.
```python
>>> import numpy as np
>>> from fnnls import fnnls
>>> from time import time
>>> np.random.seed(1)
>>> Z = np.abs(np.random.rand(100,100))
>>> x = np.abs(np.random.rand(100))
>>> start = time()
>>> d, res = fnnls(Z,x) #run with no initial support
>>> end = time()
>>> end-start #time without initial support
0.0065343379974365234
>>> support = np.nonzero(d)[0] #find the support of the solution
>>> start = time()
>>> d_sup, res = fnnls(Z, x, P_initial = support) #run with initial support
>>> end = time()
>>> end-start #time with initial support
0.0012061595916748047
>>> np.array_equal(d,d_sup) #check the two solutions are equal
True
```

## Authors
* Joshua Vendrow
* Jamie Haddock
