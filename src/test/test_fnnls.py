import pytest
import numpy as np

from fnnls.fnnls import fnnls

def test():
    """
    Run basic test of the fnnls implementation on 
    random gaussian data. 

    Ensures that the solution is within a small 
    epsilon of the true solution
    """
    epsilon = 0.00001
    np.random.seed(1)

    Z = np.abs(np.random.rand(5,10))
    x = np.abs(np.random.rand(5))

    d, res = fnnls(Z,x)

    expected_d = np.asarray([0, 0, 0, 0.49507457, 0, 0, 0.10518829, 0, 0, 0])

    assert(np.max(np.abs(d - expected_d)) < epsilon)





