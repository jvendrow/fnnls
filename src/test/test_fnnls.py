import pytest
import numpy as np

from fnnls.fnnls import fnnls


def test_basic():
    """
    Run a test of the fnnls implementation on
    the example from the Fast Nonnegative Least
    Squares paper by Bro and Jung
    """

    epsilon = 0.00001

    Z = np.asarray([
                    [73, 71, 52],
                    [87, 74, 46],
                    [72,  2,  7],
                    [80, 89, 71]
                    ])
    x = np.asarray([49, 67, 68, 20])

    d, res = fnnls(Z, x)

    expected_d = ([0.64953844,0,0]) 

    assert(np.max(np.abs(d - expected_d)) < epsilon)

def test_gaussian():
    """
    Run a test of the fnnls implementation on 
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





