import numpy as np

def fnnls(Z, x, P_initial = np.zeros(0, dtype=int), lstsq = lambda A, x: np.linalg.inv(A).dot(x)):
    """
    Implementation of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong.

    This algorithm seeks to find min_d ||x - Zd|| subject to d >= 0

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al. 

    Parameters
    ----------
    Z: NumPy array
        Z is an m x n matrix.

    x: Numpy array
        x is a m x 1 vector.

    P_initial: Numpy array, dtype=int
        By default, an empty array. An estimate for
        the indices of the support of the solution.

        lstsq: function
        By default, numpy.linalg.lstsq with rcond=None.
        Least squares function to use when calculating the
        least squares solution min_x ||Ax - b||. 
        Must be of the form x = f(A,b).
        
    Returns
    -------
    d: Numpy array
        d is a nx1 vector
    """

    # map Z, x, and P_initial to np arrays to standardize from any input 
    Z, x, P_initial = map(np.asarray_chkfinite, (Z, x, P_initial))

    m, n = Z.shape

    if len(Z.shape) != 2:
        raise ValueError("Expected a two-dimensional array, but Z is of shape {}".format(Z.shape))
    if len(x.shape) != 1:
        raise ValueError("Expected a one-dimensional array, but x is of shape {}".format(x.shape))
    if len(P_initial.shape) != 1:
        raise ValueError("Expected a one-dimensional array, but P_initial is of shape {}".format(P_initial.shape))

    if not np.all((P_initial - P_initial.astype(int)) == 0):
        raise ValueError("Expected only integer values, but P_initial has values {}".format(P_initial[(P_initial - P_initial.astype(int)) != 0]))
    if np.any(P_initial >= n):
        raise ValueError("Expected values between 0 and Z.shape[1], but P_initial has max value {}".format(np.max(P_initial)))
    if np.any(P_initial < 0):
        raise ValueError("Expected values between 0 and Z.shape[1], but P_initial has min value {}".format(np.min(P_initial)))
    if P_initial.dtype != np.dtype('int64') and P_initial.dtype != np.dtype('int32') :
        raise TypeError("Expected type int64 or int32, but P_initial is type {}".format(P_initial.dtype))
    if x.shape[0] != m:
        raise ValueError("Incompatable dimensions. The first dimension of Z should match the length of x, but Z is of shape {} and x is of shape {}".format(Z.shape, x.shape))

    # Calculating ZTZ and ZTx in advance to improve the efficiency of calculations
    ZTZ = Z.T.dot(Z)
    ZTx = Z.T.dot(x)

    # Declaring constants for tolerance and max repetitions
    epsilon = 2.2204e-16
    tolerance = epsilon * n

    # Number of contsecutive times the set P can remain unchanged loop until we terminate
    max_repetitions = 5

    # A1 + A2
    # P is a boolean array that represents the passive set
    P = np.zeros(n, dtype=np.bool)
    P[P_initial] = True

    # A3
    # Initialize d to zero vector
    d = np.zeros(n)

    # A4
    # Set w = Z^T*x - (Z^T*Z)d
    # This is the most important distinction from standard nnls,
    # in which w = Z^T(x - Zd)
    w = ZTx - (ZTZ) @ d

    # Initialize s
    s = np.zeros(n)

    # Count of amount of consecutive times set P has remained unchanged
    no_update = 0

    # Extra loop in case a support is set to update s and d
    if P_initial.shape[0] != 0:

        s[P] = lstsq((ZTZ)[P][:,P], (ZTx)[P])
        d = s.clip(min=0)

    # B1
    while (not np.all(P))  and np.max(w[~P]) > tolerance:
        
        current_P = P.copy() # Make copy of passive set to check for change at end of loop

        # B2 + B3 
        # Move the element in active set with largest value
        # of w into the passive set
        P[np.argmax(w * ~P)] = True

        # B4
        # Set s to the least squares solution along the passive set
        s[P] = lstsq((ZTZ)[P][:,P], (ZTx)[P])

        # C1
        # We loop until either the passive set is empty or every
        # element in s in the passive set is above the tolerance
        while np.any(P) and np.min(s[P]) <= tolerance:

            s, d, P = fix_constraint(ZTZ, ZTx, s, d, P, tolerance, lstsq)

        # B5
        d = s.copy() 
        # B6
        w = ZTx - (ZTZ) @ d

        # Check if there has been a change to the passive set
        if(np.all(current_P == P)): 
            no_update += 1
        else:
            no_update = 0

        if no_update >= max_repetitions:
            break

    res = np.linalg.norm(x - Z@d)  #Calculate residual loss ||x - Zd||

    return [d, res]


def fix_constraint(ZTZ, ZTx, s, d, P, tolerance, lstsq = lambda A, x: np.linalg.inv(A).dot(x)):
    """
    The inner loop of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong.

    One iteration of the loop to adjust the new estimate s to satisfy the
    nonnegativity contraint of the solution.

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al. 

    Parameters
    ----------
    ZTZ: NumPy array
        ZTZ is an n x n matrix equal to Z.T * Z

    ZTx: Numpy array
        ZTx is an n x 1 vector equal to Z.T * x

    s: Numpy array
        The new estimate of the solution with possible
        negative values that do not meet the constraint

    d: Numpy array
        The previous estimate of the solution that satisfies
        the nonnegativity contraint

    P: Numpy array, dtype=np.bool
        The current passive set, which comtains the indices
        that are not fixed at the value zero. 

    tolerance: float
        A tolerance, below which values are considered to be
        0, allowing for more reasonable convergence.

    lstsq: function
        By default, numpy.linalg.lstsq with rcond=None.
        Least squares function to use when calculating the
        least squares solution min_x ||Ax - b||. 
        Must be of the form x = f(A,b).
        
    Returns
    -------
    s: Numpy array
        The updated new estimate of the solution.
    d: Numpy array
        The updated previous estimate, now as close
        as possible to s while maintaining nonnegativity.
    P: Numpy array, dtype=np.bool
        The updated passive set
        """ 

    # C2
    # find largest alpha such that d + alpha(s-d) 
    # is close to s but non-negative
    q = P * (s <= tolerance)
    alpha = np.min(d[q] / (d[q] - s[q]))

    # C3
    # Set d as close to s as possible while maintaining non-negativity
    d = d + alpha * (s-d)

    # C4
    # Move elements with d less than tolerance to active set
    P[d <= tolerance] = False

    # C5
    # Set s to the least squares solution along the passive set
    s[P] = lstsq((ZTZ)[P][:,P], (ZTx)[P])

    # C6
    # Set values of s in active set to 0.
    s[~P] = 0.

    return s, d, P

def RK(A,b,k=100, random_state=None):
    """
    Function that runs k iterations of randomized Kaczmarz iterations (with uniform sampling).

    Parameters
    ----------
    A : NumPy array
        The measurement matrix (size m x n).
    b : NumPy array
        The measurement vector (size m x 1).
    k : int_, optional
        Number of iterations (default is 100).
    random_state: int, optional
        Random state for NumPy random sampling 
        
    Returns
    -------
    x : NumPy array
        The approximate solution 
    """ 
    
    if random_state != None:
        np.random.seed(random_state)

    m, n = np.shape(A)
    x = np.zeros([n])

    # At each iteration, we examine the indices at a subset of the indices,
    # and update x accordingly
    for i in range(k):

        ind = np.random.choice(range(m))
        # Here we calculate the randomized kaczmarz step:
        #     x = x + A^T * (b - Ax) / || A ||^2
        # where A is replaced with the a single row of A, A[ind].
        x = x + np.transpose(A[ind,:])*(b[ind] - A[ind,:] @ x)/(np.linalg.norm(A[ind,:])**2)

    return x

def RGS(A,b,k=100, random_state=None):
    """
    Function that runs k iterations of randomized Gauss-Seidel iterations (with uniform sampling).

    Parameters
    ----------
    A : NumPy array
        The measurement matrix (size m x n).
    b : NumPy array
        The measurement vector (size m x 1).
    k : int_, optional
        Number of iterations (default is 100).
    random_state: int, optional
        Random state for NumPy random sampling
        
    Returns
    -------
    x : NumPy array
        The approximate solution 
    """ 

    if random_state != None:
        np.random.seed(random_state)

    m, n = np.shape(A)
    x = np.zeros([n])

    # At each iteration, we examine the indices at a subset of the indices,
    # and update x accordingly
    for i in range(k):

        ind = np.random.choice(range(n))
        # Here we calculate the randomized gauss-seidel step:
        #     x_ind = x_ind + A_ind^T * (b - Ax) / || A ||^2
        # Where A_ind is the ind row of A
        x[ind] = x[ind] + (np.transpose(A[:,ind]) @ (b - A @ x)/(np.linalg.norm(A[:,ind])**2))

    return x
