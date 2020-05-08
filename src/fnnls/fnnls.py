import numpy as np

def fnnls(Z, x, P = set()):
    """
    Implementation of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong

    This algorithm seeks to find min_d ||x - Zd|| subject to d >= 0

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al. 

    Parameters
    ----------
    Z: NumPy array
        Z is an m x n matrix

    x: Numpy array
        x is a m x 1 vector

    P: set of integers
        Empty set by default, or an estimate for set P of the
        algorithm based on the expected support of the solution 
        
    Returns
    -------
    d: Numpy array
        d is a nx1 vector
    """

    Z, x = map(np.asarray_chkfinite, (Z, x))

    if len(Z.shape) != 2:
        raise ValueError("Expected a two-dimensional array, but Z is of shape {}".format(Z.shape))
    if len(x.shape) != 1:
        raise ValueError("Expected a one-dimensional array, but x is of shape {}".format(x.shape))

    m, n = Z.shape

    if x.shape[0] != m:
        raise ValueError("Incompatable dimensions. The first dimension of Z should match the length of x, but Z is of shape {} and x is of shape {}".format(Z.shape, x.shape))


    #Declaring constants for tolerance and max repetitions
    epsilon = 2.2204e-16
    tolerance = epsilon * np.linalg.norm(ZTZ, ord=1) * n

    max_repetitions = 5

    #Calculating ZTZ and ZTx in advance to improve the efficiency of calculations
    ZTZ = Z.T.dot(Z)
    ZTx = Z.T.dot(x)

    #A1 --> initialized as parameter
    #A2
    R = {i for i in range(0,n) if i not in P}

    R_ind = list(R)
    #A3
    d = np.zeros(n)
    #A4
    w = ZTx - (ZTZ) @ d

    s = np.zeros(n)

    epsilon = 2.2204e-16
    tolerance = epsilon * np.linalg.norm(ZTZ, ord=1) * n

    max_repetitions = 5

    #B1
    no_update = 0

    while len(R) and np.max(w[R_ind]) > tolerance:

        current_passive = P.copy() #make copy of passive set to check for change at end of loop

        #B2 
        ind = R_ind[np.argmax(w[R_ind])]

        #B3
        P.add(ind)
        R.remove(ind)

        P_ind = list(P)
        R_ind = list(R)

        #B4
        s[P_ind] = np.linalg.lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], rcond=None)[0]

        #C1
        while len(P) and np.min(s[P_ind]) <= tolerance:
            times.append(time())
            #C2
            q = [a for a in P_ind if s[a] <= tolerance]
            alpha = np.min(d[q] / (d[q] - s[q]))

            times.append(time())
            #C3
            d = d + alpha * (s-d) #set d as close to s as possible while maintaining non-negativity
            #C4
            passive = {p for p in P_ind if s[p] <= tolerance}


            P.difference_update(passive)
            R.update(passive)
            P_ind = list(P)
            R_ind = list(R)


            #C5
            s[P_ind] = np.linalg.lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], rcond=None)[0]

            #C6
            s[R_ind] = np.zeros(len(R))

        #B5
        d = s.copy() 
        w = ZTx - (ZTZ) @ d

        if(current_passive == P): #check of there has been a check to the passive set
            no_update += 1
        else:
            no_update = 0

        if no_update > max_repetitions:
            break

    res = np.linalg.norm(x - Z@d) #Calculate residual loss ||x - Zd||
    
    return [d, res]
