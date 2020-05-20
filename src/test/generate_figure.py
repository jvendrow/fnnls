#!/usr/bin/env python
# coding: utf-8

import pytest

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import sparse
from time import time

from fnnls.fnnls import fnnls



class nnls_comparison():

    def __init__(self):

        return

    def test(self, repetitions, dimensions, optimizers, names, generator=np.random.rand, verbose=True):
        """
        Measures the speed of each optimizer at each dimensions,
        averaged over repetitions

        Parameters
        ----------
        repetitions: int
            The number of times to run at optimizer at each dimensions
            for precision. 

        dimensions: numpy array
            A list of dimensions of matrix A and vector x to use 
            for testing.

        optimizers: list
            A list of nonnegative least squares functions that
            we will test the complexity of.

        names: list
            A list corresponding to the optimizers list of what
            to call the optimizers for plotting

        """

        nnls_times = []
        nnls_res = []
        nnls_d = []

        for dim in dimensions:

            matrices = [generator(dim*10, dim) for _ in range(repetitions)]
            vectors = [generator(dim*10, 1).reshape((dim*10)) for _ in range(repetitions)]

            for index, optimizer in enumerate(optimizers):

                time_total = 0
                res_total = 0
                ds = []

                for i in range(repetitions):

                    #define matrix A and vector x
                    #------------------------------

                    Z = matrices[i]
                    x = vectors[i]

                    #Measure the speed of running the optimizer
                    start = time()

                    [d, res] = optimizer(Z, x)

                    end = time()
                
                    time_total += end - start
                    res_total += res
                    ds.append(d)

                #Print and store results
                #-----------------------
                if verbose:
                    print(dim)
                    print(names[index] + ": " + str(time_total))
                    print(names[index] + ": " + str(res_total))

                if dim == dimensions[0]:
                    nnls_times.append([time_total])
                    nnls_res.append([res_total])
                    nnls_d.append([ds])

                else:
                    nnls_times[index].append(time_total)
                    nnls_res[index].append(res_total)
                    nnls_d[index].append(ds)

            self.repetitions = repetitions
            self.dimensions = dimensions
            self.names = names
            self.nnls_times = nnls_times
            self.nnls_res = nnls_res
            self.nnls_d = nnls_d

    def plot_times(self):
        """
        Plots the time complexities of each optimizer

        """
        #Plot the times for nnls
        #-----------------------
        for nnls_time in self.nnls_times:
            plt.plot(self.dimensions, nnls_time)

        plt.xlabel("Dimension")
        plt.ylabel("time (s) for " +  str(self.repetitions) + " runs")
        plt.legend(self.names)

        plt.show()

    def plot_residuals(self):

        """
        Plots the residuals of each optimizer

        """
        #Plot the times for nnls
        #-----------------------
        for nnls_res in self.nnls_res:
            plt.plot(self.dimensions, [i / self.repetitions for i in nnls_res])

        plt.xlabel("Dimension")
        plt.ylabel("Average Residual")
        plt.legend(self.names)

        plt.show()

    def diff_residuals(self):
        """
        Calculates max difference between residuals of
        the first two optimization functions
        """

        differences = []
        for i in range(len(self.nnls_d[0])):

            residuals_1 = self.nnls_d[0][i]
            residuals_2 = self.nnls_d[1][i]

            avg_diff = 0
            for j in range(len(residuals_1)):
                avg_diff += np.linalg.norm(residuals_1[j] - residuals_2[j]) / np.linalg.norm(residuals_1[j])
            differences.append(avg_diff / len(residuals_1))
            
        return differences



sparsity = 0.01
SPARSE = False

#Declare generator, either sparse or dense
generator_sparse = lambda m, n: sparse.random(m, n, sparsity).toarray()
generator_gaussian = np.random.randn


#Declare parameters for testing
repetitions = 100
dimensions = np.arange(10, 411, 40)
optimizers = [optimize.nnls, fnnls]
names = ["scipy.optimize.nnls", "fnnls"]

#run comparison tests
testing_dense = nnls_comparison()
testing_dense.test(repetitions, dimensions, optimizers, names, generator=generator_gaussian, verbose=True)
#run comparison tests
testing_sparse = nnls_comparison()
testing_sparse.test(repetitions, dimensions, optimizers, names, generator=generator_sparse, verbose=True)

times_dense = testing_dense.nnls_times
times_sparse = testing_sparse.nnls_times

#Calcualte differences between residuals, should be VERY small
differences = testing_dense.diff_residuals()
print("Array of difference between solutions: {}".format(differences))
differences = testing_sparse.diff_residuals()
print("Array of difference between solutions: {}".format(differences))

plt.plot(dimensions, times_dense[0], 'C0--')
plt.plot(dimensions, times_dense[1], 'C0-')
plt.plot(dimensions, times_sparse[0], 'C1--')
plt.plot(dimensions, times_sparse[1], 'C1-')

plt.xlabel("Dimension")
plt.ylabel("Time (s) for " +  str(repetitions) + " runs")
plt.legend(["scipy.optimize.nnls (dense)", "fnnls (dense)", "scipy.optimize.nnls (sparse)", "fnnls (sparse)"])

plt.show()
