#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import nnls
from fnnls import fnnls
from time import time



class nnls_testing():

    def __init__(self):

        return

    def test(self, repetitions, dimensions, optimizers, names, verbose=True):
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

        for d in dimensions:


            for index, optimizer in enumerate(optimizers):

                time_total = 0
                res_total = 0

                for i in range(repetitions):

                    #define matrix A and vector x
                    #------------------------------

                    A = np.abs(np.random.rand(d*10, d)) * 100

                    x = np.abs(np.random.rand(d*10)) * 100

                    #Measure the speed of running the optimizer
                    start = time()

                    [s, res] = optimizer(A, x)

                    end = time()
                
                    time_total += end - start
                    res_total += res

                #Print and store results
                #-----------------------
                if verbose:
                    print(d)
                    print(names[index] + ": " + str(time_total))
                    print(names[index] + ": " + str(res_total))

                if d == dimensions[0]:
                    nnls_times.append([time_total])
                    nnls_res.append([res_total])

                else:
                    nnls_times[index].append(time_total)
                    nnls_res[index].append(res_total)

            self.repetitions = repetitions
            self.dimensions = dimensions
            self.names = names
            self.nnls_times = nnls_times
            self.nnls_res = nnls_res

    def plot_times(self):
        """
        Plots the time complexities of each optimizer

        """
        #Plot the times for nnls
        #-----------------------
        for nnls_time in self.nnls_times:
            plt.plot(self.dimensions, nnls_time)

        plt.xlabel("dimension")
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

        plt.xlabel("dimension")
        plt.ylabel("Average Residual")
        plt.legend(self.names)

        plt.show()



testing = nnls_testing()

repetitions = 25
dimensions = np.arange(10, 350, 40)
optimizers = [nnls, fnnls]
names = ["scipy.optimize.nnls", "fnnls"]

testing.test(repetitions, dimensions, optimizers, names, verbose=True)

testing.plot_times()
testing.plot_residuals()
