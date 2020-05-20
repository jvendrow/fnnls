
---
title: 'fnnls: An implementation of Fast Nonnegative Least Squares'
tags:
  - Nonnegative least square
  - Optimization
  - Machine Learning
  - Python
authors:
  - name: Joshua Vendrow
    orcid: 0000-0002-1041-5782
    affiliation: 1
  - name: Jamie Haddock
    orcid: 0000-0002-1449-2574
    affiliation: 1
affiliations:
 - name: Department of Mathematics, University of California, Los Angeles
   index: 1
date: 20 May 2020
bibliography: paper.bib

---


# Summary

`fnnls` is a Python package that offers a fast algorithm for solving the nonnegative least squares problem. The Fast Nonnegative Least Squares (fnnls) algorithm was first presented in the paper "A fast non‐negativity‐constrained least squares algorithm" [@bro1997fast]. This algorithm exploits speed-up opportunities and improves upon the nonnegative least squares active-set algorithm of [@lawson1995solving].

Given a matrix $\mathbf{Z} \in \mathbb{R}^{m \times n}$ and a vector $\mathbf{x} \in \mathbb{R}^{n}$ the goal of nonnegative least squares is to find
$$\min_{\mathbf{d} \in \mathbb{R}^m} ||\mathbf{x} - \mathbf{Zd}|| \textrm{ subject to } \mathbf{d} \ge 0.$$

The fnnls algorithm improves upon the computational effort of [@lawson1995solving] by precomputing the values of $\mathbf{Z^{\top}Z}$ and  $\mathbf{Z^{\top}x}$ and using them throughout the algorithm. If we assume that $\textbf{Z}$ is tall $(m \gg n)$ then this precomputation significantly decreases the computing time by replacing matrix and vector multiplications with computations of a smaller dimension. 

The nonnegative least squares problem has many applications in the field of applied math and specifically as a subproblem for various matrix and tensor factorization algorithms, including nonnegative matrix factorization (NMF) [@lee2001algorithms] and nonnegative tensor decomposition (NTD) [@bro1997parafac].

`fnnls` is currently in use in ongoing research on NMF and NTD methods. Efficient solution of the nonnegative least squares problem is the backbone of the new algorithms for computing hierarchical  NMF and NTD models via a forward propagation and backpropagation procedure [@GHMNSWZ19; @WHZMNGS20Neural; @VHN20Neural].

# Comparison to Other Algorithms

The standard implementation for nonnegative least squares is the `scipy.optimize.nnls` function within the SciPy open-source Python library [@2020SciPy-NMeth]. SciPy uses an implementation of the Lawson and Hanson algorithm, which was first presented in 1974 [@lawson1995solving]. 

Below, we test the efficiency and accuracy of `fnnls` against the SciPy `scipy.optimize.nnls` function. We measure the time taken by each method over 100 runs each on random Gaussian and sparse uniform data generated separately for each run. In each run, we randomly generate $\mathbf{Z}$ and $\mathbf{x}$ and record the time spent by `fnnls` and `scipy.optimize.nnls`. For runs on Gaussian data, $\mathbf{Z}$ and $\mathbf{x}$ are generated with the `numpy.random.rand` function which draws each entry i.i.d. from the standard normal distribution, taking the absolute value of the outcome [@oliphant2006guide]. For the runs on sparse uniform data, $\mathbf{Z}$ and $\mathbf{x}$ are generated with the `scipy.sparse.random` function which constructs a matrix with density $0.1$, where the nonzero entries are drawn i.i.d. from the uniform distribution $[0,1)$. We graph the time consumption versus the size of the matrices. Here, dimension $n$ indicates that we generate $\mathbf{Z} \in \mathbb{R}^{10n \times n}$  and $\mathbf{x} \in \mathbb{R}^{10n}$,  yielding a tall matrix. 

![image](https://github.com/jvendrow/fnnls/blob/master/paper/nnls_comparison.png?raw=true)
**Fig. 1:**  The total running time over 100 runs of each algorithm on random Gaussian data (dense) and random uniform sparse data (sparse) of varying dimension. `fnnls` required significantly less time at higher dimensions for both types of matrices. 

Let $\mathbf{d_f}$ and $\mathbf{d_n}$ be the solution vectors produced by `fnnls` and the SciPy `nnls` function, respectively. Then we compute the relative error between the solutions as 
$$\dfrac{||\mathbf{d_f}-\mathbf{d_n}||_2} {||\mathbf{d_n}||_2}.$$
The average relative error between $\mathbf{d_f}$ and $\mathbf{d_n}$ across the 100 runs did not exceed $10^{-12}$ for any dimension of the Gaussian data, and did not exceed $10^{-14}$ for any dimension of the sparse uniform data. 

# References
