# Based on P. C. Hansen, Regularization Tools Version 4.0 for Matlab 7.3 files gcv.m and gcvfun.m:

# Calculate Plot the GCV function and find its minimum.
#
# [reg_min,G,reg_param] = gcv(U,s,b,method)
#
# Calculates the GCV-function
#          || A*x - b ||^2
#    G = -------------------
#        (trace(I - A*A_I)^2
# as a function of the regularization parameter reg_param.
# Here, A_I is a matrix which produces the regularized solution.
# and x is the solution calculated with Tikhonov regularization
# using the regularization parameter reg_param.
#
# If any output arguments are specified, then the minimum of G is
# identified and the corresponding reg. parameter reg_min is returned.

# Per Christian Hansen, IMM, Dec. 16, 2003.

# Reference: G. Wahba, "Spline Models for Observational Data",
# SIAM, 1990.

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def gcv(U, s, b, lambdarange, plot: bool):
    """
    Calculates a generalized cross correlation:

    Input:
        U, s - Singular value Decomposition U @ diag(s) @ V.A of A (V ommitted)
        b - b vector (see above)
        lambdarange - Vector Regularization parameters for which the GCV function is evaluated
        plot - Plot resulting curve using matplotlib

    Returns:
        reg_min - Regularization parameter correspondig to GCV-function's minimum
        minG - Mimimal value of the GCV-function
        G - Result vector containing the value of the gcv function at the sampling points
        reg_param - range of sampling points (identical to lambdarange)
    """
    npoints = lambdarange.size
    # smin_ratio = 16 * np.spacing(1)  # Smallest regularization parameter.

    m, n = U.shape
    p = s.size
    UT = U.T.conjugate()
    beta = UT.dot(b)
    beta2 = np.linalg.norm(b) ** 2 - np.linalg.norm(beta) ** 2

    # Vector of regularization parameters.
    reg_param = np.zeros(npoints)
    G = np.copy(reg_param)  # very important to copy here!!!
    s2 = s ** 2
    # reg_param[-1] = np.max([s[-1], s[0]*smin_ratio])
    # ratio = (s[0] / reg_param[-1])**(1.0 / (npoints-1))
    # for i in range(npoints-2, -1, -1):
    #    reg_param[i] = ratio * reg_param[i+1]
    reg_param = np.copy(lambdarange)

    # Intrinsic residual.
    delta0 = 0
    if m > n and beta2 > 0:
        delta0 = beta2

    # Vector of GCV-function values.
    for i in range(npoints):
        G[i] = gcvfun(reg_param[i], s2, beta[:p], delta0, m - n)

    minG, minGi = G.min(0), G.argmin(0)  # Initial guess.
    reg_min = optimize.fmin(
        gcvfun,
        x0=reg_param[np.max([minGi, 0])],
        args=(s2, beta[:p], delta0, m - n),
        disp=0,
    )[0]
    minG = gcvfun(reg_min, s2, beta[:p], delta0, m - n)  # Minimum of GCV function.

    if plot:
        # Plot GCV function.
        plt.plot(reg_param, G, "-")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$G(\lambda)$")
        plt.plot([reg_min], [minG], "*r")
        plt.plot([reg_min, reg_min], [minG / 1000, minG], ":r")
        plt.title(r"GCV function, minimum at $\lambda = %.2e$" % reg_min)
        plt.show()
    return reg_min, minG, G, reg_param


def gcvfun(lmbda, *args):  # s2, beta, delta0, mn):
    # Auxiliary routine for gcv.  PCH, IMM, Feb. 24, 2008.
    # Note: f = 1 - filter-factors.
    s2, beta, delta0, mn = args
    f = (lmbda ** 2) / (s2 + lmbda ** 2)
    G = (np.linalg.norm(f * beta) ** 2 + delta0) / (mn + np.sum(f)) ** 2
    return G
