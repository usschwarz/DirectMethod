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

from typing import Tuple

from numba import njit

def gcv_blockdiag(
    U: np.ndarray, s: np.ndarray, b: np.ndarray, lambdarange: np.ndarray, plot: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    npoints = lambdarange.size

    p = s.size
    beta = blkmul_adj(U, b)

    # Vector of regularization parameters.
    reg_param = np.zeros(npoints)
    G = np.copy(reg_param)  # very important to copy here!!!
    s2 = s ** 2
    reg_param = np.copy(lambdarange)

    # Vector of GCV-function values.
    for i in range(npoints):
        G[i] = gcvfun(reg_param[i], s2, beta[:p], 0., 0)  # , delta0, m - n)

    minGi = G.argmin(0)  # Initial guess.
    reg_min = optimize.fmin(
        gcvfun,
        x0=reg_param[np.max([minGi, 0])],
        args=(s2, beta[:p], 0., 0),  # delta0, m - n),
        disp=0,
    )[0]
    minG = gcvfun(reg_min, s2, beta[:p], 0., 0)  # delta0, m - n)  # Minimum of GCV function.

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
    return float(reg_min), float(minG), G, reg_param


def gcvfun(lmbda, s2, beta, delta0, mn):
    # Auxiliary routine for gcv.  PCH, IMM, Feb. 24, 2008.
    # Note: f = 1 - filter-factors.

    f = (lmbda ** 2) / (s2 + lmbda ** 2)
    G = (np.linalg.norm(f * beta) ** 2 + delta0) / (mn + np.sum(f)) ** 2
    return G


@njit(cache=True)
def blkmul_adj(mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ Calculate (mat.H) @ v """
    a, b, c = mat.shape
    assert ((a * b,) == v.shape)
    assert (a >= 1)
    MT0 = mat[0].T.conjugate()
    out0 = MT0 @ v[:c]
    out = np.empty(a * c, dtype=out0.dtype)
    out[:c] = out0
    for i in range(1, a):
        MT = mat[i].T.conjugate()
        out[i * c:i * c + c] = MT @ v[i * b:i * b + b]
    return out
