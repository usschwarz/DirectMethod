""" Some helper functions implemented in numba """

import numpy as np
import numba

complex_t = np.complex128


@numba.njit
def calculate_2x2_inv(U):
    """ Calculates inverse of 2x2 matrix """
    U_inv = np.empty((2, 2), dtype=complex_t)
    detU = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    invdetU = 1.0 / detU
    U_inv[0, 0] = invdetU * U[1, 1]
    U_inv[0, 1] = -invdetU * U[0, 1]
    U_inv[1, 0] = -invdetU * U[1, 0]
    U_inv[1, 1] = invdetU * U[0, 0]
    return U_inv


@numba.njit
def calculate_traction_2d(FtGmn, L):
    """ Calculates Tikhonov regularized inverse of FTGmn """
    M = len(FtGmn[0, 0])
    N = len(FtGmn[0, 0, 0])

    FtGmnInv = np.empty((2, 2, M, N), dtype=complex_t)
    Tikh = np.zeros((2, 2), dtype=complex_t)
    Tikh[0, 0] = L
    Tikh[1, 1] = L

    GG = np.empty((2, 2), dtype=complex_t)  # Added to satisfy numba
    for i in range(M):
        for j in range(N):
            # GG = FtGmn[:, :, i, j]
            GG[0, 0] = FtGmn[0, 0, i, j]
            GG[0, 1] = FtGmn[0, 1, i, j]
            GG[1, 0] = FtGmn[1, 0, i, j]
            GG[1, 1] = FtGmn[1, 1, i, j]

            FtGmnInv[:, :, i, j] = np.dot(
                calculate_2x2_inv(np.dot(GG.T, GG) + Tikh), GG.T
            )
    return FtGmnInv
