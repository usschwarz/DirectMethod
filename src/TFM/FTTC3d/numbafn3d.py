""" Some helper functions implemented in numba """

import numpy as np
import numba

DTYPEi = np.int32
DTYPEi_t = numba.int32

DTYPEf = np.float64
DTYPEf_t = numba.float64

DTYPEc = np.complex128
DTYPEc_t = numba.complex128


@numba.njit
def calculate_2x2_inv(U):
    U_inv = np.empty((2, 2), dtype=DTYPEc)
    detU = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    invdetU = 1. / detU
    U_inv[0, 0] = invdetU * U[1, 1]
    U_inv[0, 1] = - invdetU * U[0, 1]
    U_inv[1, 0] = - invdetU * U[1, 0]
    U_inv[1, 1] = invdetU * U[0, 0]
    return U_inv


@numba.njit
def calculate_3x3_inv(U):
    U_inv = np.zeros((3, 3), dtype=DTYPEc)
    detU = U[0, 0] * (U[1, 1] * U[2, 2] - U[2, 1] * U[1, 2]) - \
           U[0, 1] * (U[1, 0] * U[2, 2] - U[2, 0] * U[1, 2]) + \
           U[0, 2] * (U[1, 0] * U[2, 1] - U[2, 0] * U[1, 1])
    # print  U[0,0], U[1,1], U[2,2], U[0,0] * U[1,1] * U[2,2], U[1,1] * U[2,2] - U[2,1] * U[1,2], detU, np.linalg.det(U)
    invdetU = 1. / detU
    U_inv[0, 0] = invdetU * (U[1, 1] * U[2, 2] - U[1, 2] * U[2, 1])
    U_inv[0, 1] = invdetU * (U[0, 2] * U[2, 1] - U[0, 1] * U[2, 2])
    U_inv[0, 2] = invdetU * (U[0, 1] * U[1, 2] - U[0, 2] * U[1, 1])
    U_inv[1, 0] = invdetU * (U[1, 2] * U[2, 0] - U[1, 0] * U[2, 2])
    U_inv[1, 1] = invdetU * (U[0, 0] * U[2, 2] - U[0, 2] * U[2, 0])
    U_inv[1, 2] = invdetU * (U[0, 2] * U[1, 0] - U[0, 0] * U[1, 2])
    U_inv[2, 0] = invdetU * (U[1, 0] * U[2, 1] - U[1, 1] * U[2, 0])
    U_inv[2, 1] = invdetU * (U[0, 1] * U[2, 0] - U[0, 0] * U[2, 1])
    U_inv[2, 2] = invdetU * (U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0])
    return U_inv


@numba.njit
def calculate_traction_2d(FtGmn, L):
    M = len(FtGmn[0, 0])
    N = len(FtGmn[0, 0, 0])

    FtGmnInv = np.empty((2, 2, M, N), dtype=DTYPEc)
    Tikh = np.zeros((2, 2), dtype=DTYPEc)
    Tikh[0, 0] = L
    Tikh[1, 1] = L

    GG = np.empty((2, 2), dtype=DTYPEc)  # Added to satisfy numba
    for i in range(M):
        for j in range(N):
            # GG = FtGmn[:, :, i, j]
            GG[0, 0] = FtGmn[0, 0, i, j]
            GG[0, 1] = FtGmn[0, 1, i, j]
            GG[1, 0] = FtGmn[1, 0, i, j]
            GG[1, 1] = FtGmn[1, 1, i, j]

            FtGmnInv[:, :, i, j] = np.dot(calculate_2x2_inv(np.dot(GG.T, GG) + Tikh), GG.T)
    return FtGmnInv


@numba.njit
def calculate_traction_3d(FtGmn, L):
    M = len(FtGmn[0, 0])
    N = len(FtGmn[0, 0, 0])

    FtGmnInv = np.empty((3, 3, M, N), dtype=DTYPEc)
    Tikh = np.zeros((3, 3), dtype=DTYPEc)
    Tikh[0, 0] = L
    Tikh[1, 1] = L
    Tikh[2, 2] = L

    GG = np.empty((3, 3), dtype=DTYPEc)  # Added to satisfy numba
    for i in range(M):
        for j in range(N):
            # We need to make an extra copy out
            for k in range(3):
                for l in range(3):
                    GG[k, l] = FtGmn[k, l, i, j]
            FtGmnInv[:, :, i, j] = np.dot(calculate_3x3_inv(np.dot(GG.T, GG) + Tikh), GG.T)
    return FtGmnInv
