#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:26:56 2018

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

from math import isclose
from typing import Tuple

import numpy as np

from .gradient5point import gradient5point
from ..utils.diffSquareFit import gradientFitArea


def E2mu(E, nu):
    return E / (2.0 * (1.0 + nu))


def mu2E(mu, nu):
    return 2 * (1 + nu) * mu


def calculate_qxyz(mu, nu, us, vs, ws, usz, vsz, wsz, x_spacing, y_spacing=None, use5p=False):
    if y_spacing is None:
        y_spacing = x_spacing

    # All input arrays should be 2D (surface layer)

    gradFun = np.gradient
    if use5p:
        gradFun = gradient5point

    print(us.shape)
    usx, usy = gradFun(us, x_spacing, y_spacing)  # pylint: disable=unused-variable
    vsx, vsy = gradFun(vs, x_spacing, y_spacing)  # pylint: disable=unused-variable
    wsx, wsy = gradFun(ws, x_spacing, y_spacing)

    divU = usx + vsy + wsz

    qx = -mu * (usz + wsx)
    qy = -mu * (vsz + wsy)

    if isclose(nu, 0.5):
        # In the incompressible case we can make use of div(u) = 0
        # to avoid the divergent term nu/(1-2*nu)
        # Actually this should be checked
        qz = -2 * mu * wsz
    else:
        qz = -mu * (wsz + wsz + 2 * nu / (1 - 2 * nu) * divU)

    return qx, qy, qz  # p = qz


### Full field expressions ###


def get_F_Strain_square(ux, uy, uz, xSpacing, ySpacing, zSpacing) -> np.ndarray:
    # Returns compleat F Strain Tensor
    assert (ux.shape == uy.shape)
    assert (ux.shape == uz.shape)

    gradFun = gradientFitArea

    print("Shape of input form {}".format(ux.shape))

    uxx, uxy, uxz = gradFun(ux, xSpacing, ySpacing, zSpacing)
    uyx, uyy, uyz = gradFun(uy, xSpacing, ySpacing, zSpacing)
    uzx, uzy, uzz = gradFun(uz, xSpacing, ySpacing, zSpacing)

    return np.asanyarray([[uxx, uxy, uxz], [uyx, uyy, uyz], [uzx, uzy, uzz]])


def readoutSurface(ux, uy, uz, F):
    us = ux[:, :, 0]
    vs = uy[:, :, 0]
    ws = uz[:, :, 0]
    return us, vs, ws, F[0, 2, :, :, 0], F[1, 2, :, :, 0], F[2, 2, :, :, 0]


VoigtMatNdarrayTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def linearStress(mu: float, nu: float, F: np.ndarray) -> VoigtMatNdarrayTuple:
    # Returns all 6 Stress components in Voigt notation
    uxx, uxy, uxz = F[0]
    uyx, uyy, uyz = F[1]
    uzx, uzy, uzz = F[2]

    if isclose(nu, 0.5):
        # In the incompressible case we can make use of div(u) = 0
        # to avoide the divergent term nu/(1-2*nu)
        # Actually this should be checked
        sgxx = 2 * mu * uxx
        sgyy = 2 * mu * uyy
        sgzz = 2 * mu * uzz
    else:
        # Non incompressible case
        divU = uxx + uyy + uzz

        sgxx = 2 * mu * (uxx + nu / (1 - 2 * nu) * divU)
        sgyy = 2 * mu * (uyy + nu / (1 - 2 * nu) * divU)
        sgzz = 2 * mu * (uzz + nu / (1 - 2 * nu) * divU)

    sgyz = mu * (uyz + uzy)
    sgxz = mu * (uxz + uzx)
    sgxy = mu * (uxy + uyx)

    return sgxx, sgyy, sgzz, sgyz, sgxz, sgxy


def linearStrain(mu, nu, t):
    # Inverts linear Stress calculation to get linear Strain.
    # Notice that nonlinear components are doubled in Voigt notation
    sgxx, sgyy, sgzz, sgyz, sgxz, sgxy = t

    trsg = sgxx + sgyy + sgzz

    epxx = (sgxx - nu / (1 + nu) * trsg) / (2 * mu)
    epyy = (sgyy - nu / (1 + nu) * trsg) / (2 * mu)
    epzz = (sgzz - nu / (1 + nu) * trsg) / (2 * mu)

    epyz2 = sgyz / mu
    epxz2 = sgxz / mu
    epxy2 = sgxy / mu

    return epxx, epyy, epzz, epyz2, epxz2, epxy2
