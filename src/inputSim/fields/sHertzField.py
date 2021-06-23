#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:55:23 2019

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)

This file calculates the displacement fields set up by an tangential Hertz profile
in y direction centered in the coordinate origin.

Parameters explained:
    x,y,z   Point observed
    a       Size of traction area
    Qy      Total traction force applied
    mu      Shear modulus
    nu      Poisson's ratio

"""

# Problem is mapped to the case of a 
# tangential Hertz-indentor in x direction.
# See tHertzField for details about the solution

import numpy as np

from numba import vectorize, float64

from tHertzField import get_u1_THertz, get_u2_THertz, get_u3_THertz
from tHertzField import get_u1_TSurf, get_u2_TSurf, get_u3_TSurf


# get_u1_THertz(x,y,z,a, Qx, mu, nu)
# get_u1_TSurf(x,y, a, Qx, mu, nu)

# define y diagonal functions

# y-directed tractions corresond to x directed ones
# in a rotated coordinate system
# To do so we need to rotate the coordinate system:
# x-> x' = y
# y-> y' = -x


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u1_SHertz(x, y, z, a, Qx, mu, nu):
    return - get_u2_THertz(y, -x, z, a, Qx, mu, nu)


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u2_SHertz(x, y, z, a, Qy, mu, nu):
    return get_u1_THertz(y, -x, z, a, Qy, mu, nu)


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u3_SHertz(x, y, z, a, Qz, mu, nu):
    return get_u3_THertz(y, -x, z, a, Qz, mu, nu)


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u1_SSurf(x, y, a, Qx, mu, nu):
    return - get_u2_TSurf(-y, x, a, Qx, mu, nu)


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u2_SSurf(x, y, a, Qy, mu, nu):
    return get_u1_TSurf(-y, x, a, Qy, mu, nu)


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u3_SSurf(x, y, a, Qz, mu, nu):
    return get_u3_TSurf(-y, x, a, Qz, mu, nu)


# Unlink symbols, that shouldn't be here
del get_u1_THertz, get_u2_THertz, get_u3_THertz
del get_u1_TSurf, get_u2_TSurf, get_u3_TSurf


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qx_SSurf(_x, _y, _a, _Qy):
    return 0


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qy_SSurf(x, y, a, Qy):
    rSq = x * x + y * y
    if rSq < a * a:
        return (3 * Qy / (2 * np.pi * a ** 3)) * np.sqrt(a * a - rSq)
    else:
        return 0


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qz_SSurf(_x, _y, _a, _Qy):
    return 0


if __name__ == "__main__":

    # Code to test integrity of the solution

    spacing = 1.
    mcVal = 200.5
    mcValZ = 10

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-mcVal, mcVal, spacing),
                          np.arange(-mcVal, mcVal, spacing),
                          np.arange(0, 2 * mcValZ, spacing))

    # Make the direction data for the arrows
    b = 120  # um    Dipole Distance
    a = 40  # um    Peak area radius
    Qx = 4  # Pa*um²

    nu = 0.499
    E = 20000.0  # Pa
    mu = E / (2.0 * (1.0 + nu))

    TH_Params = (x, y, z, a, Qx, mu, nu)

    TS_Params = (x[..., -1], y[..., -1], a, Qx, mu, nu)

    TD_Params = (x[:, :, (0, -10)], y[:, :, (0, -10)], z[:, :, (0, -10)], a)

    ua = get_u1_SHertz(*TH_Params)
    va = get_u2_SHertz(*TH_Params)
    wa = get_u3_SHertz(*TH_Params)

    us = get_u1_SSurf(*TS_Params)
    vs = get_u2_SSurf(*TS_Params)
    ws = get_u3_SSurf(*TS_Params)


    def hDataInfo(name, dx, dy, dz):
        print("{} data info".format(name))
        print(" type and dimensions:", type(dx[0, 0, 0]), dx.shape)
        print(" no of datapoints", np.prod(dx.shape), np.prod(dy.shape), np.prod(dz.shape))
        print(" no of Nan-points", len(dx[dx == np.nan]), len(dy[dy == np.nan]), len(dz[dz == np.nan]))
        print(" Peak Values:", np.nanmax(dx), np.nanmax(dy), np.nanmax(dz))


    hDataInfo("u", ua, va, wa)


    def hGetNthLargest(n, dArr):
        np.partition(dArr.flatten(), -n)[-n]


    def hCutPeaks(n, dx, dy, dz):
        # Calculate Norm of a vector (Squared)TD_Params
        drSq = dx * dx + dy * dy + dz * dz
        # (Because Sqrt is strictly monotomos we can skip it here)
        # Find element where to cutTzz = 1/4*((2*a**2-rSq +2*z**2)*arg2+abs1*(a*np.cos(tArg)-3*z*np.sin(tArg)))

        pivot = np.partition(drSq.flatten(), -n)[-n]
        print(pivot)[(0, -1), :]

        # We cut away the too long arrows to reveal the normal structur          
        dx[pivot <= drSq] = 0
        dy[pivot <= drSq] = 0
        dz[pivot <= drSq] = 0

        return dx, dy, dz


    # Now plot
    import matplotlib.pyplot as plt

    def plotLayer(imgLayer, title, cBarLabel=None, saveFig=True):
        xLen = imgLayer.shape[0] * spacing
        yLen = imgLayer.shape[1] * spacing

        extent = -xLen / 2, xLen / 2, -yLen / 2, yLen / 2
        plt.title(title)
        plt.xlabel(r'x/$\mu m$')
        plt.ylabel(r'y/$\mu m$')
        plt.imshow(imgLayer, origin='lower', interpolation='bilinear', extent=extent)
        cbar = plt.colorbar()
        if cBarLabel: cbar.set_label(cBarLabel)
        # if (saveFig): plt.savefig('plots/plot-'+title+'.pdf')

        if (saveFig): print("DEBUG:", title, "Min:", np.nanmin(imgLayer), "Max:", np.nanmax(imgLayer))

        if (saveFig): plt.show()

    plotLayer(x[..., -1], 'x', r'$\mu m$')
    plotLayer(y[..., -1], 'y', r'$\mu m$')

    plotLayer(ua[..., 0], 'ux_a', r'$\mu m$')
    plotLayer(us, 'ux_s', r'$\mu m$')
    plotLayer(ua[..., 0] - us, 'ux_a-ux_s', r'$\mu m$')

    plotLayer(va[..., 0], 'uy_a', r'$\mu m$')
    plotLayer(vs, 'uy_s', r'$\mu m$')
    plotLayer(va[..., 0] - vs, 'uy_a-uy_s', r'$\mu m$')

    plotLayer(wa[..., 0], 'uz_a', r'$\mu m$')
    plotLayer(ws, 'uz_s', r'$\mu m$')
    plotLayer(wa[..., 0] - ws, 'uz_a-uz_s', r'$\mu m$')

    uy, ux, uz = np.gradient(ua, spacing)
    vy, vx, vz = np.gradient(va, spacing)
    wy, wx, wz = np.gradient(wa, spacing)

    divU = ux + vy + wz

    # The surface normal of the plane is directed outwards, hence
    # n = (0,0,-1). Now use \vec{q} = \vec{n}\cdot\boldsymbol{\sigma}
    # qx = -sigma_xz
    # qy = -sigma_yz
    # qz = -sigma_zz

    qx = -mu * (uz + wx)
    qy = -mu * (vz + wy)
    qz = -mu * (wz + wz + 2 * nu / (1 - 2 * nu) * divU)

    qx = qx[..., 0]
    qy = qy[..., 0]
    qz = qz[..., 0]

    plotLayer(qx, 'qx', 'Pa')
    plotLayer(qy, 'qy', 'Pa')
    plotLayer(qz, 'qz', 'Pa')

    qyTheo = get_qy_SSurf(x[..., -1], y[..., -1], a, Qx)
    plotLayer(qyTheo, 'qy_theo', 'Pa')

    plotLayer(qy - qyTheo, 'qy-qy_theo', r'$\mu m$')
