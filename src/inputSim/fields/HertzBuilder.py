#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:29:44 2020

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)

This file builds up a force profile using Hertz like adheasion sites
using tHertzField, sHertzField and nHertzFields

It returns lambda expressions to handle it Functions expects list of tuples
with the shape
(x,y,a,Qx,Qy,Qz)

where:
    x,y     Adhesion Center
    a       Size of traction area
    Qx      Total traction force applied in x (tangential) direction
    Qy      Total traction force applied in y (tangential) direction
    Qz      Total traction force applied in z (normal)     direction
"""

import numpy as np

import os
import sys

from numba import vectorize, float64

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from tHertzField import get_u1_THertz, get_u2_THertz, get_u3_THertz
from tHertzField import get_u1_TSurf, get_u2_TSurf, get_u3_TSurf
from tHertzField import get_qx_TSurf

from sHertzField import get_u1_SHertz, get_u2_SHertz, get_u3_SHertz
from sHertzField import get_u1_SSurf, get_u2_SSurf, get_u3_SSurf
from sHertzField import get_qy_SSurf

from nHertzField import get_u1_NHertz, get_u2_NHertz, get_u3_NHertz
from nHertzField import get_u1_NSurf, get_u2_NSurf, get_u3_NSurf
from nHertzField import get_qz_NSurf

__all__ = ['get_u_hertz_pattern', 'get_uSurf_hertz_pattern', 'get_q_hertz_pattern',
           'get_u_hertz_single', 'get_uSurf_hertz_single', 'get_q_hertz_single']


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def _zero_ufield(_x, _y, _z, _a, _Qx, _mu, _nu):
    return 0


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def _zero_usurf(_x, _y, _a, _Qz, _mu, _nu):
    return 0


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def _zero_qsurf(_x, _y, _a, _Qy):
    return 0


def get_u_hertz_pattern(pointList):
    #  Compile pointlist
    vec  = []
    for point in pointList:
        x0, y0, a, Qx, Qy, Qz = point
        if Qx != 0:
            vec.append((x0, y0, a, Qx, get_u1_THertz, get_u2_THertz, get_u3_THertz))
        if Qy != 0:
            vec.append((x0, y0, a, Qy, get_u1_SHertz, get_u2_SHertz, get_u3_SHertz))
        if Qz != 0:
            vec.append((x0, y0, a, Qz, get_u1_NHertz, get_u2_NHertz, get_u3_NHertz))

    if len(vec) == 0:
        # Dummy entry to make sure that vec has at least one entry
        vec.append((0, 0, 0, 0, _zero_ufield, _zero_ufield, _zero_ufield))

    # Extracts a single element ot of the list, so it can be used for the nosum part
    extra = vec.pop()

    def ux_fun(x, y, z, mu, nu):
        xc, yc, ac, Qi, get_ux_i, _, _ = extra
        res = get_ux_i(x - xc, y - yc, z, ac, Qi, mu, nu)
        for entry in vec:
            xc, yc, ac, Qi, get_ux_i, _, _ = entry
            res += get_ux_i(x - xc, y - yc, z, ac, Qi, mu, nu)
        return res

    def uy_fun(x, y, z, mu, nu):
        xc, yc, ac, Qi, _, get_uy_i, _ = extra
        res = get_uy_i(x - xc, y - yc, z, ac, Qi, mu, nu)
        for entry in vec:
            xc, yc, ac, Qi, _, get_uy_i, _ = entry
            res += get_uy_i(x - xc, y - yc, z, ac, Qi, mu, nu)
        return res

    def uz_fun(x, y, z, mu, nu):
        xc, yc, ac, Qi, _, _, get_uz_i = extra
        res = get_uz_i(x - xc, y - yc, z, ac, Qi, mu, nu)
        for entry in vec:
            xc, yc, ac, Qi, _, _, get_uz_i = entry
            res += get_uz_i(x - xc, y - yc, z, ac, Qi, mu, nu)
        return res

    return ux_fun, uy_fun, uz_fun


def get_uSurf_hertz_pattern(pointList):
    #  Compile pointlist
    vec  = []
    for point in pointList:
        x0, y0, a, Qx, Qy, Qz = point
        if Qx != 0:
            vec.append((x0, y0, a, Qx, get_u1_TSurf, get_u2_TSurf, get_u3_TSurf))
        if Qy != 0:
            vec.append((x0, y0, a, Qy, get_u1_SSurf, get_u2_SSurf, get_u3_SSurf))
        if Qz != 0:
            vec.append((x0, y0, a, Qz, get_u1_NSurf, get_u2_NSurf, get_u3_NSurf))

    if len(vec) == 0:
        # Dummy entry to make sure that vec has at least one entry
        vec.append((0, 0, 0, 0, _zero_usurf, _zero_usurf, _zero_usurf))

    # Extracts a single element ot of the list, so it can be used for the nosum part
    extra = vec.pop()

    print(len(vec))

    def ux_surf(x, y, mu, nu):
        xc, yc, ac, Qi, get_ux_i, _, _ = extra
        res = get_ux_i(x - xc, y - yc, ac, Qi, mu, nu)
        for entry in vec:
            xc, yc, ac, Qi, get_ux_i, _, _ = entry
            res += get_ux_i(x - xc, y - yc, ac, Qi, mu, nu)
        return res

    def uy_surf(x, y, mu, nu):
        xc, yc, ac, Qi, _, get_ux_i, _ = extra
        res = get_ux_i(x - xc, y - yc, ac, Qi, mu, nu)
        for entry in vec:
            xc, yc, ac, Qi, _, get_ux_i, _ = entry
            res += get_ux_i(x - xc, y - yc, ac, Qi, mu, nu)
        return res

    def uz_surf(x, y, mu, nu):
        xc, yc, ac, Qi, _, _, get_ux_i = extra
        res = get_ux_i(x - xc, y - yc, ac, Qi, mu, nu)
        for entry in vec:
            xc, yc, ac, Qi, _, _, get_ux_i = entry
            res += get_ux_i(x - xc, y - yc, ac, Qi, mu, nu)
        return res

    return ux_surf, uy_surf, uz_surf


def get_q_hertz_pattern(pointList):
    #  Compile pointlist
    vec1 = []
    vec2 = []
    vec3 = []

    for point in pointList:
        x0, y0, a, Qx, Qy, Qz = point
        if Qx != 0:
            vec1.append((x0, y0, a, Qx, get_qx_TSurf))
        if Qy != 0:
            vec2.append((x0, y0, a, Qy, get_qy_SSurf))
        if Qz != 0:
            vec3.append((x0, y0, a, Qz, get_qz_NSurf))

    if len(vec1) == 0:
        # Dummy entry to make sure that vec1 has at least one entry
        vec1.append((0, 0, 0, 0, _zero_qsurf))

    if len(vec2) == 0:
        # Dummy entry to make sure that vec2 has at least one entry
        vec2.append((0, 0, 0, 0, _zero_qsurf))

    if len(vec3) == 0:
        # Dummy entry to make sure that vec3 has at least one entry
        vec3.append((0, 0, 0, 0, _zero_qsurf))

    # Extracts a single element ot of the list, so it can be used for the nosum part
    extra1 = vec1.pop()
    extra2 = vec2.pop()
    extra3 = vec3.pop()

    def qx_fun(x, y):
        xc, yc, ac, Qi, get_qi = extra1
        res = get_qi(x - xc, y - yc, ac, Qi)
        for entry in vec1:
            xc, yc, ac, Qi, get_qi = entry
            res += get_qi(x - xc, y - yc, ac, Qi)
        return res

    def qy_fun(x, y):
        xc, yc, ac, Qi, get_qi = extra2
        res = get_qi(x - xc, y - yc, ac, Qi)
        for entry in vec2:
            xc, yc, ac, Qi, get_qi = entry
            res += get_qi(x - xc, y - yc, ac, Qi)
        return res

    def qz_fun(x, y):
        xc, yc, ac, Qi, get_qi = extra3
        res = get_qi(x - xc, y - yc, ac, Qi)
        for entry in vec3:
            xc, yc, ac, Qi, get_qi = entry
            res += get_qi(x - xc, y - yc, ac, Qi)
        return res

    return qx_fun, qy_fun, qz_fun


def get_u_hertz_single(x0, y0, a, Qx, Qy, Qz):
    get_q_hertz_pattern([(x0, y0, a, Qx, Qy, Qz)])


def get_uSurf_hertz_single(x0, y0, a, Qx, Qy, Qz):
    get_uSurf_hertz_pattern([(x0, y0, a, Qx, Qy, Qz)])


def get_q_hertz_single(x0, y0, a, Qx, Qy, Qz):
    get_q_hertz_pattern([(x0, y0, a, Qx, Qy, Qz)])


if __name__ == "__main__":
    # Code to test integrity of the solution

    spacing = 1.
    mcVal = 200.5
    mcValZ = 10

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-mcVal, mcVal, spacing),
                          np.arange(-mcVal, mcVal, spacing),
                          np.arange(0, 2 * mcValZ, spacing))

    nu = 0.499
    E = 20000.0  # Pa
    mu = E / (2.0 * (1.0 + nu))

    TH_Params = (x, y, z, mu, nu)

    TS_Params = (x[..., -1], y[..., -1], mu, nu)

    pattern = [(0, 0, 8, 0, 0, 10), (10, 10, 8, 4, 4, 0), (-10, -10, 8, -4, -4, 0)]

    getu1, getu2, getu3 = get_u_hertz_pattern(pattern)

    ua = getu1(*TH_Params)
    va = getu2(*TH_Params)
    wa = getu3(*TH_Params)


    def hDataInfo(name, dx, dy, dz):
        print("{} data info".format(name))
        print(" type and dimensions:", type(dx[0, 0, 0]), dx.shape)
        print(" no of datapoints", np.prod(dx.shape), np.prod(dy.shape), np.prod(dz.shape))
        print(" no of Nan-points", len(dx[dx == np.nan]), len(dy[dy == np.nan]), len(dz[dz == np.nan]))
        print(" Peak Values:", np.nanmax(dx), np.nanmax(dy), np.nanmax(dz))


    hDataInfo("u", ua, va, wa)

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
        if cBarLabel:
            cbar.set_label(cBarLabel)
        if saveFig:
            print("DEBUG:", title, "Min:", np.nanmin(imgLayer), "Max:", np.nanmax(imgLayer))
            plt.show()


    plotLayer(x[..., -1], 'x', r'$\mu m$')
    plotLayer(y[..., -1], 'y', r'$\mu m$')

    plotLayer(ua[..., 0], 'ux_a', r'$\mu m$')
    plotLayer(va[..., 0], 'uy_a', r'$\mu m$')
    plotLayer(wa[..., 0], 'uz_a', r'$\mu m$')

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

    # hDataInfo("q",qx,qy,qz)

    plotLayer(qx, 'qx', 'Pa')
    plotLayer(qy, 'qy', 'Pa')
    plotLayer(qz, 'qz', 'Pa')

    getqx, getqy, getqz = get_q_hertz_pattern(pattern)

    qxTheo = getqx(x[..., -1], y[..., -1])
    qyTheo = getqy(x[..., -1], y[..., -1])
    qzTheo = getqz(x[..., -1], y[..., -1])
    plotLayer(qxTheo, 'qx_theo', 'Pa')
    plotLayer(qyTheo, 'qy_theo', 'Pa')
    plotLayer(qzTheo, 'qz_theo', 'Pa')

    plotLayer(qx - qxTheo, 'qx-qx_theo', r'$\mu m$')
    plotLayer(qy - qyTheo, 'qy-qy_theo', r'$\mu m$')
    plotLayer(qz - qzTheo, 'qz-qz_theo', r'$\mu m$')
