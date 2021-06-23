#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:56:27 2019

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import numpy as np


def dumpu2npz(filename, ux, uy, uz, uAbs, cc, dmz, dm, m, E, nu, nCritical):
    """ Store deformation field into a npz file """
    np.savez(filename, ux=ux, uy=uy, uz=uz, uAbs=uAbs, cc=cc, dmz=dmz, dm=dm, mx=m[0], my=m[1], mz=m[2],
             InfoBlob=np.array([E, nu]), nCritical=nCritical)
    print("Saved results in ", filename)


def uFieldSim(uFun, xpix, ypix, umPerPx, nLayers, dLayers, E=None, nu=None, noiseLevel=0, fname="uField.npz",
              verbose=False, dm=8):
    """ Sample deformation field uFun as it would be returned by a DVC analysis """
    xysep = dm * umPerPx
    zsep = dLayers * dm

    xlen = xpix * umPerPx
    ylen = ypix * umPerPx
    zlen = dLayers * nLayers

    uFieldSimR(uFun, xysep, zsep, xlen, ylen, zlen, E, nu, noiseLevel, fname, verbose)


def uFieldSimR(uFun, xysep, zsep, xlen, ylen, zlen, E, nu, noiseLevel, fname="uField.npz", verbose=False):
    """ Sample deformation field """
    xR = np.arange(0, xlen, xysep)
    yR = np.arange(0, ylen, xysep)
    zR = np.arange(0, zlen, zsep)

    xR -= xR[-1] / 2
    yR -= yR[-1] / 2

    print("Debug xR", xR)
    print("Debug zR", zR)

    x, y, z = np.meshgrid(xR, yR, zR, indexing='ij')

    u, v, w = uFun(x, y, z)

    if verbose:
        print("Simulated shape {}".format(u.shape))

    nanPoints = np.logical_or(np.logical_or(np.isnan(u), np.isnan(v)), np.isnan(w))
    nCount = len(x[nanPoints])
    # print(nCount)
    assert (nCount == 0)

    uAbs = np.sqrt(u * u + v * v + w * w)

    # uMax = np.max(uAbs)
    uiMean = np.mean(uAbs) / np.sqrt(3)

    if noiseLevel == 0:
        if verbose:
            print("We have no noise")
        uB = u
        vB = v
        wB = w
        uBAbs = uAbs
        cc = np.ones_like(u)
    else:
        uNoise = np.random.normal(scale=noiseLevel * uiMean, size=(3,) + uAbs.shape)
        uB = u + uNoise[0]
        vB = v + uNoise[1]
        wB = w + uNoise[2]
        uBAbs = np.sqrt(uB * uB + vB * vB + wB * wB)
        cc = np.sqrt(uNoise[0] * uNoise[0] + uNoise[1] * uNoise[1] + uNoise[2] * uNoise[2])
        ccMax = np.max(cc)
        cc = ccMax - cc

    m = [xR, yR, zR]
    dumpu2npz(fname, uB, vB, wB, uBAbs, cc, zsep, xysep, m, E, nu, 0)
