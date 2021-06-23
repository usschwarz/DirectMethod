#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:28:59 2019

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

# Calculate field properties

from scipy import integrate
import numpy as np


def getQiRef(refFile):
    with np.load(refFile) as data:
        qx = data['qx']
        qy = data['qy']
        qz = data['qz']
    return qx, qy, qz


def getQiGes(ai, xSpacing, ySpacing):
    return integrate.simps(integrate.simps(ai, dx=ySpacing), dx=xSpacing)


def getCenter(x, y):
    mx = x[:, 0]
    my = y[0, :]

    cmx = np.median(mx)
    cmy = np.median(my)

    print("CMS", cmx, cmy)
    return cmx, cmy, x - cmx, y - cmy


def getQGes(ax, ay, az, xSpacing, ySpacing):
    return getQiGes(ax, xSpacing, ySpacing), getQiGes(ay, xSpacing, ySpacing), getQiGes(az, xSpacing, ySpacing)


def getDipole(x2cm, y2cm, ax, ay, xSpacing, ySpacing):
    mxx = integrate.simps(integrate.simps(ax * x2cm, dx=ySpacing), dx=xSpacing)
    mxy = integrate.simps(integrate.simps(ax * y2cm, dx=ySpacing), dx=xSpacing)
    myx = integrate.simps(integrate.simps(ay * x2cm, dx=ySpacing), dx=xSpacing)
    myy = integrate.simps(integrate.simps(ay * y2cm, dx=ySpacing), dx=xSpacing)

    print("MVals", mxx, mxy, myx, myy)

    # Symmetrise
    mxyN = (mxy + myx) / 2

    mMat = np.array([[mxx, mxyN], [mxyN, myy]])

    eigval, eigvec = np.linalg.eig(mMat)

    return mMat, eigval, eigvec


def getDipoleDirect(x, y, ax, ay, az, xSpacing, ySpacing):
    _cmx, _cmy, x2cm, y2cm = getCenter(x, y)

    return getDipole(x2cm, y2cm, ax, ay, xSpacing, ySpacing)
