"""
Calculate field properties

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""


from scipy import integrate
import numpy as np


def getQiGes(ai, xSpacing, ySpacing):
    return integrate.simps(integrate.simps(ai, dx=ySpacing), dx=xSpacing)


def getCenterOld(x, y, ax, ay, az, xSpacing, ySpacing):
    # print("Fololo",x.shape,y.shape,ax.shape,ay.shape,xSpacing,ySpacing)

    a = np.sqrt(ax * ax + ay * ay + az * az)

    aTot = getQiGes(a, xSpacing, ySpacing)

    bx = integrate.simps(integrate.simps(a * x, dx=ySpacing), dx=xSpacing)
    by = integrate.simps(integrate.simps(a * y, dx=ySpacing), dx=xSpacing)

    cmx, cmy = bx / aTot, by / aTot

    return cmx, cmy, x - cmx, y - cmy


def getCenter(x, y, ax, ay, az, xSpacing, ySpacing):
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
    cmx, cmy, x2cm, y2cm = getCenter(x, y, ax, ay, az, xSpacing, ySpacing)

    return getDipole(x2cm, y2cm, ax, ay, xSpacing, ySpacing)


def getl2Diff(qx, qy, qz, qxR, qyR, qzR, xSpacing, ySpacing):
    nx, ny, nz = qx - qxR, qy - qyR, qz - qzR
    nAbsSq = nx * nx + ny * ny + nz * nz
    normSq = integrate.simps(integrate.simps(nAbsSq, dx=ySpacing), dx=xSpacing)
    return np.sqrt(normSq)


def get_bg_noise_level(qx, qy, qz, qxR, qyR, qzR):
    mask = np.isclose(qxR * qxR + qyR * qyR + qzR * qzR, 0.)
    qAbsSq = qx * qx + qy * qy + qz * qz
    return np.sqrt(np.mean(qAbsSq[mask]))


def getSNR(qx, qy, qz, qxR, qyR, qzR, xSpacing, ySpacing):
    nx, ny, nz = qx - qxR, qy - qyR, qz - qzR
    nAbs = np.sqrt(nx * nx + ny * ny + nz * nz)
    qRAbs = np.sqrt(qxR * qxR + qyR * qyR + qzR * qzR)
    nSigma = np.std(nAbs)
    qmean = np.mean(qRAbs)
    return qmean / nSigma
