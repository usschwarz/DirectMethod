"""
Calculates the characteristic quantities


We use a list or Hertz like point forces.

It returns lambda expressions to handle it Functions expects list of tuples
with the shape
(x0,y0,a,Qx,Qy,Qz)

where:
    x0,y0   Center of traction site
    a       Size of traction area
    Qx      Total traction force applied in x (tangential) direction
    Qy      Total traction force applied in y (tangential) direction
    Qz      Total traction force applied in z (normal)     direction

Notice that this doesn't work for overlapping structures

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""


import numpy as np
from numba import vectorize


def getInnerAreaMask(x, y, x0, y0, a):
    rSqr = (x - x0) ** 2 + (y - y0) ** 2
    return rSqr <= a * a


def getOutherAreaMask(x, y, x0, y0, a):
    rSqr = (x - x0) ** 2 + (y - y0) ** 2
    return rSqr > a * a


@vectorize(["f8(f8,f8,f8,f8,f8,f8)"], nopython=True)
def getTraction(x, y, x0, y0, a, Qi):
    rSqr = (x - x0) ** 2 + (y - y0) ** 2
    if rSqr < a * a:
        return (3 * Qi / (2 * np.pi * a ** 3)) * np.sqrt(a * a - rSqr)
    else:
        return 0.
    # return np.where(rSqr<a*a,(3*Qi/(2*np.pi*a**3))*np.sqrt(a*a-rSqr),0)



def get_DTMA_Hertz(pointList, x, y, qxExp, qyExp, qzExp):
    """
    calculates the DTMA of a set of Hertz like adhesions

    Huang, Y., Schell, C., Huber, T.B. et al.
    Traction force microscopy with optimized regularization
    and automated Bayesian parameter selection for comparing cells.
    Sci Rep 9, 539 (2019). https://doi.org/10.1038/s41598-018-36896-x

    equation 9
    """
    assert (len(pointList) > 0)
    qAbsExp = np.sqrt(qxExp ** 2 + qyExp ** 2 + qzExp ** 2)
    acc = 0
    for cTuple in pointList:
        x0, y0, a, Qx, Qy, Qz = cTuple
        QNorm = np.sqrt(Qx * Qx + Qy * Qy + Qz * Qz)
        qAbsi = getTraction(x, y, x0, y0, a, QNorm)  # ||t^true_i||_2(x,y)
        p = getInnerAreaMask(x, y, x0, y0, a)
        contrib = np.mean(qAbsExp[p] - qAbsi[p]) / np.mean(qAbsi[p])
        acc += contrib

    return acc / len(pointList)


def get_DTMB_Hertz(pointList, x, y, qxExp, qyExp, qzExp):
    """
    calculates the DTMB of a set of Hertz like adhesions

    Huang, Y., Schell, C., Huber, T.B. et al.
    Traction force microscopy with optimized regularization
    and automated Bayesian parameter selection for comparing cells.
    Sci Rep 9, 539 (2019). https://doi.org/10.1038/s41598-018-36896-x

    equation 10
    """
    assert (len(pointList) > 0)
    qAbsExp = np.sqrt(qxExp ** 2 + qyExp ** 2 + qzExp ** 2)
    pOut = (x == x)  # Truth grid
    acc = 0  # Sums up denominator contributions
    for cTuple in pointList:
        x0, y0, a, Qx, Qy, Qz = cTuple
        QNorm = np.sqrt(Qx * Qx + Qy * Qy + Qz * Qz)
        qAbsi = getTraction(x, y, x0, y0, a, QNorm)  # ||t^true_i||_2(x,y)
        p = getInnerAreaMask(x, y, x0, y0, a)
        acc += np.mean(qAbsi[p])

        # Set all points in p to false if they show up in pAcc
        # e.g.
        # pOut 0 1 0 1
        # p    0 0 1 1
        # out  0 1 0 0
        pOut = (pOut > p)
    return (np.mean(qAbsExp[pOut]) * len(pointList)) / acc


def get_SNR_Hertz(pointList, x, y, qxExp, qyExp, qzExp):
    """
    calculates the SNR of a set of Hertz like adhesions

    Huang, Y., Schell, C., Huber, T.B. et al.
    Traction force microscopy with optimized regularization
    and automated Bayesian parameter selection for comparing cells.
    Sci Rep 9, 539 (2019). https://doi.org/10.1038/s41598-018-36896-x

    equation 11
    """
    assert (len(pointList) > 0)
    qAbsExp = np.sqrt(qxExp ** 2 + qyExp ** 2 + qzExp ** 2)
    pOut = (x == x)  # Truth grid
    acc = 0
    for cTuple in pointList:
        x0, y0, a, _Qx, _Qy, _Qz = cTuple
        p = getInnerAreaMask(x, y, x0, y0, a)
        acc += np.mean(qAbsExp[p])

        # Set all points in p to false if they show up in pAcc
        # e.g.
        # pOut 0 1 0 1
        # p    0 0 1 1
        # out  0 1 0 0
        pOut = (pOut > p)

    sigma = np.sqrt(np.std(qxExp[pOut]) + np.std(qyExp[pOut]) + np.std(qzExp[pOut]))
    return (acc / len(pointList)) / sigma



def get_DMA_Hertz(pointList, x, y, qxExp, qyExp, qzExp):
    """
    calculates the DMA of a set of Hertz like adhesions

    Huang, Y., Schell, C., Huber, T.B. et al.
    Traction force microscopy with optimized regularization
    and automated Bayesian parameter selection for comparing cells.
    Sci Rep 9, 539 (2019). https://doi.org/10.1038/s41598-018-36896-x

    equation 12,  Assume that N_A = N_i
    """
    assert (len(pointList) > 0)
    qAbsExp = np.sqrt(qxExp ** 2 + qyExp ** 2 + qzExp ** 2)
    acc = 0
    for cTuple in pointList:
        x0, y0, a, Qx, Qy, Qz = cTuple
        QNorm = np.sqrt(Qx * Qx + Qy * Qy + Qz * Qz)
        qAbsi = getTraction(x, y, x0, y0, a, QNorm)  # ||t^true_i||_2(x,y)
        p = getInnerAreaMask(x, y, x0, y0, a)
        contrib = (np.max(qAbsExp[p]) - np.max(qAbsi[p])) / np.max(qAbsi[p])
        acc += contrib

    return acc / len(pointList)



def get_DTA_Hertz(pointList, x, y, qxExp, qyExp, qzExp):
    """
    calculates the DTA of a set of Hertz like adhesions

    Modelled after
    Sabass, B., Gardel, M. L., Waterman, C. M. & Schwarz, U. S.
    High resolution traction force microscopy based on experimental
    and computational advances.
    Biophys. J. 94, 207–220 (2008). https://doi.org/10.1529/biophysj.107.113670

    equation 15
    """
    assert (len(pointList) > 0)
    acc = 0
    for cTuple in pointList:
        x0, y0, a, Qx, Qy, Qz = cTuple
        QNorm = np.sqrt(Qx * Qx + Qy * Qy + Qz * Qz)
        qAbsi = getTraction(x, y, x0, y0, a, QNorm)  # ||t^true_i||_2(x,y)
        qxi = getTraction(x, y, x0, y0, a, Qx)
        qyi = getTraction(x, y, x0, y0, a, Qx)
        qzi = getTraction(x, y, x0, y0, a, Qx)
        p = getInnerAreaMask(x, y, x0, y0, a)
        prodI = qxi[p] * qxExp[p] + qyi[p] * qyExp[p] + qzi[p] * qzExp[p]
        qAbsExp = np.sqrt(qxExp[p] ** 2 + qyExp[p] ** 2 + qzExp[p] ** 2)
        acc += np.mean(prodI / (qAbsi[p] * qAbsExp))

    return acc / len(pointList)


def get_DTMA2D_Hertz(pointList, x, y, qxExp, qyExp, _qzExp):
    """ Variant of the DTMA for 2D """
    assert (len(pointList) > 0)
    qAbsExp = np.sqrt(qxExp ** 2 + qyExp ** 2)
    acc = 0
    for cTuple in pointList:
        x0, y0, a, Qx, Qy, Qz = cTuple
        QNorm = np.sqrt(Qx * Qx + Qy * Qy)
        if not np.isclose(QNorm, 0.):
            qAbsi = getTraction(x, y, x0, y0, a, QNorm)  # ||t^true_i||_2(x,y)
            p = getInnerAreaMask(x, y, x0, y0, a)
            contrib = np.mean(qAbsExp[p] - qAbsi[p]) / np.mean(qAbsi[p])
            acc += contrib

    return acc / len(pointList)


def get_DMA2D_Hertz(pointList, x, y, qxExp, qyExp, _qzExp):
    """ Variant of the DMA for 2D """
    assert (len(pointList) > 0)
    qAbsExp = np.sqrt(qxExp ** 2 + qyExp ** 2)
    acc = 0
    for cTuple in pointList:
        x0, y0, a, Qx, Qy, Qz = cTuple
        QNorm = np.sqrt(Qx * Qx + Qy * Qy)
        if not np.isclose(QNorm, 0.):
            qAbsi = getTraction(x, y, x0, y0, a, QNorm)  # ||t^true_i||_2(x,y)
            p = getInnerAreaMask(x, y, x0, y0, a)
            contrib = (np.max(qAbsExp[p]) - np.max(qAbsi[p])) / np.max(qAbsi[p])
            acc += contrib

    return acc / len(pointList)


## No Hertz Functions ##
def get_DTMA(qAbsTheo, qAbsReal):
    p = np.isclose(qAbsTheo, 0.)
    assert (np.sum(p) != 0)
    return np.mean(qAbsReal[p] - qAbsTheo[p]) / np.mean(qAbsTheo[p])


def get_DTMB(qAbsTheo, qAbsReal):
    p = np.isclose(qAbsTheo, 0.)
    assert (np.sum(p) != 0)
    return np.mean(qAbsReal[np.logical_not(p)]) / np.mean(qAbsTheo[p])


def get_SNR(qAbsTheo, qAbsReal):
    p = np.isclose(qAbsTheo, 0.)
    assert (np.sum(p) != 0)
    return np.mean(qAbsReal[p]) / np.std(qAbsTheo[np.logical_not(p)])
