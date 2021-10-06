"""
Functions used to generate simulated deformation field

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
    xysep = dm * umPerPx  # xy spacing in µm
    zsep = dLayers * dm  # z sep in µm

    xlen = xpix * umPerPx  # size of the image in µm
    ylen = ypix * umPerPx  # size of the image in µm
    zlen = dLayers * nLayers  # depth in µm

    uFieldSimR(uFun, xysep, zsep, xlen, ylen, zlen, E, nu, noiseLevel, fname, verbose)


def uFieldSimNew(
        uFun, n_points_x, n_points_y, n_points_z, sp_xy, sp_z, E=None, nu=None,
        noiseLevel=0, fname="uField.npz", verbose=False
):
    xysep = sp_xy  # xy spacing in µm
    zsep = sp_z  # z sep in µm

    xlen = n_points_x * sp_xy  # size of the image in µm
    ylen = n_points_y * sp_xy  # size of the image in µm
    zlen = n_points_z * sp_z  # depth in µm

    uFieldSimR(uFun, xysep, zsep, xlen, ylen, zlen, E, nu, noiseLevel, fname, verbose)

def uFieldSimR(uFun, xysep, zsep, xlen, ylen, zlen, E, nu, noiseLevel, fname="uField.npz", verbose=False):
    """ Sample deformation field """

    # Find sampling
    xR = np.arange(0, xlen, xysep)
    yR = np.arange(0, ylen, xysep)
    zR = np.arange(0, zlen, zsep)

    xR -= xR[-1] / 2
    yR -= yR[-1] / 2

    # xR, yR, zR describe a mesh grid input in µm

    print("Debug xR", xR)
    print("Debug zR", zR)

    x, y, z = np.meshgrid(xR, yR, zR, indexing='ij')

    # u, v, w are in µm
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
    dumpu2npz(
        filename=fname,
        ux=uB,
        uy=vB,
        uz=wB,
        uAbs=uBAbs,
        cc=cc,
        dmz=zsep,
        dm=xysep,
        m=m,
        E=E,
        nu=nu,
        nCritical=0,
    )