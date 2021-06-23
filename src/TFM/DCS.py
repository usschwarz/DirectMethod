#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:03:31 2018

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

# Implement the algorithm from 
# http://ltces.dem.ist.utl.pt/lxlaser/lxlaser2016/finalworks2016/papers/03.14_2_104paper.pdf
# and
# Wang, C., Gao, Q., Wei, R. et al.
# Weighted divergence correction scheme and its fast implementation.
# Exp Fluids 58, 44 (2017). https://doi.org/10.1007/s00348-017-2307-0

import numpy as np
from numba import vectorize
from scipy import sparse
from scipy.optimize import fmin


def getDivDiag(n):
    """ Calculate finite difference matrix as sparse diag matrix """
    a = np.ones(n)
    b = np.zeros(n)
    c = -np.ones(n)
    a[1] = 2
    b[0] = -2
    b[-1] = 2
    c[-2] = -2
    m = np.array([a / 2, b / 2, c / 2])
    return sparse.dia_matrix((m, (1, 0, -1)), shape=(n, n), dtype="float64")


def _prepare_DCS(inShape, xsp, ysp, zsp):
    # Findes the 1d differential operator matrices
    # and the needed Eigenvalue decomposition:
    # di @ di.T = Pi @ diag(li) @ Pi.T
    #
    # Note that Pi and li can also be found using an SVD:
    # Di = U @ diag(s) @ V.A, then Pi = U  li = |s|^2 
    # 
    # These will then be used for the divFreeFit
    # As this computation is rather complex and does not depend on the
    # the actual input field but only it's shape, this information is calculated
    # beforehand and then cached for the GCV as well as the main DCS calculation. 
    nx, ny, nz = inShape

    # Avoid dublicate calculation in eigenbase decomposition
    # As this operation is expensive
    # Also use the fact that
    # di(n,sp1)*sp1 = di(n,sp2)*sp2

    def getNewDi(n, sp):
        Di = (getDivDiag(n) / sp).todense()
        li, Pi = np.linalg.eigh(Di @ Di.T)
        # li[0]= 0 # Fixes some problems I hope
        return Di, li, Pi

    def recycleDi(DiOld, liOld, PiOld, spOld, spNew):
        # Use this if we have allread calculated the decomposition
        # for a matrix with this dimension
        assert (spOld != 0)
        if spOld == spNew:
            # Full recycle
            return DiOld, liOld, PiOld
        else:
            # Need to rescale
            # Eigenstates remain the same
            return (spOld / spNew) * DiOld, (spNew / spOld) * liOld, PiOld

    Dx, lx, Px = getNewDi(nx, xsp)

    # print("DShapes",Dx.shape,lx.shape,Px.shape)

    if nx == ny:
        Dy, ly, Py = recycleDi(Dx, lx, Px, xsp, ysp)
    else:
        Dy, ly, Py = getNewDi(ny, ysp)

    if nx == nz:
        Dz, lz, Pz = recycleDi(Dx, lx, Px, xsp, zsp)
    elif ny == nz:
        Dz, lz, Pz = recycleDi(Dy, ly, Py, ysp, zsp)
    else:
        Dz, lz, Pz = getNewDi(nz, zsp)

    return (Dx, Dy, Dz), (lx, ly, lz), (Px, Py, Pz)


def get_div_vec3d(prepData, u, v, w):
    """ Get divergence of a 3d vector """

    # Check if u,v,w are really 3 matching components
    assert (u.shape == v.shape)
    assert (u.shape == w.shape)

    nx, ny, nz = u.shape

    # This contains derivative matrices Di, And the
    # Eigenvalue decomposition of Di @ Di.T = Pi @ diag(li) @ Pi.T
    (Dx, Dy, Dz), _, _ = prepData

    # Quantity S
    S = np.einsum("io,ojk->ijk", Dx, u) + \
        np.einsum("jp,ipk->ijk", Dy, v) + \
        np.einsum("kq,ijq->ijk", Dz, w)

    return S


def get_div_voigt(prepData, mvoigt):
    """ Get divergence of a 3x3 symmetric matrix in Voigt notation """

    mxx, myy, mzz, myz, mxz, mxy = mvoigt
    Sx = get_div_vec3d(prepData, mxx, mxy, mxz)
    Sy = get_div_vec3d(prepData, mxy, myy, myz)
    Sz = get_div_vec3d(prepData, mxz, myz, mzz)
    return Sx, Sy, Sz


def _getWeightDCS(prepData, uvw, S, verbose=False):
    # Unpack prepData
    # This contains derivative matrices Di, And the
    # Eigenvalue decomposition of Di @ Di.T = Pi @ diag(li) @ Pi.T
    # Note that Pi and li can also be found using an SVD:
    # Di = U @ diag(s) @ V.A, then Pi = U  li = |s|^2
    _, (lx, ly, lz), _ = prepData
    u, v, w = uvw

    # Note that the GCV calculation requiers calling up the WDCS
    # main function
    def GCV(al1, al2, al3):
        uc, vc, wc, G = _DCS_main(prepData, (u, v, w), S, al1, al2, al3)
        invG = 1 / G  # This is save as zero components of G are fixed
        return (al1 ** 4) * np.sum((uc - u) ** 2) / np.einsum("i,ijk->", lx, invG) \
               + (al2 ** 4) * np.sum((vc - v) ** 2) / np.einsum("j,ijk->", ly, invG) \
               + (al3 ** 4) * np.sum((wc - w) ** 2) / np.einsum("k,ijk->", lz, invG)

    alo, GCVo, nIter, fcalls, warn, allvecs = fmin(lambda aly, alz: GCV(1., aly, alz), np.zeros(2))
    al1, al2, al3 = 1., alo[0], alo[1]
    if verbose:
        print("alpha = ({},{},{})".format(al1, al2, al3))
        print("GCVo = {},iter = {},fcalls = {}, warn= {}".format(GCVo, nIter, fcalls, warn))
        print("allvecs = {}".format(allvecs))
    return al1, al2, al3


def _DCS_main(prepData, uvw, S, al1=1., al2=1., al3=1.):
    u, v, w = uvw
    # Performs actual calculations.

    # Check if u,v,w are really 3 matching components
    assert (u.shape == v.shape)
    assert (u.shape == w.shape)

    nx, ny, nz = u.shape

    # Unpack prepData
    # This contains derivative matrices Di, And the
    # Eigenvalue decomposition of Di @ Di.T = Pi @ diag(li) @ Pi.T
    (Dx, Dy, Dz), (lx, ly, lz), (Px, Py, Pz) = prepData

    # Catch if a wrong set of prepData has been passed
    assert (Dx.shape[0] == nx)
    assert (Dy.shape[0] == ny)
    assert (Dz.shape[0] == nz)

    # Quantity S (precalculated)
    # S = np.einsum("io,ojk->ijk", Dx, u) + \
    #     np.einsum("jp,ipk->ijk", Dy, v) + \
    #     np.einsum("kq,ijq->ijk", Dz, w)

    # Quantity Γ
    # G = np.einsum("i,j,k->ijk",lx/(al1**2),ly/(al2**2),lz/(al3**2))
    G = np.add.outer(lx / (al1 ** 2), np.add.outer(ly / (al2 ** 2), lz / (al3 ** 2)))

    # Zero element removal.
    # G will contain exactly 1 zero element, that must be replaced by an arbitay non zero value.
    # Due to numerical errors, this might not be exacty zero.
    # While this value is mathematically valid, it will mess up the numerics later on.
    # Therefore replace it by 1.
    # We simply replace the element with the smallest ampitude
    # This can be done because we only add Kern(DivOp.T) modes to Mu

    # Eigenvalues in li are sorted and prositive. Therefore smallest element is allways the first element
    G[0, 0, 0] = 1

    # Quantity 𝜇
    Mu = np.einsum("il,jm,kn,lmn,ol,pm,qn,opq->ijk", Px, Py, Pz, np.reciprocal(G), Px, Py, Pz, S, optimize=True) # noqa

    uc = u - np.einsum("oi,ojk->ijk", Dx, Mu) / (al1 * al1)
    vc = v - np.einsum("pj,ipk->ijk", Dy, Mu) / (al2 * al2)
    wc = w - np.einsum("qk,ijq->ijk", Dz, Mu) / (al3 * al3)

    # Also return G as this is needed for the GCV calculation
    return uc, vc, wc, G


def DCS(u, v, w, xsp, ysp, zsp, weighting=False):
    """ Calculate DCS of the vector field (u, v, w) """
    assert (u.shape == v.shape)
    assert (u.shape == w.shape)

    # Calculate derivative operator matricies and their eigenvalue decomposition
    prepData = _prepare_DCS(u.shape, xsp, ysp, zsp)

    S = get_div_vec3d(prepData, u, v, w)

    # If needed prepare weighting:
    if weighting:
        al1, al2, al3 = _getWeightDCS(prepData, (u, v, w), S)
    else:
        al1, al2, al3 = 1., 1., 1.

    # Calculate and return divFreeFit
    uc, vc, wc, _G = _DCS_main(prepData, (u, v, w), S, al1, al2, al3)
    return uc, vc, wc


def mean_norm_3d(vx, vy, vz):
    """ Get the mean norm of a 3d vector """

    vnorm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return float(np.mean(vnorm))


def mean_norm_voigt(mvoigt):
    """ Get mean of a 3x3 symmetric matrix in Voigt notation """

    mxx, myy, mzz, myz, mxz, mxy = mvoigt
    mnorm = np.sqrt(mxx ** 2 + myy ** 2 + mzz ** 2 + 2 * mxy ** 2 + 2 * mxz ** 2 + 2 * mzz ** 2)
    print("DBG: mnorm", np.mean(mnorm))
    return float(np.mean(mnorm))


@vectorize("f8(f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def frob3x3(m11, m12, m13, m21, m22, m23, m31, m32, m33):
    """ Calculate Frobinius norm of a 3x3 matrix """
    normSq = m11 ** 2 + m12 ** 2 + m13 ** 2 + m21 ** 2 + m22 ** 2 + m23 ** 2 + m31 ** 2 + m32 ** 2 + m33 ** 2
    return np.sqrt(normSq)


@vectorize("f8(f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def symmetrisity3x3(m11, m12, m13, m21, m22, m23, m31, m32, m33):
    """ Calculate Relative symmetrisity """
    # s(A) = 1 - (1/2*norm(A-AT)/norm(A))

    F8_SMALL = 2.2250738585072009e-308  # Smallest normal number

    normU = np.sqrt(2 * (m12 - m21) ** 2 + 2 * (m13 - m31) ** 2 + 2 * (m23 - m32) ** 2)
    normA = frob3x3(m11, m12, m13, m21, m22, m23, m31, m32, m33)

    # This will likely not give a usefull result any more
    if normA < F8_SMALL:
        return np.nan
    else:
        return 1 - normU / (2 * normA)


def DCS_voigt_internal(tFlat, xsp, ysp, zsp, weighting=False, divergenceThreshold=0.0002, symmetryThreshold=1.0,
                       maxIterations=20, verbose=False):
    """ Calculate DCS of a 3x3 symmetric matrix in Voigt representation using specific convergence conditions """
    assert (len(tFlat) == 6)

    def print_verbose(text):
        if verbose:
            print(text)

    # Unpack Stuff
    # txx,tyy,tzz,tyz,txz,txy = tFlat

    # These 3 pic the right components for fixing one index
    #  0, 1, 2, 3, 4, 5
    # xx,yy,zz,yz,xz,xy

    xsel = [0, 5, 4]
    ysel = [5, 1, 3]
    zsel = [4, 3, 2]

    nTot = np.prod(tFlat[0].shape)

    # Prepare matrices that can be reused during all iterations
    print_verbose("Calculating decomposition matricies used")
    prepData = _prepare_DCS(tFlat[0].shape, xsp, ysp, zsp)

    # "Fit"

    # Rename to avoid writeback to global var
    ti = np.array(tFlat)

    print_verbose("Use weighting? {}".format(weighting))

    tnormMean = mean_norm_voigt(ti)

    iteration = 0

    SNormMeanList = []
    DivlevelList = []
    MdpeakList = [0.]
    SmatmeanList = [1.]
    nancountList = [0]

    while True:
        # Analyse main metric: Divergence of the resulting tensor
        Sx, Sy, Sz = get_div_voigt(prepData, ti)
        SNormMean = mean_norm_3d(Sx, Sy, Sz)
        divergence_level = SNormMean / tnormMean
        print_verbose("Iteration {}: abs. mean div={}, rel. mean div={}".format(iteration, SNormMean, divergence_level))
        SNormMeanList.append(SNormMean)
        DivlevelList.append(divergence_level)

        if divergence_level < divergenceThreshold:
            print_verbose(
                "Reached convergence divergence level {} after iteration {}".format(divergence_level, iteration))
            break

        # Next iteration
        iteration += 1

        if iteration > maxIterations:
            if divergenceThreshold == 0.0:
                print_verbose("Exiting after {} iterations".format(maxIterations))
            else:
                print_verbose("Maximum number of iterations reached {}".format(maxIterations))
            break

        # Fit all modes
        if weighting:
            al1, al2, al3 = _getWeightDCS(prepData, ti[xsel], Sx)
            wxx, wxy, wxz, _ = _DCS_main(prepData, ti[xsel], Sx, al1, al2, al3)
            al1, al2, al3 = _getWeightDCS(prepData, ti[ysel], Sx)
            wyx, wyy, wyz, _ = _DCS_main(prepData, ti[ysel], Sy, al1, al2, al3)
            al1, al2, al3 = _getWeightDCS(prepData, ti[zsel], Sx)
            wzx, wzy, wzz, _ = _DCS_main(prepData, ti[zsel], Sz, al1, al2, al3)
        else:
            wxx, wxy, wxz, _ = _DCS_main(prepData, ti[xsel], Sx)
            wyx, wyy, wyz, _ = _DCS_main(prepData, ti[ysel], Sy)
            wzx, wzy, wzz, _ = _DCS_main(prepData, ti[zsel], Sz)

        # Old critearia 1
        mdyz = np.linalg.norm(np.ravel(wyz - wzy)) / nTot
        mdxz = np.linalg.norm(np.ravel(wxz - wzx)) / nTot
        mdxy = np.linalg.norm(np.ravel(wxy - wyx)) / nTot
        print_verbose("average Differences {} {} {}".format(mdyz, mdxz, mdxy))
        mdPeak = float(np.max((mdyz, mdxz, mdxy)))
        MdpeakList.append(mdPeak)

        mdMax = 0.05
        if mdPeak < mdMax:
            print_verbose("Old break criteria fullfilled.")

        # Relative symmetrisity
        smat = symmetrisity3x3(wxx, wxy, wxz, wyx, wyy, wyz, wzx, wzy, wzz)
        frob = frob3x3(wxx, wxy, wxz, wyx, wyy, wyz, wzx, wzy, wzz)

        pickcount = 10

        smat_picked = []

        for i in range(10):
            idx = np.nanargmax(frob)
            frob.flat[idx] = -frob.flat[idx]
            smat_picked.append(smat.flat[idx])

        nancount = np.count_nonzero(smat == np.nan)

        nancountList.append(nancount)
        smat[smat == np.nan] = 1.
        smat_mean = float(np.mean(smat))

        SmatmeanList.append(smat_mean)

        if nancount == 0:
            print_verbose("Mean symmetricisty {}".format(smat_mean))
        else:
            print_verbose("Mean symmetricisty {} (#nans={})".format(smat_mean, nancount))

        if smat_mean > symmetryThreshold:
            print("Reached desired symmetrisity after iteration {}".format(iteration))

        # Symmetrized writeback
        ti = np.array([wxx, wyy, wzz, (wyz + wzy) / 2, (wxz + wzx) / 2, (wxy + wyx) / 2])

    #
    # Okey return our results
    return ti, SNormMeanList, DivlevelList, MdpeakList, SmatmeanList, nancountList


def DCS_voigt(tFlat, xsp, ysp, zsp, weighting=False, divergenceThreshold=0.0, symmetryThreshold=1.0, maxIterations=20):
    ti, _, _, _, _, _ = DCS_voigt_internal(tFlat, xsp, ysp, zsp, weighting, divergenceThreshold, symmetryThreshold,
                                           maxIterations)
    """ Calculate DCS of a 3x3 symmetric matrix in Voigt representation using standard convergence criteria """
    return ti
