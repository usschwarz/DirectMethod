#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and plot traction forces for the deformation profile, see README for details

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import sys
import os
import numpy as np

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('macosx')

import matplotlib.pyplot as plt

from src.TFM.uFieldType import load_from_ufile
from src.TFM import tfmFunctions
from src.TFM.resultEval import getQGes, getDipoleDirect, getQiGes
from src.utils.plotLayer import plotLayerWithArrows


def get_tfm_func(name):
    """ Selects and return matching method for TFM mode specification """
    tfm_aliases = {
        "modefit": "modeFitTFM",
        "direct": "directTFM",
        "conefit": "coneFitTFM",
        "4P": "fourPSurfaceTFM",
        "div-removed": "divFreeFitTFM",
        "non-div-removed": "squareFitTFM",
        "squarefit": "squareFitTFM"
    }

    try:
        if name in tfm_aliases:
            return getattr(tfmFunctions, tfm_aliases[name])
        else:
            return getattr(tfmFunctions, name)
    except AttributeError:
        print("TFM type: {} is not supported".format(name))
        raise


def getDiagData(imgLayer, xySpacing, antiDiag=False):
    """ Extracts diagonal data from a grid sampled scalar field

    Params:
        imgLayer - grid sampled scalar field, (2D array, nxn)
        xySpacing - Grid parameters
        antiDiag - If true sample along y=-x instead of y=x

    Returns:
        lParam - Path lengh parametrization of the sampling line at sampling points
        dFeld - scalar field value at sampling points
        secname - String that specifies the sampling line
    """
    assert (imgLayer.shape[0] == imgLayer.shape[1])

    if antiDiag:
        dFeld = np.diag(np.flipud(imgLayer))
        secname = '135degSec'
    else:
        dFeld = np.diag(imgLayer)
        secname = '45degSec'

    xyLen = imgLayer.shape[0] * xySpacing
    lParam = np.linspace(-xyLen / np.sqrt(2), xyLen / np.sqrt(2), len(dFeld))
    return lParam, dFeld, secname


def plotDiagSection(imgLayer, xySpacing, title, antiDiag=False, saveFig=True, folder="plots"):
    """ Plots a diagonal profile of the sampled scalar field

    imgLayer - grid sampled scalar field, (2D array, nxn)
    xySpacing - Grid parameters
    title - Name of the scalar field
    antiDiag - If true sample along y=-x instead of y=x
    saveFig - If set, save plot to disk
    folder - Folder to save plot to.
    """
    plt.rcParams.update({'font.size': 25})

    lParam, dFeld, secname = getDiagData(imgLayer, xySpacing, antiDiag)

    if saveFig:
        # plt.gcf().set_size_inches(4, 3)
        plt.xlabel(r'l/$\mu m$')
        plt.ylabel(r'Traction/Pa')
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

        plt.plot(lParam, dFeld)
        plt.autoscale()
        plt.tight_layout()
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig('{}/{}-{}.pdf'.format(folder, secname, title), bbox_inches='tight')
        plt.close()
        plt.cla()
    else:
        plt.title(title)
        plt.xlabel(r'l/$\mu m$')
        plt.ylabel(r'Traction/Pa')
        # plt.gcf().set_size_inches(1, 1)
        plt.plot(lParam, dFeld)
        plt.show()


def printHelp():
    """ Show cmd line usage help """
    print("Calculates the TFM of a sample.")
    print("")
    print("Usage: runtfm.py <type> <deffield.npz>")
    print("Uses TFM method <type> to calculate the traction forces of the")
    print("deformation profile stipulated in <deffield.npz")
    print("Types:")

    print("direct - Direct method using a simple 2 point derivative")
    print("squarefit - Direct method using a 3x3x3 area fit (recommended)")
    print("div-removed - Direct method using a 3x3x3 area fit and divergence removal")
    print("4P - Direct method using a 4 point derivative")
    print("FTTC - 2d FTTC method")
    print("FTTC3d - 3d FTTC method")
    print("reference - Generate reference deformation file by inspecting 'description.toml'")
    print("")
    print("help   - show this help")


def main():
    """ Main program loop """

    # Part 1: Read in deformation profile
    if len(sys.argv) < 2:
        printHelp()
        exit(0)

    TFM_type = sys.argv[1]

    if TFM_type == "help":
        printHelp()
        exit(0)

    if len(sys.argv) == 2:
        raise RuntimeError("Incorrect argument specification")

    if len(sys.argv) == 3:
        DVCPath = sys.argv[2]
        uCurr = load_from_ufile(DVCPath)

    if len(sys.argv) == 4:
        if sys.argv[2] == "load":
            imgDir = sys.argv[3]
            DVCPath = os.path.join(imgDir, "DVCResult.npz")
            uCurr = load_from_ufile(DVCPath)
        else:
            imgDir = sys.argv[2]
            if imgDir == "-":
                imgDir = ""
            DVCPath = os.path.join(imgDir, sys.argv[3])
            uCurr = load_from_ufile(DVCPath)
    elif len(sys.argv) > 4:
        raise RuntimeError("Incorrect argument specification")

    # Part 2: Calculate TFM

    # Select tfmfunction
    runtfm = get_tfm_func(TFM_type)
    print("Print dm", uCurr.dm)

    # Calculate traction force and surface deformation prediction
    grid, UVec, UzVec, QVec = runtfm(uCurr)
    us, vs, ws = UVec
    qx, qy, qz = QVec

    zeros = np.zeros_like(grid[0])  # Used in some plots

    if runtfm == tfmFunctions.directTFM:
        uAbs = np.sqrt(us * us + vs * vs)

        # This is a syncronization plot used to ensure we have a correct assumption about axis orientation
        plotLayerWithArrows(grid, grid[0], grid[1], np.zeros_like(grid[0]), 'grid', "µm", saveFig=False)

        plotLayerWithArrows(grid, us, vs, uAbs, 'Tangential Displacment [µm] {}'.format(TFM_type), "µm", saveFig=True,
                            savetitle='uT_{}'.format(TFM_type))
        plotLayerWithArrows(grid, zeros, zeros, ws, 'Normal Displacment [µm] {}'.format(TFM_type), "µm", saveFig=True,
                            savetitle='uN_{}'.format(TFM_type))

    plotDiagSection(us, uCurr.dm, 'ux_{}'.format(TFM_type))
    plotDiagSection(vs, uCurr.dm, 'uy_{}'.format(TFM_type))
    plotDiagSection(ws, uCurr.dm, 'uz_{}'.format(TFM_type))

    qAbs = np.sqrt(qx * qx + qy * qy)

    plotLayerWithArrows(grid, qx, qy, qAbs, 'Tangential Stress [Pa] {}'.format(TFM_type), "Pa", saveFig=True,
                        savetitle='qT_{}'.format(TFM_type))
    plotLayerWithArrows(grid, zeros, zeros, qz, 'Normal Stress [Pa] {}'.format(TFM_type), "Pa", saveFig=True,
                        savetitle='qN_{}'.format(TFM_type))

    plotDiagSection(qx, uCurr.dm, 'qx_{}'.format(TFM_type))
    plotDiagSection(qy, uCurr.dm, 'qy_{}'.format(TFM_type))
    plotDiagSection(qz, uCurr.dm, 'p_{}'.format(TFM_type))

    x, y = grid
    Qx, Qy, Qz = getQGes(qx, qy, qz, uCurr.dm, uCurr.dm)
    QzS = getQiGes(qz * qz, uCurr.dm, uCurr.dm)
    mMat, eigval, eigvec = getDipoleDirect(x, y, qx, qy, qz, uCurr.dm, uCurr.dm)
    alpha = np.arctan2(eigvec[1], eigvec[0])

    # Now the superplot

    print("QVector = ({},{},{}) Pa*µm^2".format(Qx, Qy, Qz))
    print("Dipol: ", eigval, eigvec, alpha)
    print("QzS: ", QzS)


exit(main())
