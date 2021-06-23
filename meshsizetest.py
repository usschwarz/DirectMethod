#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 02:38:22 2019

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.eval.focalAdhesionMetrics import get_DTMA_Hertz, get_DTMB_Hertz, get_SNR_Hertz, get_DMA_Hertz
from src.TFM.uFieldType import load_from_ufile
from src.TFM import tfmFunctions as tfmFn
from src.utils import tomlLoad
from src.inputSim.fields.HertzBuilder import get_u_hertz_pattern
from src.inputSim.uFieldSimulation import dumpu2npz


def evalMeth(u, meth, pointList):
    """
    Calculate traction field using the method function meth and return the different metrics

    u - Input deformation field
    meth - Used metric
    pointList - Ground truth about the ahesion sites in the point list format (defined in utils/tomlLoad)
    """
    pos, _uVec, _uzVec, qVec = meth(u)
    x, y = pos
    qx, qy, qz = qVec

    DTMA = get_DTMA_Hertz(pointList, x, y, qx, qy, qz)
    DTMB = get_DTMB_Hertz(pointList, x, y, qx, qy, qz)
    SNR = get_SNR_Hertz(pointList, x, y, qx, qy, qz)
    DMA = get_DMA_Hertz(pointList, x, y, qx, qy, qz)

    return DTMA, DTMB, SNR, DMA


def build_u_file(E, nu, uFun, xyLen, xyPix0, dLayers0, nLayers, factor, index):
    """
    Builds an u file using the mesh configuration specified

    E - Youngs modulus
    nu - Poissons ratio
    uFun - Function to get deformation tripel (ux,uy,uz) at any point in space
    xyPix0 - Normal distance factor
    """
    nLayers = int(nLayers)
    xCnt = int(xyPix0 / factor)
    zSep = dLayers0 * factor

    xyRng = np.linspace(-xyLen / 2, xyLen / 2, xCnt)
    zRng = zSep * np.arange(nLayers)

    xySep = xyRng[1] - xyRng[0]

    print("Range done")
    Y, Z = np.meshgrid(xyRng, zRng, indexing='ij')
    shape = (len(xyRng), len(xyRng), nLayers)

    print("Shape:", shape)

    u = np.empty(shape)
    v = np.empty(shape)
    w = np.empty(shape)

    print("Mesh done")
    for i in range(len(xyRng)):
        print("Doing Row", i)
        u[i, :, :], v[i, :, :], w[i, :, :] = uFun(xyRng[i], Y, Z)
    print("Simulated shape {}".format(u.shape))

    uAbs = np.sqrt(u * u + v * v + w * w)
    cc = 0
    m = [xyRng, xyRng, zRng]

    dumpu2npz("meshtest-{}.npz".format(index), u, v, w, uAbs, cc, zSep, xySep, m, E, nu, 0)


def prep_meshtest():
    """ generate deformation data with different mesh sizes using 'description.toml'  """
    fname, E, nu, micronsPerPixel0, dLayers0 = tomlLoad.loadDataDescription()
    nLayers, xyPix0, _ = tomlLoad.loadSimulationData(silent=True)
    pointList = tomlLoad.loadAdheasionSites(silent=True)

    xyPixMin, xyPixMax = tomlLoad.loadMeshSizeRange(silent=True)

    factors = np.logspace(np.log10(xyPix0 / xyPixMax), np.log10(xyPix0 / xyPixMin), 10)

    xyLen = micronsPerPixel0 * xyPix0

    uf, vf, wf = get_u_hertz_pattern(pointList)

    def uFun(x, y, z):
        """ Vectorized u function """
        mu = E / (2.0 * (1.0 + nu))
        return uf(x, y, z, mu, nu), vf(x, y, z, mu, nu), wf(x, y, z, mu, nu)

    for i, factor in enumerate(factors):
        build_u_file(E, nu, uFun, xyLen, xyPix0, dLayers0, nLayers, 8 * factor, i)
        print("Current factor:", factor)


def calc_meshtest():
    """ calculate traction fields and extract metrics for all mesh sizes """
    _, xyPix0, _ = tomlLoad.loadSimulationData(silent=True)
    xyPixMin, xyPixMax = tomlLoad.loadMeshSizeRange(silent=True)

    factors = np.logspace(np.log10(xyPix0 / xyPixMax), np.log10(xyPix0 / xyPixMin), 10)
    methods = [tfmFn.squareFitTFM, tfmFn.divFreeFitTFM, tfmFn.FTTC]  # ,tfmfn.FTTC3d]

    DTMA = np.empty((len(methods), len(factors)))
    DTMB = np.empty((len(methods), len(factors)))
    SNR = np.empty((len(methods), len(factors)))
    DMA = np.empty((len(methods), len(factors)))

    pointList = tomlLoad.loadAdheasionSites(silent=True)

    for i, factor in enumerate(factors):
        uN = load_from_ufile("meshtest-{}.npz".format(i))
        # Load for all methods
        for j, mtd in enumerate(methods):
            DTMA[j, i], DTMB[j, i], SNR[j, i], DMA[j, i] = evalMeth(uN, mtd, pointList)

    print("Saving results to file")
    headertext = "#factors\tnodiv\tdiv\tunregFTTC\tFTTC"
    np.savetxt("dtma-meshtest.txt", np.concatenate(([factors], DTMA)).T, header=headertext)
    np.savetxt("dtmb-meshtest.txt", np.concatenate(([factors], DTMB)).T, header=headertext)
    np.savetxt("snr-meshtest.txt", np.concatenate(([factors], SNR)).T, header=headertext)
    np.savetxt("dma-meshtest.txt", np.concatenate(([factors], DMA)).T, header=headertext)


def plotStuff_with_legend(factors, quant, yLable, name=None):
    """
    Plot visualisations of the results, with a legend.
    This plot should only be used to identify lines correctly

    factors - Mesh size factors used
    quant - 2D array containing mean value for the different plotlines for all factors
    errors - 2D array containing error range for the different plotlines for all factors
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    """
    plt.close()

    plt.plot(factors, quant.T)
    plt.xlabel("mesh_size/default mesh_size")
    plt.xscale("log")
    plt.gca().xaxis.set_major_locator(mticker.LogLocator(subs='all'))
    plt.gca().xaxis.set_minor_locator(mticker.NullLocator())
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.ylabel(yLable)

    plt.legend(['3D-DTFM (raw)', '3D-DTFM + DCS', 'FTTC (2d)'],
               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    plt.gcf().canvas.set_window_title(name)
    plt.show()
    plt.close()


def plotStuff_no_legend(factors, quant, yLable, name=None, noshow=False, line=False):
    """
    Creates visualisations of the results

    factors - Mesh size factors used
    quant - 2D array containing mean value for the different plotlines for all factors
    errors - 2D array containing error range for the different plotlines for all factors
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    noshow - If set, plot is only created (an possibly saved), but not shown in a popup windows
    """
    plt.close()

    plt.plot(factors, quant.T)
    plt.xlabel("mesh_size/default mesh_size")
    plt.xscale("log")
    plt.gca().xaxis.set_major_locator(mticker.LogLocator(subs='all'))
    plt.gca().xaxis.set_minor_locator(mticker.NullLocator())
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.ylabel(yLable)

    if line:
        plt.gca().axhline(0, color="black")

    plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    if not noshow:
        plt.gcf().canvas.set_window_title(name)
        plt.show()
    plt.close()


def plotStuff(factors, quant, yLable, name=None, withLegend=False, line=True):
    """
    Plot visualisations of the results, with a legend.
    This plot should only be used to identify lines correctly

    factors - Mesh size factors used
    quant - 2D array containing mean value for the different plotlines for all factors
    errors - 2D array containing error range for the different plotlines for all factors
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    withLegend - If specified, create additional plot that can contains the legend.
    """
    if withLegend:
        if name is not None:
            ename = name + "-withLegend"
        else:
            ename = None
        plotStuff_with_legend(factors, quant, yLable, ename)
        plotStuff_no_legend(factors, quant, yLable, name, noshow=True, line=line)
    else:
        plotStuff_no_legend(factors, quant, yLable, name, line=line)


def plot_meshtest():
    """ create plots of the different metrics """
    t1 = np.loadtxt("dtma-meshtest.txt").T
    factors = t1[0]

    DTMA = t1[1:]
    DTMB = np.loadtxt("dtmb-meshtest.txt").T[1:]
    SNR = np.loadtxt("snr-meshtest.txt").T[1:]
    DMA = np.loadtxt("dma-meshtest.txt").T[1:]

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    plt.rcParams.update({'font.size': 16})

    plotStuff(factors, DTMA, "", "DTMA", withLegend=True)
    plotStuff(factors, DTMB, "", "DTMB")
    plotStuff(factors, SNR, "", "SNR", line=False)
    plotStuff(factors, DMA, "", "DMA")


# This complex argument has to catch both cases
if (sys.argv[1] == "sim_full") if len(sys.argv) >= 2 else True:
    prep_meshtest()
    calc_meshtest()
elif sys.argv[1] == "gen":
    prep_meshtest()
elif sys.argv[1] == "sim":
    calc_meshtest()
elif sys.argv[1] == "plot":
    plot_meshtest()
elif sys.argv[1] == "all":
    prep_meshtest()
    calc_meshtest()
    plot_meshtest()
else:
    raise RuntimeError("Optition {} not recognised.".format(sys.argv[1]))
