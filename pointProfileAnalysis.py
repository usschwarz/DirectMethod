#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 02:38:22 2019

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import argparse

import os
import numpy as np
import matplotlib.pyplot as plt

from src.eval.focalAdhesionMetrics import get_DTMA_Hertz, get_DTMB_Hertz, get_SNR_Hertz, get_DMA_Hertz
from src.eval.getNoiseLevels import getNoiseLevels
from src.TFM.uFieldType import load_from_ufile
from src.TFM import tfmFunctions as tfmFn
from src.utils import tomlLoad

from src.eval.generateNoised import gen_unnoised_noised


def gen_noised():
    """
    generate a subfolder 'noised' containing deformation data with different noise levels from 'description.toml'
    """
    noiseLevels = np.arange(50) * 0.1
    gen_unnoised_noised(noiseLevels)


def evalMeth(u, meth, pointList):
    """
    Calculate traction field using the method function meth and return the different metrics

    u - Input deformation field
    meth - Used metric
    qfun - Function Ground returning the ground truth traction force triple
    """
    pos, _uVec, _uzVec, qVec = meth(u)
    x, y = pos
    qx, qy, qz = qVec

    DTMA = get_DTMA_Hertz(pointList, x, y, qx, qy, qz)
    DTMB = get_DTMB_Hertz(pointList, x, y, qx, qy, qz)
    SNR = get_SNR_Hertz(pointList, x, y, qx, qy, qz)
    DMA = get_DMA_Hertz(pointList, x, y, qx, qy, qz)

    return DTMA, DTMB, SNR, DMA


def run_nodvctest():
    """ calculate traction fields and extract metrics for all noise levels """
    methods = [tfmFn.squareFitTFM, tfmFn.divFreeFitTFM, tfmFn.FTTC, tfmFn.FTTC3d]

    noiseLevels = getNoiseLevels()

    DTMA = np.empty((len(methods), len(noiseLevels)))
    DTMB = np.empty((len(methods), len(noiseLevels)))
    SNR = np.empty((len(methods), len(noiseLevels)))
    DMA = np.empty((len(methods), len(noiseLevels)))

    pointList = tomlLoad.loadAdheasionSites()

    for i in range(len(noiseLevels)):
        cnoise = noiseLevels[i]
        print("Current noise level:", cnoise)
        noiseFilename = "noised/noise{:d}ppt.npz".format(int(cnoise * 1000))
        # "noise{:.4f}.npz".format(cnoise)

        uN = load_from_ufile(noiseFilename)

        # Load for all methods
        for j, mtd in enumerate(methods):
            DTMA[j, i], DTMB[j, i], SNR[j, i], DMA[j, i] = evalMeth(uN, mtd, pointList)

    print("Saving results to file")
    headertext = "#noise-level\tnodiv\tdiv\tFTTC\tFTTC3d"
    np.savetxt("dtma.txt", np.concatenate(([noiseLevels], DTMA)).T, header=headertext)
    np.savetxt("dtmb.txt", np.concatenate(([noiseLevels], DTMB)).T, header=headertext)
    np.savetxt("snr.txt", np.concatenate(([noiseLevels], SNR)).T, header=headertext)
    np.savetxt("dma.txt", np.concatenate(([noiseLevels], DMA)).T, header=headertext)


def plotStuff_with_legend(noiseLevels, quant, yLable, name=None):
    """
    Plot visualisations of the results, with a legend.
    This plot should only be used to identify lines correctly

    noiselevels - Noise Levels where data has been observed
    quant - 2D array containing mean value for the different plotlines for all noiselevels
    errors - 2D array containing error range for the different plotlines for all noiselevels
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    """
    plt.close()
    ns = noiseLevels

    plt.plot(ns, quant.T)

    plt.xlabel(r"$\sigma_N/<||u||>$")
    plt.ylabel(yLable)

    plt.legend(['3D-DTFM (raw)', '3D-DTFM + DCS', 'FTTC (2d)', 'FTTC (3d)'],
               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)

    plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    plt.gcf().canvas.set_window_title(name)
    plt.show()
    plt.close()


def plotStuff_no_legend(noiseLevels, quant, yLable, name=None, noshow=False, line=True):
    """
    Creates visualisations of the results

    noiselevels - Noise Levels where data has been observed
    quant - 2D array containing values for the different plotlines for all noiselevels
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    noshow - If set, plot is only created (an possibly saved), but not shown in a popup windows
    """
    plt.close()
    fig = plt.figure(figsize=[6.9, 4.5])
    ns = noiseLevels
    ax = fig.add_axes([0.11, 0.14, 0.88, 0.84])

    ax.plot(ns, quant.T)

    ax.set_xlabel(r"$\sigma_N/<||u||>$")
    ax.set_ylabel(yLable)

    if line:
        ax.axhline(0, color="black")

    fig.tight_layout()
    if name is not None:
        fig.savefig('plots/{}.pdf'.format(name))
        fig.canvas.draw_idle()  # need this if 'transparent=True' to reset colors

    if not noshow:
        fig.canvas.set_window_title(name)
        plt.show()
    plt.close()


def plotStuff(noiseLevels, quant, yLable, name=None, withLegend=False, line=True):
    """
    Plot visualisations of the results, with a legend.
    This plot should only be used to identify lines correctly

    noiselevels - Noise Levels where data has been observed
    quant - 2D array containing mean value for the different plotlines for all noiselevels
    errors - 2D array containing error range for the different plotlines for all noiselevels
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    withLegend - If specified, create additional plot that can contains the legend.
    """
    if withLegend:
        if name is not None:
            ename = name + "-withLegend"
        else:
            ename = None
        plotStuff_with_legend(noiseLevels, quant, yLable, ename)
        plotStuff_no_legend(noiseLevels, quant, yLable, name, noshow=True, line=line)
    else:
        plotStuff_no_legend(noiseLevels, quant, yLable, name, line=line)


def plot_nodvctest(showlegend=False):
    """ create plots of the different metrics """
    t1 = np.loadtxt("dtma.txt").T
    noiseLevels = t1[0]

    DTMA = t1[1:]
    DTMB = np.loadtxt("dtmb.txt").T[1:]
    SNR = np.loadtxt("snr.txt").T[1:]
    DMA = np.loadtxt("dma.txt").T[1:]

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    plt.rcParams.update({'font.size': 16})

    plotStuff(noiseLevels, DTMA, "", "DTMA", withLegend=showlegend)
    plotStuff(noiseLevels, DTMB, "", "DTMB")
    plotStuff(noiseLevels, SNR, "", "SNR", line=False)
    plotStuff(noiseLevels, DMA, "", "DMA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Statistically examin profiles of focal adhesions. Profile must be described '
                    'in a "description.toml" file')

    # parser.add_argument('--foo', action='store_true', help='foo help')
    subparsers = parser.add_subparsers(dest='option', help='sub-command help')

    # create the parser for the "a" command
    parser_gen = subparsers.add_parser('gen', help='Generate noisy profiles')
    parser_calc = subparsers.add_parser('calc', help='Calculates results')
    parser_plot = subparsers.add_parser('plot', help='Plot results')
    parser_plot_leg = subparsers.add_parser('plot_leg', help='Plot results (Show legend on first plot)')
    parser_run = subparsers.add_parser('all', help='Do all of the above in one step')
    args = parser.parse_args()

    if args.option == "calc":
        run_nodvctest()
    elif args.option == "plot":
        plot_nodvctest()
    elif args.option == "plot_leg":
        plot_nodvctest(True)
    elif args.option == "gen":
        gen_noised()
    elif args.option == "all":
        gen_noised()
        run_nodvctest()
        plot_nodvctest(True)
    else:
        parser.print_usage()
