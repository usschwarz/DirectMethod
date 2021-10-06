#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses the effect of noise on different TFM strategies using the analysis for point like
tangential profiles.
'description.toml' should specify a profil consisting only of non-overlapping tangential adhesions

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.eval.focalAdhesionMetrics import get_DTMA_Hertz, get_DTMB_Hertz, get_SNR_Hertz
from src.eval.focalAdhesionMetrics import get_DMA_Hertz, get_DTMA2D_Hertz, get_DMA2D_Hertz

from src.eval.getNoiseLevels import getNoiseLevels
from src.TFM.uFieldType import load_from_ufile
from src.TFM import tfmFunctions as tfmFn
from src.utils import tomlLoad

from src.eval.generateNoised import gen_unnoised_noised

methods = [tfmFn.squareFitTFM, tfmFn.divFreeFitTFM, tfmFn.FTTC, tfmFn.FTTC3d]
methodfilenames = ["squarefit", "divrem", "fttc2d", "fttc3d"]


def gen_noised():
    """
    generate a subfolder 'noised' containing deformation data with different noise
    levels from 'description.toml'
    """
    noiseLevels = np.arange(40) * 0.1
    gen_unnoised_noised(noiseLevels)


def evaluate_method(f_file, pointList):
    """
    Load result traction field, compare with ground truth and return the different metrics

    f_file - file containing traction data
    pointList - Ground truth about the ahesion sites (format defined in utils/tomlLoad)
    """
    loaddata = np.load(f_file)

    pos = loaddata['pos']
    qVec = loaddata['qVec']

    x, y = pos
    qx, qy, qz = qVec

    DTMA = get_DTMA_Hertz(pointList, x, y, qx, qy, qz)
    DTMB = get_DTMB_Hertz(pointList, x, y, qx, qy, qz)
    SNR = get_SNR_Hertz(pointList, x, y, qx, qy, qz)
    DMA = get_DMA_Hertz(pointList, x, y, qx, qy, qz)
    DTMA2D = get_DTMA2D_Hertz(pointList, x, y, qx, qy, qz)
    DMA2D = get_DMA2D_Hertz(pointList, x, y, qx, qy, qz)

    return DTMA, DTMB, SNR, DMA, DTMA2D, DMA2D


def calc_fields(outfile_dir):
    """ calculate traction fields for all noise levels and saves them to outfile_dir"""

    if not os.path.isdir(outfile_dir):
        os.mkdir(outfile_dir)

    noiseLevels = getNoiseLevels()

    for i in range(len(noiseLevels)):
        cnoise = noiseLevels[i]
        print("Current noise level:", cnoise)
        noiseFilename = "noised/noise{:d}ppt.npz".format(int(cnoise * 1000))
        # "noise{:.4f}.npz".format(cnoise)

        uN = load_from_ufile(noiseFilename)

        # Load for all methods
        for j, (mtdname, mtd) in enumerate(zip(methodfilenames, methods)):
            outfile_name = os.path.join(
                outfile_dir, f"reconst_f_field_nlidx{i}_{mtdname}.npz"
            )
            pos, uVec, uzVec, qVec = mtd(uN)

            np.savez(outfile_name, pos=pos, uVec=uVec, uzVec=uzVec, qVec=qVec)


def calc_from_saved_f_files(f_file_dir):
    """ load reconstructed fields from f_file_dir and extract metrics for all noise levels """
    noiseLevels = getNoiseLevels()

    DTMA = np.empty((len(methods), len(noiseLevels)))
    DTMB = np.empty((len(methods), len(noiseLevels)))
    SNR = np.empty((len(methods), len(noiseLevels)))
    DMA = np.empty((len(methods), len(noiseLevels)))

    DTMA2D = np.empty((len(methods), len(noiseLevels)))
    DMA2D = np.empty((len(methods), len(noiseLevels)))

    pointList = tomlLoad.loadAdheasionSites()

    for i in range(len(noiseLevels)):
        cnoise = noiseLevels[i]

        # Load for all methods
        for j, (mtdname, mtd) in enumerate(zip(methodfilenames, methods)):
            infile_name = os.path.join(
                f_file_dir, f"reconst_f_field_nlidx{i}_{mtdname}.npz"
            )
            DTMA[j, i], DTMB[j, i], SNR[j, i], DMA[j, i], DTMA2D[j, i], DMA2D[j, i] =\
                evaluate_method(infile_name, pointList)

    print("Saving results to file")
    headertext = "#noise-level\tnodiv\tdiv\tFTTC\tFTTC3d"
    np.savetxt("dtma.txt", np.concatenate(([noiseLevels], DTMA)).T, header=headertext)
    np.savetxt("dtmb.txt", np.concatenate(([noiseLevels], DTMB)).T, header=headertext)
    np.savetxt("snr.txt", np.concatenate(([noiseLevels], SNR)).T, header=headertext)
    np.savetxt("dma.txt", np.concatenate(([noiseLevels], DMA)).T, header=headertext)
    np.savetxt("dtma2d.txt", np.concatenate(([noiseLevels], DTMA2D)).T, header=headertext)
    np.savetxt("dma2d.txt", np.concatenate(([noiseLevels], DMA2D)).T, header=headertext)


def calc_all(use_cached=False):
    """ calculate traction fields and extract metrics for all noise levels """
    rec_folder = "reconstructed_traction"

    if not use_cached:
        calc_fields(rec_folder)
    calc_from_saved_f_files(rec_folder)


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
        ax.axhline(0, color="black", linewidth=0.5)

    # fig.tight_layout()
    if name is not None:
        fig.savefig('plots/{}.pdf'.format(name))
        fig.canvas.draw_idle()  # need this if 'transparent=True' to reset colors

    if not noshow:
        fig.canvas.set_window_title(name)
        plt.show()
    plt.close()


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

    # plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    plt.gcf().canvas.set_window_title(name)
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
    DTMA2D = np.loadtxt("dtma2d.txt").T[1:]
    DMA2D = np.loadtxt("dma2d.txt").T[1:]

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    plt.rcParams.update({'font.size': 16})

    plotStuff(noiseLevels, DTMA, "", "DTMA", withLegend=showlegend)
    plotStuff(noiseLevels, DTMB, "", "DTMB")
    plotStuff(noiseLevels, SNR, "", "SNR", line=False)
    plotStuff(noiseLevels, DMA, "", "DMA")
    plotStuff(noiseLevels, DTMA2D, "", "DTMA2D")
    plotStuff(noiseLevels, DMA2D, "", "DMA2D")


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
    parser_run = subparsers.add_parser('all', help='Do all of the above in one step')

    for parser_x in [parser_plot, parser_run]:
        parser_x.add_argument(
            "--show-legend-plot", action='store_true', help='Show and save first plot with legend'
        )

    for parser_y in [parser_run, parser_calc]:
        parser_y.add_argument(
            "--use-cached", action='store_true', help='Use cached deformation field files'
        )

    args = parser.parse_args()

    # plt.style.use('fivethirtyeight')  # Select color style
    # plt.rcParams['figure.facecolor'] = 'white'
    # plt.rcParams['axes.facecolor'] = 'white'
    # plt.rcParams['axes.edgecolor'] = 'white'
    # plt.rcParams['savefig.facecolor'] = 'white'
    # plt.rcParams['savefig.edgecolor'] = 'white'

    # Switch color cycle
    # Scip C3 (red)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
        color=['#1f77b4', 'ff7f0e', '#2ca02c', '#9467bd',  # '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    )

    # # Alternative mode
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    #     color=["#ff1f5b", "#009adf", "#af58ba", "#00cd6c", "#ffc61e", "#f28522"]
    # )

    if args.option == "calc":
        calc_all(use_cached=args.use_cached)
    elif args.option == "plot":
        plot_nodvctest(args.show_legend_plot)
    elif args.option == "gen":
        gen_noised()
    elif args.option == "all":
        if not args.use_cached:
            gen_noised()
        calc_all(use_cached=args.use_cached)
        plot_nodvctest(args.show_legend_plot)
    else:
        parser.print_usage()
