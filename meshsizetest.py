#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses the effect of changing the mesh size using the analysis for point like
tangential profiles.
'description.toml' should specify a profil consisting only of non-overlapping tangential adhesions

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.eval.focalAdhesionMetrics import get_DTMA_Hertz, get_DTMB_Hertz, get_SNR_Hertz
from src.eval.focalAdhesionMetrics import get_DMA_Hertz, get_DTMA2D_Hertz, get_DMA2D_Hertz
from src.TFM.uFieldType import load_from_ufile
from src.TFM import tfmFunctions as tfmFn
from src.utils import tomlLoad
from src.inputSim.fields.HertzBuilder import get_u_hertz_pattern
from src.inputSim.uFieldSimulation import dumpu2npz

methods = [tfmFn.squareFitTFM, tfmFn.divFreeFitTFM, tfmFn.FTTC, tfmFn.FTTC3d]
methodfilenames = ["squarefit", "divrem", "fttc2d", "fttc3d"]

# substuple = (1., 1.3, 1.6, 2, 2.5, 3., 4., 5., 6., 8.)

substuple = (1., 1.2, 1.4, 1.6, 1.8, 2, 3., 4., 5., 6., 7., 8., 9.)


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


def _get_factors():
    n_points_xy_0, _ = tomlLoad.loadSimulationData(silent=True)
    n_points_xy_min, n_points_xy_max = tomlLoad.loadMeshSizeRange(silent=True)
    return np.logspace(np.log10(n_points_xy_0 / n_points_xy_max), np.log10(n_points_xy_0 / n_points_xy_min), 10)


def build_u_file(E, nu, uFun, xyLen, xyPix0, dLayers0, nLayers, factor, index):
    """
    Builds an u file using the mesh configuration specified

    E - Youngs modulus
    nu - Poissons ratio
    uFun - Function to get deformation tripel (ux,uy,uz) at any point in space
    xyLen - Size of the images field of view in x-y-direction
    n_points_xy - Number of grid points in x-y-direction in the reference configuration
    dLayers - Reference  spacing between different layers
    n_points_z - Number of layers in z-direction generated
    factor - scaling factor for the mesh
    index - Number affixed to the generated output file
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
    fname, E, nu, spacing_xy_0, spacing_z_0 = tomlLoad.loadDataDescription()

    n_points_xy_0, n_points_z = tomlLoad.loadSimulationData(silent=True)
    pointList = tomlLoad.loadAdheasionSites(silent=True)

    factors = _get_factors()

    xyLen = spacing_xy_0 * n_points_xy_0

    uf, vf, wf = get_u_hertz_pattern(pointList)

    def uFun(x, y, z):
        """ Vectorized u function """
        mu = E / (2.0 * (1.0 + nu))
        return uf(x, y, z, mu, nu), vf(x, y, z, mu, nu), wf(x, y, z, mu, nu)

    for i, factor in enumerate(factors):
        build_u_file(E, nu, uFun, xyLen, n_points_xy_0, spacing_z_0, n_points_z, factor, i)
        print("Current factor:", factor)


def calc_field(outfile_dir):
    """ calculate traction fields for all mesh sizes and save them to outfile_dir """

    if not os.path.isdir(outfile_dir):
        os.mkdir(outfile_dir)

    factors = _get_factors()

    for i, factor in enumerate(factors):
        uN = load_from_ufile("meshtest-{}.npz".format(i))
        # Load for all methods
        for j, (mtdname, mtd) in enumerate(zip(methodfilenames, methods)):
            print(f"Building reconst_f_field_meshfactor_nr{i}_{mtdname}.npz")
            outfile_name = os.path.join(
                outfile_dir, f"reconst_f_field_meshfactor_nr{i}_{mtdname}.npz"
            )
            pos, uVec, uzVec, qVec = mtd(uN)

            np.savez(outfile_name, pos=pos, uVec=uVec, uzVec=uzVec, qVec=qVec)


def calc_meshtest_from_saved_f_files(f_file_dir):
    """ load reconstructed fields from f_file_dir and extract metrics for all mesh sizes """
    factors = _get_factors()

    DTMA = np.empty((len(methods), len(factors)))
    DTMB = np.empty((len(methods), len(factors)))
    SNR = np.empty((len(methods), len(factors)))
    DMA = np.empty((len(methods), len(factors)))

    DTMA2D = np.empty((len(methods), len(factors)))
    DMA2D = np.empty((len(methods), len(factors)))

    pointList = tomlLoad.loadAdheasionSites(silent=True)

    for i, factor in enumerate(factors):

        # Load for all methods
        for j, (mtdname, mtd) in enumerate(zip(methodfilenames, methods)):
            infile_name = os.path.join(
                f_file_dir, f"reconst_f_field_meshfactor_nr{i}_{mtdname}.npz"
            )
            DTMA[j, i], DTMB[j, i], SNR[j, i], DMA[j, i], DTMA2D[j, i], DMA2D[j, i] =\
                evaluate_method(infile_name, pointList)

    print("Saving results to file")
    headertext = "#factors\tnodiv\tdiv\tunregFTTC\tFTTC"
    np.savetxt("dtma-meshtest.txt", np.concatenate(([factors], DTMA)).T, header=headertext)
    np.savetxt("dtmb-meshtest.txt", np.concatenate(([factors], DTMB)).T, header=headertext)
    np.savetxt("snr-meshtest.txt", np.concatenate(([factors], SNR)).T, header=headertext)
    np.savetxt("dma-meshtest.txt", np.concatenate(([factors], DMA)).T, header=headertext)
    np.savetxt("dtma2d-meshtest.txt", np.concatenate(([factors], DTMA2D)).T, header=headertext)
    np.savetxt("dma2d-meshtest.txt", np.concatenate(([factors], DMA2D)).T, header=headertext)


def calc_meshtest(use_cached=False):
    """ calculate traction fields and extract metrics for all mesh sizes """

    rec_folder = "reconstructed_traction"

    if not use_cached:
        calc_field(rec_folder)
    calc_meshtest_from_saved_f_files(rec_folder)


def plotStuff_with_legend(xy_spacing, quant, yLable, name=None):
    """
    Plot visualisations of the results, with a legend.
    This plot should only be used to identify lines correctly

    xy_spacing - Mesh spacing used (xy)
    quant - 2D array containing mean value for the different plotlines for all factors
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    """
    plt.close()

    # plt.plot(factors, quant.T)
    # plt.xlabel("mesh scaling factor")
    plt.plot(xy_spacing, quant.T)
    plt.xscale("log")

    plt.gca().xaxis.set_major_locator(mticker.LogLocator(base=10, subs=substuple)) # 'all'))
    plt.gca().xaxis.set_minor_locator(mticker.NullLocator())
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.ylabel(yLable)

    plt.legend(['3D-DTFM (raw)', '3D-DTFM + DCS', 'FTTC (2d)', 'FTTC (3D)'],
               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    plt.gcf().canvas.set_window_title(name)
    plt.show()
    plt.close()


def plotStuff_no_legend(xy_spacing, quant, yLable, name=None, noshow=False, line=False):
    """
    Creates visualisations of the results

    xy_spacing - Mesh spacing used (xy)
    quant - 2D array containing mean value for the different plotlines for all factors
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    noshow - If set, plot is only created (an possibly saved), but not shown in a popup windows
    line - If set plot a reference line at x==0
    """
    plt.close()

    # plt.plot(factors, quant.T)
    # plt.xlabel("mesh scaling factor")
    plt.plot(xy_spacing, quant.T)
    plt.xlabel("xy spacing µm")
    plt.xscale("log")
    plt.gca().xaxis.set_major_locator(mticker.LogLocator(subs=substuple))  # 'all'))
    plt.gca().xaxis.set_minor_locator(mticker.NullLocator())
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.ylabel(yLable)

    if line:
        plt.gca().axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    if not noshow:
        plt.gcf().canvas.set_window_title(name)
        plt.show()
    plt.close()


def plotStuff(xy_spacing, quant, yLable, name=None, withLegend=False, line=True):
    """
    Plot visualisations of the results, with a legend.
    This plot should only be used to identify lines correctly

    xy_spacing - Mesh spacing used (xy)
    quant - 2D array containing mean value for the different plotlines for all factors
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    withLegend - If specified, create additional plot that can contains the legend.
    line - If set plot a reference line at x==0
    """
    if withLegend:
        if name is not None:
            ename = name + "-withLegend"
        else:
            ename = None
        plotStuff_with_legend(xy_spacing, quant, yLable, ename)
        plotStuff_no_legend(xy_spacing, quant, yLable, name, noshow=True, line=line)
    else:
        plotStuff_no_legend(xy_spacing, quant, yLable, name, line=line)


def plot_meshtest(showlegend=False):
    """ create plots of the different metrics """
    _, _, _, spacing_xy_0, _ = tomlLoad.loadDataDescription()

    t1 = np.loadtxt("dtma-meshtest.txt").T
    xy_spacing = t1[0] * spacing_xy_0

    DTMA = t1[1:]
    DTMB = np.loadtxt("dtmb-meshtest.txt").T[1:]
    SNR = np.loadtxt("snr-meshtest.txt").T[1:]
    DMA = np.loadtxt("dma-meshtest.txt").T[1:]
    DTMA2D = np.loadtxt("dtma2d-meshtest.txt").T[1:]
    DMA2D = np.loadtxt("dma2d-meshtest.txt").T[1:]

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    plt.rcParams.update({'font.size': 16})

    plotStuff(xy_spacing, DTMA, "", "DTMA", withLegend=showlegend)
    plotStuff(xy_spacing, DTMB, "", "DTMB")
    plotStuff(xy_spacing, SNR, "", "SNR", line=False)
    plotStuff(xy_spacing, DMA, "", "DMA")
    plotStuff(xy_spacing, DTMA2D, "", "DTMA2D")
    plotStuff(xy_spacing, DMA2D, "", "DMA2D")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Statistically examin arbitry profiles. '
                    + 'Profile must be described in a "description.toml" file'
    )

    # Add help commands
    subparsers = parser.add_subparsers(dest='option', help='sub-command help')

    # Add other commands
    parser_calc = subparsers.add_parser('calc', help='Calculates results')
    parser_plot = subparsers.add_parser('plot', help='Plot results')
    parser_gen = subparsers.add_parser('gen', help='Generate noisy profiles')
    parser_run = subparsers.add_parser('all', help='Calculates, then plots results')

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
        calc_meshtest(use_cached=args.use_cached)
    elif args.option == "plot":
        plot_meshtest(args.show_legend_plot)
    elif args.option == "gen":
        prep_meshtest()
    elif args.option == "all":
        if not args.use_cached:
            prep_meshtest()
        calc_meshtest(use_cached=args.use_cached)
        plot_meshtest(args.show_legend_plot)
    else:
        parser.print_usage()
