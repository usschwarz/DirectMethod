#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses the effect of noise on different TFM strategies using the analysis for normal profiles
This version creates box_plot rather them banded plots

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from src.eval.generateNoised import gen_unnoised_noised
from src.eval.fieldProperties import getQGes, getl2Diff, get_bg_noise_level
from src.TFM import tfmFunctions as tfmFn
from src.TFM.uFieldType import load_from_ufile

from src.utils import tomlLoad
from src.utils.figure_setup import figure_setup
from src.inputSim.fields import HertzBuilder

pickle_name = "newNormalAnalysis.pickle"
sample_count = 10
# noiseLevels = np.arange(10) * 0.5
noiseLevels = np.array([1, 2, 3, 4])
noiseLevelNames = ['low', 'medium-low', 'medium-high', 'high']
# noiseLevels = np.array([1, 2, 5])
# noiseLevelNames = ['low', 'medium', 'high']

methods = [
    tfmFn.directTFM, tfmFn.squareFitTFM, tfmFn.divFreeFitTFM,
    tfmFn.fourPSurfaceTFM, tfmFn.reference
]
methodnames = ["direct", "square fit", "div rem.", "4th ord. poly.", "reference"]
methodfilenames = ["direct", "squarefit", "divrem", "fourth_order_poly", "reference"]


def evaluate_method(u, f_file, qfun):
    """
    Load result traction field, compare with ground truth return the different metrics

    u - Input deformation field
    f_file - file containing traction data
    qfun - Function Ground returning the ground truth traction force triple
    """
    loaddata = np.load(f_file)

    pos = loaddata['pos']
    qVec = loaddata['qVec']

    x, y = pos
    qx, qy, qz = qVec
    Qx, Qy, Qz = getQGes(qx, qy, qz, u.dm, u.dm)

    qxRfun, qyRfun, qzRfun = qfun
    qxR = qxRfun(x, y)
    qyR = qyRfun(x, y)
    qzR = qzRfun(x, y)

    QL2 = getl2Diff(qx, qy, qz, qxR, qyR, qzR, u.dm, u.dm)
    bgn = get_bg_noise_level(qx, qy, qz, qxR, qyR, qzR)
    return Qx, Qy, Qz, QL2, bgn


def _set_box_color(bp, color):
    """
    Fixes color of a boxplot

    bp - Boxplot to color
    color -color to use
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], markeredgecolor=color)



def plot_with_errorbars(
        noiselevels, quant, errors, yLable, name=None, withLegend=False, noshow=False
):
    """
    Create a visualization of the result using simple point marks
    This should only be used in case of non overlapping datasets

    noiselevels - Noise Levels where data has been observed
    quant - 2D array containing mean value for the different plotlines for all noiselevels
    errors - 2D array containing error range for the different plotlines for all noiselevels
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    withLegend - If specified, create additional plot that can contains the legend.
    """

    plt.close()
    fig, ax = figure_setup()
    ns = noiselevels

    # Draw reference line if needed:
    yref = quant[-1, 0]
    ax.axhline(yref, color="black", linewidth=0.5)

    for i, (qt, er) in enumerate(zip(quant[:-1], errors[:-1])):
        ax.errorbar(x=ns, y=qt, yerr=er, fmt=':', ls="-", lw=0.5, marker="x", capsize=3, capthick=1)
        #plt.plot(ns, qt)
        # plt.fill_between(ns, qt - er, qt + er, alpha=0.3)

    ax.set_xlabel(r"$\sigma_N/<||u||>$")
    ax.set_ylabel(yLable)

    if withLegend:
        plt.legend()
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #            ncol=2, mode="expand", borderaxespad=0.)

    # plt.tight_layout()
    if name is not None:
        fig.savefig('plots/{}.pdf'.format(name))

    if not noshow:
        fig.canvas.set_window_title(name)
        plt.show()
    plt.close()


def plot_with_boxplots(
        noiselevels, data, yLable, name=None, withLegend=False, noshow=False
):
    """
    Create a visualization of the results using boxplots
    This should be used for overlapping datasets

    noiselevels - Noise Levels where data has been observed
    data - 3D array containing sampled data for the different
            plotlines for all noiselevels (method, noiseleve, sample)
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    withLegend - If specified, create additional plot that can contains the legend.
    """
    assert len(methodnames) == data.shape[0]
    assert len(noiselevels) == data.shape[1]

    plt.close()
    fig, ax = figure_setup()

    # Get a list of all colors
    colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    n_colors = len(colors)

    def _get_color(num):
        return colors[num % n_colors]

    base_range = np.arange(len(noiselevels)) * (data.shape[0] + 2)
    offset = np.arange(data.shape[0] - 1) - (data.shape[0] - 2) / 2

    bplist = []

    xmin = offset[0] - 1
    xmax = len(noiselevels) * (data.shape[0] + 2) - (data.shape[0] - 1) / 2

    # Draw reference line if needed:
    yref = data[-1, 0, 0]
    ax.axhline(y=yref, xmin=xmin, xmax=xmax, c='k', linewidth=0.5)

    for i, _ in enumerate(methodnames[:-1]):
        bp = ax.boxplot(data[i].T, positions=base_range + offset[i])
        _set_box_color(bp, _get_color(i))
        bplist.append(bp)

    # Draw fictional lines to generate legend entries
    if withLegend:
        for i, meth in enumerate(methodnames[:-1]):
            ax.plot([], c=_get_color(i), label=meth)

    ax.set_xticks(base_range)
    ax.set_xticklabels(noiseLevels)
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(r"$\sigma_N/<||u||>$")
    ax.set_ylabel(yLable)

    if withLegend:
        plt.legend()
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #            ncol=2, mode="expand", borderaxespad=0.)

    if name is not None:
        fig.savefig('plots/{}.pdf'.format(name))

    if not noshow:
        fig.canvas.set_window_title(name)
        plt.show()
    plt.close()


def calc_fields(outfile_dir):
    """ calculate traction fields for all noise levels and save them to outfile_dir """

    if not os.path.isdir(outfile_dir):
        os.mkdir(outfile_dir)

    for i in range(len(noiseLevels)):
        cnoise = noiseLevels[i]

        for k in range(sample_count):
            print(f"Current noise level: {cnoise}, current sample {k}")
            noiseFilename = "noised/noise{:d}ppt-{}.npz".format(int(cnoise * 1000), k)
            uN = load_from_ufile(noiseFilename)

            # Load for all methods
            for j, (mtdname, mtd) in enumerate(zip(methodfilenames, methods)):
                outfile_name = os.path.join(
                    outfile_dir, f"reconst_f_field_nl{i}_smpl{k}_{mtdname}.npz"
                )
                pos, uVec, uzVec, qVec = mtd(uN)

                np.savez(outfile_name, pos=pos, uVec=uVec, uzVec=uzVec, qVec=qVec)


def calc_from_saved_f_files(f_file_dir):
    """ extract metrics for all noise levels """
    Qx = np.empty((len(methods), len(noiseLevels), sample_count))
    Qy = np.empty((len(methods), len(noiseLevels), sample_count))
    Qz = np.empty((len(methods), len(noiseLevels), sample_count))
    QL2 = np.empty((len(methods), len(noiseLevels), sample_count))

    BGN = np.empty((len(methods), len(noiseLevels), sample_count))

    pointlist = tomlLoad.loadAdheasionSites(silent=True)
    qfun = HertzBuilder.get_q_hertz_pattern(pointlist)

    for i in range(len(noiseLevels)):
        cnoise = noiseLevels[i]

        for k in range(sample_count):
            print(f"Current noise level: {cnoise}, current sample {k}")
            noiseFilename = "noised/noise{:d}ppt-{}.npz".format(int(cnoise * 1000), k)
            # "noise{:.4f}.npz".format(cnoise)

            uN = load_from_ufile(noiseFilename)
            # rFile = "refData.npz"

            # Load for all methods
            for j, (mtdname, mtd) in enumerate(zip(methodfilenames, methods)):
                infile_name = os.path.join(
                    f_file_dir, f"reconst_f_field_nl{i}_smpl{k}_{mtdname}.npz"
                )
                Qx[j, i, k], Qy[j, i, k], Qz[j, i, k], QL2[j, i, k], BGN[j, i, k] = \
                    evaluate_method(uN, infile_name, qfun)

    print("Saving results to file")
    with open(pickle_name, "wb") as fd:
        dumpTuple = Qx, Qy, Qz, QL2, BGN
        pickle.dump(dumpTuple, fd)


def calc_all(use_cached=False):
    """ calculate traction fields and extract metrics for all noise levels """
    rec_folder = "reconstructed_traction"

    if not use_cached:
        calc_fields(rec_folder)
    calc_from_saved_f_files(rec_folder)


def dostat_1D(quant):
    """ Does statistics on scalar results

    Use dataset "quant" containing multiple samples for each (method, datapoint) to determine
    mean and standard derivation for each value pair.
    """
    assert quant.ndim == 3
    # Array should be 3d: (method, datapoint, sample)
    mean = np.mean(quant, axis=2)
    std = np.std(quant, axis=2, ddof=1)
    return mean, std


def dostat_2D(qx, qy):
    """ Does statistics on 2d vector results

    Use dataset "quant" containing multiple samples for each (method, datapoint) to determine
    statistical results. Results are assumed to have
    a 2D isotropic Gaussian distribution.
    """
    qx_mean = np.mean(qx, axis=2)
    qy_mean = np.mean(qy, axis=2)
    mean_abs = np.hypot(qx_mean, qy_mean)
    std_x = np.std(qx, axis=2, ddof=1)
    std_y = np.std(qy, axis=2, ddof=1)
    std = np.hypot(std_x, std_y)
    return mean_abs, std


def plot_all(noshow: bool = True, plot_with_legend: bool = True):
    """ create plots of the different metrics """
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    print("Load all variables")
    with open(pickle_name, "rb") as fd:
        Qx_all, Qy_all, Qz_all, QL2_all, BGN_all = pickle.load(fd)

    plt.rcParams.update({'font.size': 16})

    # Qx, QxE = dostat_1D(1e-6 * Qx_all)
    # Qy, QyE = dostat_1D(1e-6 * Qy_all)
    # Qz, QzE = dostat_1D(1e-6 * Qz_all)
    QL2, QL2E = dostat_1D(1e-6 * QL2_all)
    BGN, BGNE = dostat_1D(1e-6 * BGN_all)
    # QA, QAE = dostat_2D(1e-6 * Qx_all, 1e-6 * Qy_all)

    QA_all = np.sqrt(Qx_all * Qx_all + Qy_all * Qy_all)

    print("Do plotting")
    if plot_with_legend:
        plot_with_boxplots(
            noiseLevels, 1e-6 * QA_all, "Total tangential force [µN]", "QT+lengend",
            withLegend=True
        )
    plot_with_boxplots(noiseLevels, 1e-6 * Qx_all, "Total x-force [µN]", "Qx", noshow=noshow)
    plot_with_boxplots(noiseLevels, 1e-6 * Qy_all, "Total y-force [µN]", "Qy", noshow=noshow)
    plot_with_boxplots(noiseLevels, 1e-6 * Qz_all, "Total z-force [µN]", "Qz", noshow=noshow)
    plot_with_boxplots(
        noiseLevels, 1e-6 * QA_all, "Total tangential force [µN]", "QT", noshow=noshow
    )
    plot_with_errorbars(
        noiseLevels, QL2, QL2E, "Total difference in Norm [µN]", "PNorm", noshow=noshow
    )
    plot_with_errorbars(
        noiseLevels, BGN, BGNE, "Background Noise Level", "BGN", noshow=noshow
    )
    print("Plotting done")


def gen_all():
    """
    generate a subfolder 'noised' containing deformation data with different noise levels
    from 'description.toml'
    """
    gen_unnoised_noised(noiseLevels, count=sample_count)


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

    # Add options

    for parser_x in [parser_plot, parser_run]:
        parser_x.add_argument(
            "-q", "--quite", action='store_true', help='Do not show resulting plots'
        )
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
        plot_all(noshow=args.quite, plot_with_legend=args.show_legend_plot)
    elif args.option == "all":
        if not args.use_cached:
            gen_all()
        calc_all(use_cached=args.use_cached)
        plot_all(noshow=args.quite, plot_with_legend=args.show_legend_plot)
    elif args.option == "gen":
        gen_all()
    else:
        parser.print_usage()
