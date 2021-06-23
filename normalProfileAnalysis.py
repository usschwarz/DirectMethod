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
import pickle

from src.eval.generateNoised import gen_unnoised_noised
from src.eval.fieldProperties import getQGes, getl2Diff, get_bg_noise_level
from src.TFM import tfmFunctions as tfmFn
from src.TFM.uFieldType import load_from_ufile

from src.utils import tomlLoad
from src.inputSim.fields import HertzBuilder

pickle_name = "newNormalAnalysis.pickle"
sample_count = 10
noiseLevels = np.arange(10) * 0.5

methods = [tfmFn.directTFM, tfmFn.squareFitTFM, tfmFn.divFreeFitTFM, tfmFn.fourPSurfaceTFM, tfmFn.reference]
methodnames = ["direct", "square fit", "div rem.", "4th ord. poly.", "reference"]


def evalMeth(u, meth, qfun):
    """
    Calculate traction field using the method function meth and return the different metrics

    u - Input deformation field
    meth - Used metric
    qfun - Function Ground returning the ground truth traction force triple
    """
    pos, _uVec, _uzVec, qVec = meth(u)
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


def plotStuff_no_legend(noiselevels, quant, errors, yLable, name=None, noshow=False):
    """
    Creates visualisations of the results

    noiselevels - Noise Levels where data has been observed
    quant - 2D array containing mean value for the different plotlines for all noiselevels
    errors - 2D array containing error range for the different plotlines for all noiselevels
    yLable - Lable for y axis
    name - If present specifies name used to store plot
    noshow - If set, plot is only created (an possibly saved), but not shown in a popup windows
    """
    plt.close()
    _fig = plt.figure(figsize=[6.9, 4.5])
    ns = noiselevels

    _colors = plt.rcParams['axes.prop_cycle']
    for (qt, er, i) in zip(quant, errors, range(len(quant))):
        plt.plot(ns, qt)
        plt.fill_between(ns, qt - er, qt + er, alpha=0.3)

    plt.xlabel(r"$\sigma_N/<||u||>$")
    plt.ylabel(yLable)

    # plt.autoscale()
    plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    if not noshow:
        plt.gcf().canvas.set_window_title(name)
        plt.show()
    plt.close()


def plotStuff_with_legend(_noiselevels, quant, errors, yLable, name=None):
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

    for (qt, er) in zip(quant, errors):
        plt.plot(qt)

    plt.xlabel(r"$\sigma_N/<||u||>$")
    plt.ylabel(yLable)

    plt.legend(methodnames,
               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    # plt.autoscale()
    plt.tight_layout()
    if name is not None:
        plt.savefig('plots/{}.pdf'.format(name))

    plt.gcf().canvas.set_window_title(name)
    plt.show()
    plt.close()


def plotStuff(noiselevels, quant, errors, yLable, name=None, withLegend=False):
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
        plotStuff_with_legend(noiselevels, quant, errors, yLable, ename)
        plotStuff_no_legend(noiselevels, quant, errors, yLable, name, noshow=True)
    else:
        plotStuff_no_legend(noiselevels, quant, errors, yLable, name)


def calc_all():
    """ calculate traction fields and extract metrics for all noise levels """
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
            for j, mtd in enumerate(methods):
                Qx[j, i, k], Qy[j, i, k], Qz[j, i, k], QL2[j, i, k], BGN[j, i, k] = evalMeth(uN, mtd, qfun)

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    print("Saving results to file")
    with open(pickle_name, "wb") as fd:
        dumpTuple = Qx, Qy, Qz, QL2, BGN
        pickle.dump(dumpTuple, fd)


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


def plot_all():
    """ create plots of the different metrics """
    print("Load all variables")
    with open(pickle_name, "rb") as fd:
        Qx_all, Qy_all, Qz_all, QL2_all, BGN_all = pickle.load(fd)

    plt.rcParams.update({'font.size': 16})

    Qx, QxE = dostat_1D(1e-6 * Qx_all)
    Qy, QyE = dostat_1D(1e-6 * Qy_all)
    Qz, QzE = dostat_1D(1e-6 * Qz_all)
    QL2, QL2E = dostat_1D(1e-6 * QL2_all)
    BGN, BGNE = dostat_1D(1e-6 * BGN_all)
    QA, QAE = dostat_2D(1e-6 * Qx_all, 1e-6 * Qy_all)

    plotStuff(noiseLevels, QA, QAE, "Total tangential force [µN]", "QT", withLegend=True)
    plotStuff(noiseLevels, Qx, QxE, "Total x-force [µN]", "Qx")
    plotStuff(noiseLevels, Qy, QyE, "Total y-force [µN]", "Qy")
    plotStuff(noiseLevels, Qz, QzE, "Total z-force [µN]", "Qz")
    plotStuff(noiseLevels, QA, QAE, "Total tangential force [µN]", "QT")
    plotStuff(noiseLevels, QL2, QL2E, "Total differenze in Norm [µN]", "PNorm")

    plotStuff(noiseLevels, BGN, BGNE, "Background Noise Level", "BGN")


def gen_all():
    """
    generate a subfolder 'noised' containing deformation data with different noise levels from 'description.toml'
    """
    gen_unnoised_noised(noiseLevels, count=sample_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Statistically examin arbitry profiles. Profile must be described in a "description.toml" file')

    # parser.add_argument('--foo', action='store_true', help='foo help')
    subparsers = parser.add_subparsers(dest='option', help='sub-command help')

    # create the parser for the "a" command
    parser_calc = subparsers.add_parser('calc', help='Calculates results')
    parser_plot = subparsers.add_parser('plot', help='Plot results')
    parser_gen = subparsers.add_parser('gen', help='Generate noisy profiles')
    parser_run = subparsers.add_parser('all', help='Calculates, then plots results')

    # create the parser for the "b" command
    # parser_b = subparsers.add_parser('b', help='b help')
    # parser_b.add_argument('--baz', choices='XYZ', help='baz help')

    args = parser.parse_args()

    if args.option == "calc":
        calc_all()
    elif args.option == "plot":
        plot_all()
    elif args.option == "all":
        gen_all()
        calc_all()
        plot_all()
    elif args.option == "gen":
        gen_all()
    else:
        parser.print_usage()
