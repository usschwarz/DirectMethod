#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to test the convergence of the dcv algorithm

Usage:
    python dcvConvergenceTest.py <field1.npz> <field2.npz> ...

    <field1.npz>, <field2.npz>, <field3.npz>, ...

    Deformation fields

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

from pathlib import Path
from typing import Tuple, List
from matplotlib.axes import Axes

# Stress calculators
from src.TFM.calcStress import linearStress, get_F_Strain_square, VoigtMatNdarrayTuple
from src.TFM.DCS import DCS_voigt_internal
from src.TFM.uFieldType import load_from_ufile


def get_sigma_raw(fname: str) -> Tuple[VoigtMatNdarrayTuple, float, float]:
    """
    Load a derformation field from a file 'fname' and calculate
    the stress tensor using the square fit method

    Returns the stress tensor and spacing
    """
    udata = load_from_ufile(fname)
    F = get_F_Strain_square(*udata.get_u_and_sp())
    sigma = linearStress(udata.mu, udata.nu, F)
    return sigma, udata.dm, udata.dmz


def get_random_data(fname: str):
    """
    Uses the metadata of the deformation field in file 'fname' to
    generate a stess tensor filled with white nose

    Returns the stress tensor and spacing
    """
    udata = load_from_ufile(fname)
    shape = (len(udata.m[0]), len(udata.m[1]), len(udata.m[2]))

    F = get_F_Strain_square(*udata.get_u_and_sp())
    sigma_raw = linearStress(udata.mu, udata.nu, F)
    res_tuple = []
    for en in sigma_raw:
        maxi = np.max(en)
        mini = np.min(en)
        random = (maxi - mini) * np.random.rand(*shape) + mini
        res_tuple.append(random)
    return res_tuple, udata.dm, udata.dmz


def analyse(israndom: bool, fname: str, itercount: int):
    """
    Calculates the effects of divergence removal over multiple iterations

    israndom - Whether the deformation dataset should be used directly or only to generate white noise
    fname - Name of the input dataset
    itercount - Number of iterations

    returns dataset
    """
    if israndom:
        sigma, dm, dmz = get_random_data(fname)
    else:
        sigma, dm, dmz = get_sigma_raw(fname)

    _, snorm, divlevel, mdpeak, smatmean, nancount = DCS_voigt_internal(sigma, dm, dm, dmz, divergenceThreshold=0,
                                                                        maxIterations=(itercount - 1), verbose=True)

    print(len(snorm), itercount)
    assert (len(snorm) == itercount)
    assert (len(divlevel) == itercount)
    assert (len(mdpeak) == itercount)
    assert (len(smatmean) == itercount)
    assert (len(nancount) == itercount)

    return np.array(snorm), np.array(divlevel), np.array(mdpeak), np.array(smatmean), np.array(nancount, dtype=int)


def plot_set(ax: Axes, datasets: List[np.ndarray], lables: List[str], title: str, ylable: str = "", yscale="log"):
    """
    Ploting helper routines
    """
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylable)
    xlen = len(datasets[0])
    xrange = np.arange(xlen)
    ax.set_yscale(yscale)
    for lable, dataset in zip(lables, datasets):
        ax.plot(xrange, dataset)


def genme():
    """
    Analysises the effects of divergence removal in relation to the number of iterations for deformation profiles
    This function assumes multiple
    """
    itercount = 50

    if len(sys.argv) < 2:  # This should never be called
        raise RuntimeError("Please specify at least one input file")

    snorm, divlevel, mdpeak, smatmean, nancount = analyse(True, sys.argv[1], itercount)

    snorms = [snorm]
    divlevels = [divlevel]
    mdpeaks = [mdpeak]
    smatmeans = [1 - smatmean]
    nancounts = [nancount]

    lables = ["random"]

    fnames = sys.argv[1:]

    namebuff = ''

    for i, fname in enumerate(fnames):
        if namebuff == '':
            lable = namebuff
            namebuff = ''
            if not os.path.isfile(fname):
                raise RuntimeError("Argument {} ({}) must specify an input file, but it doesn't".format(i, fname))
        else:
            if os.path.isfile(fname):
                lable = Path(fname).stem
            else:
                namebuff = fname
                continue

        snorm, divlevel, mdpeak, smatmean, nancount = analyse(False, fname, itercount)
        snorms.append(snorm)
        divlevels.append(divlevel)
        mdpeaks.append(mdpeak)
        smatmeans.append(1 - smatmean)
        lables.append(lable)
        nancounts.append(nancount)

    superlist = [snorms, divlevels, mdpeaks, smatmeans, lables, nancounts]

    # Quick and dirty saveing
    pickle.dump(superlist, open("convergence_test.pickle", "wb"))


def plotme():
    """
    Attempts to load and plot the results stored in 'convergence_test.pickle'
    """
    superlist = pickle.load(open("convergence_test.pickle", "rb"))
    snorms, divlevels, mdpeaks, smatmeans, lables, nancounts = superlist

    _fig = plt.figure(figsize=[6.9, 4.5])
    ax = plt.subplot()
    plot_set(ax, snorms, lables, "Absolute mean divergence", ylable=r"$\mathrm{Pa}\cdot\mathrm{µm}^{-1}$")
    plt.show()
    plt.clf()

    _fig = plt.figure(figsize=[6.9, 4.5])
    ax = plt.subplot()
    plot_set(ax, divlevels, lables, "Relative mean divergence", ylable=r"$\mathrm{µm}^{-1}$")
    plt.show()
    plt.clf()

    _fig = plt.figure(figsize=[6.9, 4.5])
    ax = plt.subplot()
    plot_set(ax, mdpeaks, lables, "Mean peak antisymmetric component", ylable=r"$\mathrm{Pa}$")
    plt.show()
    plt.clf()

    _fig = plt.figure(figsize=[6.9, 4.5])
    ax = plt.subplot()
    plot_set(ax, smatmeans, lables, "Mean asymmetricity")
    plt.show()
    plt.clf()


def main():
    """
    Main program
    """
    if len(sys.argv) < 2:
        if not os.path.isfile("convergence_test.pickle"):
            raise RuntimeError("Please generate a solution first by specifying one or multiple deformation profiles "
                               "in the command line arguments.")
        plotme()
    else:
        genme()


exit(main())
