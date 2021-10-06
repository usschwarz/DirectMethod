#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates noised sample fields form an unnoised one.

Call signature:
python generateNoised.py referencefilename

Produces replicas of referencefilename where various types of noise have been added.

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""



import sys
import os
import numpy as np

try:  # This is Python :(
    # Try relative (if this is included)
    from ..TFM.uFieldType import load_from_ufile
    from ..inputSim.inputGen import gen_input
except (ValueError, ImportError):
    # Try absolute
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
    from src.TFM.uFieldType import load_from_ufile
    from src.inputSim.inputGen import gen_input


def parseArg(i, defaultVal):
    if len(sys.argv) > i:
        return sys.argv[i] if sys.argv[i] != '-' else defaultVal
    else:
        return defaultVal


def dumpu2npz(filename, ux, uy, uz, uAbs, cc, dmz, dm, m, E, nu, nCritical):
    np.savez(filename, ux=ux, uy=uy, uz=uz, uAbs=uAbs, cc=cc, dmz=dmz, dm=dm, mx=m[0], my=m[1], mz=m[2],
             InfoBlob=np.array([E, nu]), nCritical=nCritical)
    print("Saved results in ", filename)


def _generate_single_noised(cnoise, noiseFilename, uA, vA, wA, uAshape, uMean, dmz, dm, m, E, nu):
    if cnoise < 0.01:
        uNoise = np.zeros((3,) + uAshape)
    else:
        # cnoiseSq = np.sqrt(cnoise)
        uNoise = np.random.normal(scale=cnoise * uMean, size=(3,) + uAshape)
    uB = uA + uNoise[0]
    vB = vA + uNoise[1]
    wB = wA + uNoise[2]
    uAbsB = np.sqrt(uB * uB + vB * vB + wB * wB)
    cc = np.sqrt(uNoise[0] * uNoise[0] + uNoise[1] * uNoise[1] + uNoise[2] * uNoise[2])
    ccMax = np.max(cc)
    cc = ccMax - cc
    os.makedirs("noised", exist_ok=True)

    dumpu2npz(noiseFilename, uB, vB, wB, uAbsB, cc, dmz, dm, m, E, nu, 0)


def generate_noised(noiselevels, fname="normal.dvcsim.npz", count=1):
    if not os.path.isdir("noised"):
        os.mkdir("noised")

    # Load in reference file
    uTrue = load_from_ufile(fname)

    # dumpu2npz(filename, ux, uy, uz, uAbs, cc, dmz, dm, m, E, nu,nCritical=0)

    uA, vA, wA, uAbsA = uTrue.get_u(getAbs=True)
    # cc = uTrue.get_cc()
    dmz = uTrue.dmz
    dm = uTrue.dm
    m = uTrue.m
    E = uTrue.E
    nu = uTrue.nu

    # uGeomMean = np.sqrt(np.mean(uAbsA*uAbsA))
    uMean = np.mean(uAbsA)

    # Done Loading
    if count == 1:
        # This case is hard coded as it uses different file names
        for cnoise in noiselevels:
            noiseFilename = "noised/noise{:d}ppt.npz".format(int(cnoise * 1000))
            _generate_single_noised(cnoise, noiseFilename, uA, vA, wA, uAbsA.shape, uMean, dmz, dm, m, E, nu)
    else:
        for cnoise in noiselevels:
            for i in range(count):
                noiseFilename = "noised/noise{:d}ppt-{}.npz".format(int(cnoise * 1000), i)
                _generate_single_noised(cnoise, noiseFilename, uA, vA, wA, uAbsA.shape, uMean, dmz, dm, m, E, nu)

    # Creates a flag file that can be used to control data regeneration
    with open("noised/flag.txt", 'a'):  # Create file if does not exist
        pass


def gen_unnoised_noised(noiselevels, count=1):
    if not os.path.isdir("noised"):
        os.mkdir("noised")

    fname = "noised/normal.dvcsim.npz"

    # Generate reference file (without noise
    gen_input(noiseLevel=0.0, fname=fname) # noqa

    # Generate noised fields
    generate_noised(noiselevels, fname, count)


if __name__ == "__main__":
    # Generate for noisy files
    # SNRLevel = np.arange(1,50,1)
    # noiseLevels = 1/SNRLevel # np.arange(20)*0.001
    noiseLevels = np.arange(50) * 0.1
    exit(gen_unnoised_noised(noiseLevels))
