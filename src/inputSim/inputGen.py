#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:48:56 2019

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import os

# import warnings
# warnings.filterwarnings("error")

from .fields.HertzBuilder import get_u_hertz_pattern
from .uFieldSimulation import uFieldSim
from ..utils.tomlLoad import loadSimulationData, loadDataDescription, loadAdheasionSites


def gen_input(noiseLevel=0.0, fname="dvcSim.npz"):
    """ Generate simulated deformation field with correct noise level using details described in description.toml """
    if not os.path.isfile("description.toml"):
        raise RuntimeError("Please create a file named 'description.toml' in the current folder stipulating the "
                           "design of the adhesion.")

    name, E, nu, microns_per_pixel, dLayers = loadDataDescription()
    nLayers, xyPix, NBeads = loadSimulationData(silent=True)
    pointList = loadAdheasionSites(silent=True)

    mu = E / (2.0 * (1.0 + nu))

    u, v, w = get_u_hertz_pattern(pointList)

    def uFun(x, y, z):
        """ Function to return displacement vector tuple at a given position """
        return u(x, y, z, mu, nu), v(x, y, z, mu, nu), w(x, y, z, mu, nu)

    print("Observed thickness {} µm.".format(dLayers * nLayers))

    print("Simulating DVC result for E={},nu={} with bgNoise {} in file {}".format(E, nu, noiseLevel, fname))
    uFieldSim(uFun, xpix=xyPix, ypix=xyPix, umPerPx=microns_per_pixel,
              nLayers=nLayers, dLayers=dLayers, E=E, nu=nu, noiseLevel=noiseLevel, fname=fname)
