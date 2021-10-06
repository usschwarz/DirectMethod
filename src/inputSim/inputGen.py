"""
Main sample generator

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import os

from .fields.HertzBuilder import get_u_hertz_pattern
from .uFieldSimulation import uFieldSimNew
from ..utils.tomlLoad import loadSimulationData, loadDataDescription, loadAdheasionSites


def gen_input(noiseLevel=0.0, fname="dvcSim.npz"):
    """ Generate simulated deformation field with correct noise level using details described in description.toml """
    if not os.path.isfile("description.toml"):
        raise RuntimeError("Please create a file named 'description.toml' in the current folder stipulating the "
                           "design of the adhesion.")

    name, E, nu, spacing_xy, spacing_z = loadDataDescription()
    n_points_xy, n_points_z = loadSimulationData(silent=True)
    pointList = loadAdheasionSites(silent=True)

    mu = E / (2.0 * (1.0 + nu))

    u, v, w = get_u_hertz_pattern(pointList)

    def uFun(x, y, z):
        """ Function to return displacement vector tuple at a given position """
        return u(x, y, z, mu, nu), v(x, y, z, mu, nu), w(x, y, z, mu, nu)

    print("Observed thickness {} µm.".format(n_points_z * spacing_z))

    print("Simulating DVC result for E={},nu={} with bgNoise {} in file {}".format(E, nu, noiseLevel, fname))

    uFieldSimNew(
        uFun, n_points_x=n_points_xy, n_points_y=n_points_xy, n_points_z=n_points_z,
        sp_xy=spacing_xy, sp_z=spacing_z, E=E, nu=nu, noiseLevel=noiseLevel, fname=fname,
    )
