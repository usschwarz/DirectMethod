"""
Function that extract the used noise leves from a set of simuation fiels

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""
import numpy as np
import os


def getNoiseLevels():
    """ extract the used noise leves from a set of simuation files """
    noiseLevels = []
    for name in os.listdir("noised"):
        if (name[0:len('noise')] == 'noise') and (name[-len("ppt.npz"):] == "ppt.npz"):
            cnoise = int(name[len('noise'):-len("ppt.npz")]) / 1000
            noiseLevels.append(cnoise)

    noiseLevels.sort()
    return np.array(noiseLevels)
