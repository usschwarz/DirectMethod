#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:39:09 2019

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""
import numpy as np
import os


def getNoiseLevels():
    noiseLevels = []
    for name in os.listdir("noised"):
        if (name[0:len('noise')] == 'noise') and (name[-len("ppt.npz"):] == "ppt.npz"):
            cnoise = int(name[len('noise'):-len("ppt.npz")]) / 1000
            noiseLevels.append(cnoise)

    noiseLevels.sort()
    return np.array(noiseLevels)
