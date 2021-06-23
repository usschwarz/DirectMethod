#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate simulated deformation data in an .npz file, which can be used by the other scripts.


@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import sys
import warnings
from src.inputSim.inputGen import gen_input

warnings.filterwarnings("error")


def main():
    """ Generate simulated deformation data in an .npz file, which can be used by the other scripts. """
    def parseArg(i, defaultVal):
        """
        Helper function for accessing potentially non existant command line arguments
        """
        if len(sys.argv) > i:
            return sys.argv[i] if sys.argv[i] != '-' else defaultVal
        else:
            return defaultVal

    tCheck = parseArg(1, "")

    if tCheck == "sim":
        noiseLevel = float(parseArg(2, 0))
        fname = parseArg(3, "dvcSim.npz")
        gen_input(noiseLevel, fname)
    else:
        raise RuntimeError('First argument must be "sim"')


exit(main())
