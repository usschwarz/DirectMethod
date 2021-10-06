"""
Defines uFieldData, a structure that bundels deformation field and materials data

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import numpy as np


def E2mu(E, nu):
    return E / (2.0 * (1.0 + nu))


def mu2E(mu, nu):
    return 2 * (1 + nu) * mu


def tryToFloat(a):
    try:
        b = float(a)
    except ValueError:
        b = float(a[1:])
    return b


def dumpu2npz(filename, ux, uy, uz, uAbs, cc, dmz, dm, m, E, nu, nCritical):
    np.savez(filename, ux=ux, uy=uy, uz=uz, uAbs=uAbs, cc=cc, dmz=dm, dm=dm, m=m, InfoBlob=np.array([E, nu]),
             nCritical=nCritical)
    print("Saved results in ", filename)


class uFieldData:
    """
    Structure that bundels deformation field and materials data
    """

    def __init__(self, uDataFile):
        self.uDataFile = uDataFile

        # print("Loading Results from {} ... ".format(filename),end='', flush=True)
        with np.load(self.uDataFile, allow_pickle=True) as data:
            self.dm = float(data['dm'])
            dmz = float(data['dmz'])

            # Load m
            try:
                mtemp = data['m']
                self.m = (mtemp[0], mtemp[1], mtemp[2])
            except KeyError:
                self.m = (data['mx'], data['my'], data['mz'])

            E, nu = data['InfoBlob']
            self.E = tryToFloat(E)
            self.nu = tryToFloat(nu)
            try:
                self.nCritical = int(data['nCritical'])
            except:
                self.nCritical = 0

        # print("done")
        self.mu = E2mu(self.E, self.nu)

        print("DM Check", self.m[1][1], self.m[1][0], self.dm)
        print("DM Check", self.m[2][1], self.m[2][0], dmz)

        self.dmz = self.m[2][1] - self.m[2][0]
        self.z0 = self.m[2][0]

    def get_cc(self) -> np.ndarray:
        with np.load(self.uDataFile) as data:
            cc = data['cc']

        return cc

    def get_spacing(self):
        return self.dm

    def get_u(self, getAbs=False):
        # ux, uy, uz, uAbs, cc, dm, m, E, nu = load4npz(uDataFile)
        with np.load(self.uDataFile) as data:
            ux = data['ux']
            uy = data['uy']
            uz = data['uz']
            uAbs = data['uAbs']
        if getAbs:
            return ux, uy, uz, uAbs
        else:
            return ux, uy, uz

    def get_u_and_sp(self):
        # ux, uy, uz, uAbs, cc, dm, m, E, nu = load4npz(uDataFile)
        with np.load(self.uDataFile) as data:
            ux = data['ux']
            uy = data['uy']
            uz = data['uz']
        return ux, uy, uz, self.dm, self.dm, self.dmz


def load_from_ufile(uDataFile: str) -> uFieldData:
    return uFieldData(uDataFile)
