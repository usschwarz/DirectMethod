"""
This file implements the different methods to calculate TFM from the input filed.

All function take the u-field object definied in uFieldType as their input
An return:
(x,y) Coordinate sampling grids used for the other outputs
(us, vs, ws) - (If available) reconstructed displacement fields at the surface
(usz, vsz, wsz) - (If applicable) normal - direction deformation gradient at the surface
(qx, qy, qz) - Reconstructed traction forces at the surface. <qz> will be set to zero if not
    reconstructed

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""



import numpy as np

# us_usz getters
from ..utils.surfaceFit import get_standard_us_uzs, get_poly_us_uzs

# Stress calculators
from .calcStress import calculate_qxyz, linearStress, get_F_Strain_square
from .calcStress import readoutSurface
from .DCS import DCS_voigt

# Reference builder
from ..inputSim.fields.HertzBuilder import get_q_hertz_pattern, get_u_hertz_pattern
from ..utils.tomlLoad import loadAdheasionSites

# FTTC implementations
from .FTTC import runFTTC
from .FTTC3d import runFTTC3d


def fourPSurfaceTFM(u):
    print("Calculate 4 Point Surface Gradient TFM")
    us, vs, ws, usz, vsz, wsz = get_poly_us_uzs(*u.get_u_and_sp(), nu=u.nu)
    qx, qy, qz = calculate_qxyz(u.mu, u.nu, us, vs, ws, usz, vsz, wsz, u.dm, use5p=True)

    x, y = np.meshgrid(u.m[0], u.m[1], indexing='ij')

    return (x, y), (us, vs, ws), (usz, vsz, wsz), (qx, qy, qz)


def directTFM(u):
    print("Calculate direct Surface Gradient TFM")
    us, vs, ws, usz, vsz, wsz = get_standard_us_uzs(*u.get_u_and_sp(), nu=u.nu)

    qx, qy, qz = calculate_qxyz(u.mu, u.nu, us, vs, ws, usz, vsz, wsz, u.dm)

    x, y = np.meshgrid(u.m[0], u.m[1], indexing='ij')

    return (x, y), (us, vs, ws), (usz, vsz, wsz), (qx, qy, qz)


def squareFitTFM(u):
    print("Calculate surface strains from using the square patch method")
    F = get_F_Strain_square(*u.get_u_and_sp())
    print("Spacing: ", u.dm, u.dmz)
    sigma = linearStress(u.mu, u.nu, F)

    sxx, syy, szz, syz, sxz, sxy = sigma  # pylint: disable=unused-variable
    us, vs, ws, usz, vsz, wsz = readoutSurface(*u.get_u(), F)

    x, y = np.meshgrid(u.m[0], u.m[1], indexing='ij')

    return (x, y), (us, vs, ws), (usz, vsz, wsz), (-sxz[:, :, 0], -syz[:, :, 0], -szz[:, :, 0])


def DCS_TFM(u):
    print("Calculate surface strains from div free fitted stress tensor")
    F = get_F_Strain_square(*u.get_u_and_sp())
    sigma = linearStress(u.mu, u.nu, F)
    sigma = DCS_voigt(sigma, u.dm, u.dm, u.dmz)

    sxx, syy, szz, syz, sxz, sxy = sigma  # pylint: disable=unused-variable
    us, vs, ws, usz, vsz, wsz = readoutSurface(*u.get_u(), F)

    x, y = np.meshgrid(u.m[0], u.m[1], indexing='ij')

    return (x, y), (us, vs, ws), (usz, vsz, wsz), (-sxz[:, :, 0], -syz[:, :, 0], -szz[:, :, 0])


def WDCS_TFM(u):
    print("Calculate surface strains from div free fitted stress tensor")
    F = get_F_Strain_square(*u.get_u_and_sp())
    sigma = linearStress(u.mu, u.nu, F)
    sigma = DCS_voigt(sigma, u.dm, u.dm, u.dmz, weighting=True)

    sxx, syy, szz, syz, sxz, sxy = sigma  # pylint: disable=unused-variable
    us, vs, ws, usz, vsz, wsz = readoutSurface(*u.get_u(), F)

    x, y = np.meshgrid(u.m[0], u.m[1], indexing='ij')

    return (x, y), (us, vs, ws), (usz, vsz, wsz), (-sxz[:, :, 0], -syz[:, :, 0], -szz[:, :, 0])


divFreeFitTFM = DCS_TFM


def FTTC(u):
    ux, uy, _ = u.get_u()
    # pos = np.meshgrid(u.m[0], u.m[1], indexing='ij')
    us = ux[:, :, 0]
    vs = uy[:, :, 0]
    xy, fnorm, f, urec, u, energy, force, Ftf, Fturec = \
        runFTTC.perform_FTTC(u.m[0], u.m[1], us, vs, u.dm, u.E, u.nu)  # pylint: disable=unused-variable

    zero = np.zeros_like(urec[0])
    return xy, (urec[0].T, urec[1].T, zero.T), (zero.T, zero.T, zero.T), (f[0].T, f[1].T, zero.T)


def FTTC3d(u):
    ux, uy, uz = u.get_u()
    # pos = np.meshgrid(u.m[0], u.m[1], indexing='ij')
    us = ux[:, :, 0]
    vs = uy[:, :, 0]
    ws = uz[:, :, 0]
    xy, fnorm, f, urec, u, energy, force, Ftf, Fturec = \
        runFTTC3d.perform_FTTC(u.m[0], u.m[1], us, vs, ws, u.dm, u.E, u.nu)  # pylint: disable=unused-variable

    zero = np.zeros_like(urec[0])
    return xy, (urec[0].T, urec[1].T, urec[2].T), (zero.T, zero.T, zero.T), (f[0].T, f[1].T, f[2].T)


def reference(u):
    pointList = loadAdheasionSites()
    x, y = np.meshgrid(u.m[0], u.m[1], indexing='ij')
    qx, qy, qz = get_q_hertz_pattern(pointList)
    ux, uy, uz = get_u_hertz_pattern(pointList)
    q = (qx(x, y), qy(x, y), qz(x, y))
    void = np.zeros_like(x)
    prm = x, y, void, u.mu, u.nu
    u = (ux(*prm), uy(*prm), uz(*prm))
    return (x, y), u, (void, void, void), q


tfm_dir = {
    "4th ord. poly.": fourPSurfaceTFM,
    "direct": directTFM,
    "square fit": squareFitTFM,
    "DCS": DCS_TFM,
    "WDCS": WDCS_TFM,
    "div rem.": divFreeFitTFM,
    "FTTC": FTTC,
    "FTTC (3D)": FTTC3d,
    "reference": reference,
}
