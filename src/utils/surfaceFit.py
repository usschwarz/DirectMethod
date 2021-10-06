"""
Function used to find surface gradients

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import numpy as np

from scipy import integrate

from .diffSquareFit import gradientFitArea


def fit4PointFit2zero(f, zSpacing, z0=0):
    # Swap 3rd axis to the front. This saves us some writing.
    g = np.moveaxis(f, -1, 0)

    # Get the interpolation coefficents

    # Helper so we can postphone the zSpacing division
    r = z0 / zSpacing

    # Coefficents for the constant term
    a1 = (1 + r) * (3 + r) * (4 + r) / 6
    a2 = -(1 / 2) * r * (2 + r) * (3 + r)
    a3 = (1 / 2) * r * (1 + r) * (3 + r)
    a4 = -(1 / 6) * r * (1 + r) * (2 + r)

    # Coefficents for the linear term (we dived by zSpacing later)
    b1 = -11 / 6 - 1 / 2 * r * (4 + r)
    b2 = 3 + 5 * r + (3 * r ** 2) / 2
    b3 = -3 / 2 - 4 * r - (3 * r ** 2) / 2
    b4 = 1 / 3 + r + (r ** 2) / 2

    a = (a1 * g[0] + a2 * g[1] + a3 * g[2] + a4 * g[3])
    b = (b1 * g[0] + b2 * g[1] + b3 * g[2] + b4 * g[3]) / zSpacing

    return a, b


def fit_numpy_polyFit(f, zSpacing, z0=0):
    shp = f.shape

    # For interpolation reformat array
    # Each row in g contains the compleat xy-Plain for  z = z[i]
    # g(z[i])[:]=g[i,:]

    g = f.reshape((-1, shp[2])).T

    nFit = min(shp[2], 8)

    deg = 2  # max(3,nFit-1)

    # Get the z coordinates for the first nFit layers
    z = z0 + zSpacing * np.arange(nFit)

    p = np.polyfit(z, g[:nFit], deg)

    # The last two coefficents contain the semi-constant terms
    return p[-1].reshape(shp[:2]), p[-2].reshape(shp[:2])


def get_standard_us_uzs(u, v, w, xSpacing, ySpacing, zSpacing, nu, z0=0):
    # This function generates the surface displacement as well as well as it's
    # normal (z-derivative) using an unsymmetric 4 point form.
    # See
    # http://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
    assert len(u.shape) == 3, "Shape of u {}".format(u.shape)
    assert u.shape == v.shape
    assert u.shape == w.shape

    assert u.shape[-1] >= 4, "Shape of uvw {}".format(u.shape)  # We need at least 4 layers for this

    def getUsz(f, zSpacing):
        # Swap 3rd axis to the front. This saves us some writing.        
        g = np.moveaxis(f, -1, 0)
        return (g[1] - g[0]) / zSpacing

    def getUs(f):
        return f[:, :, 0]

    uzs = getUsz(u, zSpacing)
    vzs = getUsz(v, zSpacing)
    wzs = getUsz(w, zSpacing)

    us = getUs(u)
    vs = getUs(v)
    ws = getUs(w)

    return us, vs, ws, uzs, vzs, wzs


def get_square_us_uzs(u, v, w, xSpacing, ySpacing, zSpacing, nu, z0=0):
    # This function generates the surface displacement as well as well as it's
    # normal (z-derivative) using an unsymmetric 4 point form.
    # See
    # http://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
    assert (len(u.shape) == 3)
    assert (u.shape == v.shape)
    assert (u.shape == w.shape)

    assert (u.shape[-1] >= 4)  # We need at least 4 layers for this

    def getUsz(f, zSpacing):
        fz = gradientFitArea(f[:, :, :3], zSpacing, axis=2)
        return fz[:, :, 0]

    def getUs(f):
        return f[:, :, 0]

    uzs = getUsz(u, zSpacing)
    vzs = getUsz(v, zSpacing)
    wzs = getUsz(w, zSpacing)

    us = getUs(u)
    vs = getUs(v)
    ws = getUs(w)

    return us, vs, ws, uzs, vzs, wzs


def get_poly_us_uzs(u, v, w, xSpacing, ySpacing, zSpacing, nu, z0=0):
    # This function generates the surface displacement as well as well as it's
    # normal (z-derivative) using an unsymmetric 4 point form.
    # See
    # http://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
    assert (len(u.shape) == 3)
    assert (u.shape == v.shape)
    assert (u.shape == w.shape)

    assert (u.shape[-1] >= 4)  # We need at least 4 layers for this

    us, uzs = fit4PointFit2zero(u, zSpacing, z0)
    vs, vzs = fit4PointFit2zero(v, zSpacing, z0)
    ws, wzs = fit4PointFit2zero(w, zSpacing, z0)

    return us, vs, ws, uzs, vzs, wzs


def get_4Point_us_uzs_offs(u, v, w, xSpacing, ySpacing, zSpacing, nu, offset=0):
    # This function generates the surface displacement as well as well as it's
    # normal (z-derivative) using an unsymmetric 4 point form.
    # See
    # http://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
    assert (len(u.shape) == 3)
    assert (u.shape == v.shape)
    assert (u.shape == w.shape)

    assert (u.shape[-1] >= offset + 4)  # We need at least 4 layers for this

    def get4PointzDSuf(f, zSpacing, offset=0):
        # Swap 3rd axis to the front. This saves us some writing.
        a = -11 / 6 - 1 / 2 * offset * (4 + offset)
        b = 3 + 5 * offset + 3 * offset ** 2 / 2
        c = -3 / 2 - (offset / 2) * (8 + 3 * offset)
        d = 1 / 3 + offset + offset ** 2 / 2

        g = np.moveaxis(f, -1, 0)
        return (a * g[offset + 0] + b * g[offset + 1] + c * g[offset + 2] + d * g[offset + 3]) / zSpacing

    def get4PointUs(f, offset=0):
        if offset == 0:  # If we can trust the uppermost layer we don't extrapolate
            return f[:, :, 0]

        a = (1 + offset) * (2 + offset) * (3 + offset) / 6
        b = -(1 / 2) * offset * (2 + offset) * (3 + offset)
        c = (1 / 2) * offset * (1 + offset) * (3 + offset)
        d = -(1 / 6) * offset * (1 + offset) * (2 + offset)

        g = np.moveaxis(f, -1, 0)
        return a * g[offset + 0] + b * g[offset + 1] + c * g[offset + 2] + d * g[offset + 3]

    uzs = get4PointzDSuf(u, zSpacing, offset)
    vzs = get4PointzDSuf(v, zSpacing, offset)
    wzs = get4PointzDSuf(w, zSpacing, offset)

    us = get4PointUs(u, offset)
    vs = get4PointUs(v, offset)
    ws = get4PointUs(w, offset)

    return us, vs, ws, uzs, vzs, wzs


def get_iSmoothed_us_uzs(u, v, w, xSpacing, ySpacing, zSpacing, nu):
    # This function isn't working properly don't use it

    # This function generates the surface displacement as well as well as it's
    # normal (z-derivative) using an integral version of the Lame-Equation.
    # This uses all aviable data, so it tries to smooth out any noise

    # We will no use these z coordinates
    # ux, uy, uz = np.gradient(u,xySpacing)
    # vx, vy, vz = np.gradient(v,spacing)
    # wx, wy, wz = np.gradient(w,spacing)

    # We will allways do the integrals first, then the differentiation.

    assert (len(u.shape) == 3)
    assert (u.shape == v.shape)
    assert (u.shape == w.shape)

    # Integrate over the z-axis. This does some smoothing.
    uInt = -integrate.simps(u, dx=zSpacing, even='last')
    vInt = -integrate.simps(v, dx=zSpacing, even='last')
    wInt = -integrate.simps(w, dx=zSpacing, even='last')

    def doubleIntegrate(ui):
        Iui = integrate.cumtrapz(ui, dx=zSpacing, initial=0)
        Iui = Iui[:, :, -1:] - Iui
        return integrate.simps(Iui, dx=zSpacing, even='last')

    # Double Integration over the z-axis.
    uIInt = doubleIntegrate(u)
    vIInt = doubleIntegrate(v)
    wIInt = doubleIntegrate(w)

    def laplace2D(ui):
        uix = np.gradient(ui, xSpacing, axis=0)
        uixx = np.gradient(uix, xSpacing, axis=0)

        uiy = np.gradient(ui, ySpacing, axis=1)
        uiyy = np.gradient(uiy, ySpacing, axis=1)
        return uixx + uiyy

    def divergence2D(ui, uj):
        uix = np.gradient(ui, xSpacing, axis=0)
        ujy = np.gradient(uj, ySpacing, axis=1)
        return uix + ujy

    helper1 = (divergence2D(uIInt, vIInt) + wInt) / (1 - 2 * nu)

    us = -np.gradient(helper1, xSpacing, axis=0) - laplace2D(uIInt)
    vs = -np.gradient(helper1, ySpacing, axis=1) - laplace2D(vIInt)
    ws = -((1 - 2 * nu) * laplace2D(wIInt) + divergence2D(uInt, vInt)) / (2 * (1 - nu))

    helper2 = (divergence2D(uInt, vInt) + ws) / (1 - 2 * nu)

    uzs = -np.gradient(helper2, xSpacing, axis=0) - laplace2D(uInt)
    vzs = -np.gradient(helper2, ySpacing, axis=1) - laplace2D(vInt)
    wzs = ((1 - 2 * nu) * laplace2D(wInt) + divergence2D(us, vs)) / (2 * (1 - nu))

    return us, vs, ws, uzs, vzs, wzs
