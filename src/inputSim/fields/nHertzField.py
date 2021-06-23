#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:27:10 2018

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)

This file calculates the displacement filds set up by an tangential Hertz profile centered in the coordinate origin.

Parameters explained:
    x,y,z   Point observed
    a       Size of traction area
    Qx      Total traction force applied
    mu      Shear modulus
    nu      Poisson's ratio

"""

# Method based on
# Hamilton, G. M., and Goodman, L. E. (June 1, 1966). 
# "The Stress Field Created by a Circular Sliding Contact."
# ASME. J. Appl. Mech. June 1966; 33(2): 371–376.
# https://doi.org/10.1115/1.3625051
#
# M is Cerrutti-Potential for pure z traction
# ux = (1/(2*mu))*(-(1-2*nu)*Mxz-z*Mxzz)
# uy = (1/(2*mu))*(-(1-2*nu)*Myz-z*Myzz)
# uz = (1/(2*mu))*(2*(1-nu)*Mzz-z*Mzzz)
#
# Used Ansatz (m is arbitry sequence of derivatives):
# k(x,y,z) = 1/2*z^2-1/2*r^2*ln(z+R)-3/4*R*z+1/4r^2
# r = sqrt(x^2+y^2) R=sqrt(x^2+y^2+z^2)
#
# After substitution onto complex curve
# Mm = Im 3P/(2*pi*a^3) Int_\gamma (z-z1) km(x,y,z_1) dz1
# \gamma(t in [0,a]) = z + i*t

# Lower boundary contribution of holomophic integral can be ignored due to
# beeing purely real

# Formulas automatically converted from Mathematica using
# http://juanjose.garciaripoll.com//blog/converting-mathematica-to-python/index.html

# Surface test functions modeled using
# Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge University Press.
# doi:10.1017/CBO9781139171731

import numpy as np

from numba import njit, vectorize, float64


@njit
def iszero(floatval):
    return floatval < 1e-08


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u1_NHertz(x, y, z, a, P, mu, nu):
    z1 = z + 1j*a
    
    def mysqrt(x):
        return np.sqrt((1.+0j)*x)  # Enforces proper complex behavior

    if iszero(x**2+y**2):
        Mxz = 0.
    else:
        aux0=-3.*(((x**2)+(y**2))*(z*(np.log((z1+(mysqrt(((x**2)+((y**2)+(\
        z1**2))))))))))
        aux1=(2.*(((x**2)+(y**2))*(mysqrt(((x**2)+((y**2)+(z1**2)))))))+((z1*(\
        ((-3.*z)+(2.*z1))*((mysqrt(((x**2)+((y**2)+(z1**2)))))-z1)))+aux0)
        Mxz=(0.166667*((x*aux1)))/((x**2)+(y**2))

    Mxz = np.imag(Mxz)
    
    if iszero(x**2+y**2):
        Mxzz = 0.
    else:
        aux0=(((-2.*z)+z1)*((mysqrt(((x**2)+((y**2)+(z1**2)))))-z1))-(((x**2)+\
        (y**2))*(np.log((z1+(mysqrt(((x**2)+((y**2)+(z1**2)))))))))
        Mxzz=(0.5*((x*aux0)))/((x**2)+(y**2))

    Mxzz = np.imag(Mxzz) 
    
    return (1/(2*mu))*(3*P/(2*np.pi*a**3))*(-(1-2*nu)*Mxz-z*Mxzz)


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u2_NHertz(x, y, z, a, P, mu, nu):
    z1 = z + 1j*a
    
    def mysqrt(x):
        return np.sqrt((1.+0j)*x)

    if iszero(x**2+y**2):
        Myz = 0.
    else:
        aux0=-3.*(((x**2)+(y**2))*(z*(np.log((z1+(mysqrt(((x**2)+((y**2)+(\
        z1**2))))))))))
        aux1=(2.*(((x**2)+(y**2))*(mysqrt(((x**2)+((y**2)+(z1**2)))))))+((z1*(\
        ((-3.*z)+(2.*z1))*((mysqrt(((x**2)+((y**2)+(z1**2)))))-z1)))+aux0)
        Myz=(0.166667*((y*aux1)))/((x**2)+(y**2))

    Myz = np.imag(Myz)
    
    if iszero(x**2+y**2):
        Myzz = 0.
    else:        
        aux0=(((-2.*z)+z1)*((mysqrt(((x**2)+((y**2)+(z1**2)))))-z1))-(((x**2)+\
        (y**2))*(np.log((z1+(mysqrt(((x**2)+((y**2)+(z1**2)))))))))
        Myzz=(0.5*((y*aux0)))/((x**2)+(y**2))

    Myzz = np.imag(Myzz)
    return (1/(2*mu))*(3*P/(2*np.pi*a**3))*(-(1-2*nu)*Myz-z*Myzz)


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u3_NHertz(x, y, z, a, P, mu, nu):
    z1 = z + 1j*a
    
    def mysqrt(x):
        return np.sqrt((1.+0j)*x)

    if iszero(x**2+y**2):
        Mzz = 0.25*(z1*((-4.*z)+(z1+(((4.*z)+(-2.*z1))*(np.log((2.*z1)))))))
    else:
        aux0=((x**2)+((y**2)+(2.*(z1*((-2.*z)+z1)))))*(np.log((z1+(mysqrt(((\
        x**2)+((y**2)+(z1**2))))))))
        Mzz=0.25*(((((-4.*z)+z1)*(mysqrt(((x**2)+((y**2)+(z1**2))))))-\
        aux0))

    Mzz = np.imag(Mzz)

    if iszero(x**2+y**2):
        Mzzz = (z*(np.log((2.*z1))))-z1
    else:
        Mzzz=(z*(np.log((z1+(mysqrt(((x**2)+((y**2)+(z1**2)))))))))-(mysqrt(((\
        x**2)+((y**2)+(z1**2)))))
        
    Mzzz = np.imag(Mzzz)
    
    return (1/(2*mu))*(3*P/(2*np.pi*a**3))*(2*(1-nu)*Mzz-z*Mzzz)


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_ur_NHertz(x, y, z, a, P, mu, nu):
    r = np.sqrt(x*x+y*y)
    return x/r*get_u1_NHertz(x, y, z, a, P, mu, nu)+y/r*get_u2_NHertz(x, y, z, a, P, mu, nu)


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_ur_NSurf(x, y, a, P, mu, nu):
    """ Test function. Supposed to give the same result as get_ur_NHertz for z = 0 """
    r = np.sqrt(x*x +y*y)

    if r <= a:
        # Johnson eq. 3.41b
        return -(3*P/(2*np.pi))*((1-2*nu)/(6*mu))*(1/r)*(1-np.sqrt(a*a-r*r)**3/(a**3))
    else:
        # Johnson eq. 3.42b
        return -(3*P/(2*np.pi))*((1-2*nu)/(6*mu))*(1/r)


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u1_NSurf(x, y, a, P, mu, nu):
    """ Test function. Supposed to give the same result as get_u1_NHertz for z = 0 """
    r = np.sqrt(x*x +y*y)
    return get_ur_NSurf(x, y, a, P, mu, nu)*x/r
    

@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u2_NSurf(x, y, a, P, mu, nu):
    """ Test function. Supposed to give the same result as get_u1_NHertz for z = 0 """
    r = np.sqrt(x*x +y*y)
    return get_ur_NSurf(x, y, a, P, mu, nu)*y/r


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u3_NSurf(x, y, a, P, mu, nu):
    """ Test function. Supposed to give the same result as get_u1_NHertz for z = 0 """
    r = np.sqrt(x*x +y*y)

    if r <= a:
        # Johnson eq. 3.41a
        return (3*P/(8*a**3))*((1-nu)/(2*mu))*(2*a*a-r*r)
    else:
        # Johnson eq. 3.42a
        return (3*P/(4*np.pi*a**3))*((1-nu)/(2*mu))*((2*a*a-r*r)*np.arcsin(a/r)+a*np.sqrt(r*r-a*a))


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qx_NSurf(_x, _y, _a, _P):
    return 0


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qy_NSurf(_x, _y, _a, _P):
    return 0


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qz_NSurf(x, y, a, P):
    rSq = x*x +y*y
    if rSq <= a*a:
        return (3*P/(2*np.pi*a**3))*np.sqrt(a*a-rSq)
    else:
        return 0


if __name__ == "__main__":
    spacing = 1.
    mcVal = 200.5
    mcValZ = 10
    
    # Make the grid
    x, y, z = np.meshgrid(np.arange(-mcVal, mcVal, spacing),
                          np.arange(-mcVal, mcVal, spacing),
                          np.arange(0, 2*mcValZ, spacing))
    
    # Make the direction data for the arrows
    b = 120  # µm    Dipole Distance
    a = 40   # µm    Peak area radius
    Qx = 4   # Pa*µm²
    
    nu = 0.45
    E = 20000.0 # Pa
    mu = E/(2.0*(1.0 + nu))
    
    TH_Params = (x, y, z, a, Qx, mu, nu)
    
    TS_Params = (x[..., -1], y[..., -1], a, Qx, mu, nu)
    
    TD_Params = (x[:, :, (0, -10)], y[:, :, (0, -10)], z[:, :, (0, -10)], a)
    
    ua = get_u1_NHertz(*TH_Params)
    va = get_u2_NHertz(*TH_Params)
    wa = get_u3_NHertz(*TH_Params)
    
    us = get_u1_NSurf(*TS_Params)
    vs = get_u2_NSurf(*TS_Params)
    ws = get_u3_NSurf(*TS_Params)


    def hDataInfo(name, dx, dy, dz):
        print("{} data info".format(name))
        print(" type and dimensions:", type(dx[0,0,0]), dx.shape)
        print(" no of datapoints", np.prod(dx.shape), np.prod(dy.shape), np.prod(dz.shape))
        print(" no of Nan-points", len(dx[dx == np.nan]), len(dy[dy == np.nan]), len(dz[dz == np.nan]))
        print(" Peak Values:", np.nanmax(dx), np.nanmax(dy), np.nanmax(dz))
        
    hDataInfo("u", ua, va, wa)

    # Now plot
    import matplotlib.pyplot as plt

    
    def plotLayer(imgLayer, title, cBarLabel=None,saveFig=True):
        xLen = imgLayer.shape[0]*spacing
        yLen = imgLayer.shape[1]*spacing
        
        extent = -xLen/2, xLen/2, -yLen/2, yLen/2
        plt.title(title)
        plt.xlabel(r'x/$\mu m$')
        plt.ylabel(r'y/$\mu m$')
        plt.imshow(imgLayer,origin='lower', interpolation='bilinear', extent=extent)
        cbar = plt.colorbar()
        if cBarLabel:
            cbar.set_label(cBarLabel)
        
        if saveFig:
            print("DEBUG:", title, "Min:",np.nanmin(imgLayer), "Max:", np.nanmax(imgLayer))
            plt.show()

    plotLayer(x[..., -1], 'x', r'$\mu m$')
    plotLayer(y[..., -1], 'y', r'$\mu m$')
    
    plotLayer(ua[..., 0], 'ux_a', r'$\mu m$')
    plotLayer(us, 'ux_s', r'$\mu m$')
    plotLayer(ua[..., 0] - us, 'ux_a-ux_s', r'$\mu m$')
    
    plotLayer(va[..., 0], 'uy_a', r'$\mu m$')
    plotLayer(vs, 'uy_s', r'$\mu m$')
    plotLayer(va[..., 0] - vs, 'uy_a-uy_s', r'$\mu m$')
    
    plotLayer(wa[..., 0], 'uz_a', r'$\mu m$')
    plotLayer(ws, 'uz_s', r'$\mu m$')
    plotLayer(wa[..., 0] - ws, 'uz_a-uz_s', r'$\mu m$')
    
    uy, ux, uz = np.gradient(ua, spacing)
    vy, vx, vz = np.gradient(va, spacing)
    wy, wx, wz = np.gradient(wa, spacing)
    
    divU = ux+vy+wz

    # The surface normal of the plane is directed outwards, hence
    # n = (0,0,-1). Now use \vec{q} = \vec{n}\cdot\boldsymbol{\sigma}
    # qx = -sigma_xz
    # qy = -sigma_yz
    # qz = -sigma_zz
    
    qx = -mu*(uz+wx)
    qy = -mu*(vz+wy)
    qz = -mu*(wz+wz+2*nu/(1-2*nu)*divU)
    
    qx = qx[..., 0]
    qy = qy[..., 0]
    qz = qz[..., 0]
    
    plotLayer(qx, 'qx', 'Pa')
    plotLayer(qy, 'qy', 'Pa')
    plotLayer(qz, 'qz', 'Pa')
    
    qzTheo = get_qy_NSurf(x[..., -1], y[..., -1], a, Qx)
    plotLayer(qzTheo, 'qz_theo', 'Pa')
    
    plotLayer(qz - qzTheo, 'qz-qz_theo', r'$\mu m$')