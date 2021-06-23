#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:27:10 2018

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)

This file calculates the displacement fields set up by an tangential Hertz profile
in x direction centered in the coordinate origin.

Parameters explained:
    x,y,z   Point observed
    a       Size of traction area
    Qx      Total traction force applied
    mu      Shear modulus
    nu      Poisson's ratio

"""

# Ansatz based on
# Hamilton, G. M., and Goodman, L. E. (June 1, 1966). 
# "The Stress Field Created by a Circular Sliding Contact."
# ASME. J. Appl. Mech. June 1966; 33(2): 371–376.
# https://doi.org/10.1115/1.3625051
#
# T is Cerrutti-Potential for pure x traction
# ux = (1/(2*mu))*(2*nu*Txx+2*Tzz-z*Txxz)
# uy = (1/(2*mu))*(2*nu*Txy-z*Txyz)
# uz = (1/(2*mu))*((1-2*nu)*Txz-z*Txzz)
#
# Used Ansatz (m is arbitry sequence of derivatives):
# k(x,y,z) = 1/2*z^2-1/2*r^2*ln(z+R)-3/4*R*z+1/4r^2
# r = sqrt(x^2+y^2) R=sqrt(x^2+y^2+z^2)
#
# After substitution onto complex curve
# Tm = Im 3Q/(2*pi*a^3) Int_\gamma (z-z1) km(x,y,z_1) dz1
# \gamma(t in [0,a]) = z + i*t

# Lower boundary contribution of holomophic integral can be ignored due to
# beeing purely real

# Formulas automatically converted from mathematica using
# http://juanjose.garciaripoll.com//blog/converting-mathematica-to-python/index.html

# Surface test functions modeled using
# Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge University Press.
# doi:10.1017/CBO9781139171731
# with corrections by
# Benedikt Rennekamp

import numpy as np

from numba import njit, vectorize, float64


@njit
def iszero(floatval):
    return floatval < 1e-08


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u1_THertz(x, y, z, a, Qx, mu, nu):
    z1 = z+1j*a
    
    def mysqrt(x): return np.sqrt((1.+0j)*x)

    if iszero(x**2+y**2):
        Txx = -0.125*(z1*((-4.*z)+(z1+(((4.*z)+(-2.*z1))*(np.log((2.*z1)))))))
    else:
        aux0=(3.*(((x**2)+(y**2))*(((3.*(x**2))+(y**2))*z1)))+((8.*(((y**2)-(\
        x**2))*(z*(z1**2))))+(6.*((x-y)*((x+y)*(z1**3.)))))
        aux1=(mysqrt(((x**2)+((y**2)+(z1**2)))))*((-16.*(((x**2)+(y**2))*(((2.\
        *(x**2))+(y**2))*z)))+aux0)
        aux2=((3.*(x**2))+(y**2))*(np.log((z1+(mysqrt(((x**2)+((y**2)+(z1**2))\
        ))))))
        aux3=((((x**2)+(y**2))**2))*(z1*(((-2.*z)+z1)*(np.log((z1+(mysqrt(((\
        x**2)+((y**2)+(z1**2))))))))))
        aux4=(6.*(((y**2)-(x**2))*(z1**4.)))+(aux1+((-3.*(((((x**2)+(y**2))\
        **2))*aux2))+(-12.*aux3)))
        aux5=(6.*(((y**4.)-(x**4.))*(z1**2)))+((8.*((x-y)*((x+y)*(z*(z1**3.)))\
        ))+aux4)
        Txx=-0.0208333*(((((x**2)+(y**2))**-2.)*((12.*(((x**4.)-(y**4.))\
        *(z*z1)))+aux5)))
                            
    # Txx0=-0.125*(z1*((-4.*z)+(z1+(((4.*z)+(-2.*z1))*(np.log((2.*z1)))))))
    # Txx = wherezero(x**2+y**2,Txx0,Txx)

    Txx = np.imag(Txx)

    if iszero(x**2+y**2):
        Tzz = 0.25*(z1*((-4.*z)+(z1+(((4.*z)+(-2.*z1))*(np.log((2.*z1)))))))
    else:
        aux0=((x**2)+((y**2)+(2.*(z1*((-2.*z)+z1)))))*(np.log((z1+(mysqrt(((x**\
        2)+((y**2)+(z1**2))))))))
        Tzz=0.25*(((((-4.*z)+z1)*(mysqrt(((x**2)+((y**2)+(z1**2))))))-\
        aux0))

    # Tzz0=0.25*(z1*((-4.*z)+(z1+(((4.*z)+(-2.*z1))*(np.log((2.*z1)))))))
    # Tzz = wherezero(x**2+y**2,Tzz0,Tzz)
    
    Tzz = np.imag(Tzz)

    if iszero(x**2+y**2):
        Txxz = 0.25*(z+((2.*z1)+(-2.*(z*(np.log((2.*z1)))))))
    else:
        aux0=((x**2)*((6.*(y**2))+(((3.*z)+(-2.*z1))*z1)))+((y**2)*((2.*(y**2)\
        )+(z1*((-3.*z)+(2.*z1)))))
        aux1=(((x**2)+(y**2))**-2.)*((mysqrt(((x**2)+((y**2)+(z1**2)))))*((4.*\
        (x**4.))+aux0))
        aux2=(2.*((x-y)*((x+y)*((((x**2)+(y**2))**-2.)*(z1**3.)))))+(aux1+(-3.\
        *(z*(np.log((z1+(mysqrt(((x**2)+((y**2)+(z1**2)))))))))))
        Txxz=0.166667*((3.*(((y**2)-(x**2))*((((x**2)+(y**2))**-2.)*(z*(\
        z1**2)))))+aux2)

    # Txxz0=0.25*(z+((2.*z1)+(-2.*(z*(np.log((2.*z1)))))))
    # Txxz = wherezero(x**2+y**2,Txxz0,Txxz)
    
    Txxz = np.imag(Txxz)
    
    return (1/(2*mu))*(3*Qx/(2*np.pi*a**3))*(2*nu*Txx+2*Tzz-z*Txxz)


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u2_THertz(x, y, z, a, Qx, mu, nu):
    z1 = z + 1j*a
    
    def mysqrt(x): return np.sqrt((1.+0j)*x)
    
    if iszero(x**2+y**2):
        Txy = 0.
    else:
        aux0=(-8.*(((x**2)+(y**2))*z))+((3.*(((x**2)+(y**2))*z1))+((-8.*(z*(\
        z1**2)))+(6.*(z1**3.))))
        aux1=((((x**2)+(y**2))**-2.)*((mysqrt(((x**2)+((y**2)+(z1**2)))))*\
        aux0))+(-3.*(np.log((z1+(mysqrt(((x**2)+((y**2)+(z1**2)))))))))
        aux2=(8.*((((x**2)+(y**2))**-2.)*(z*(z1**3.))))+((-6.*((((x**2)+(y**2)\
        )**-2.)*(z1**4.)))+aux1)
        aux3=y*(((12.*(z*z1))/((x**2)+(y**2)))+(((-6.*(z1**2))/((x**2)+(y**2))\
        )+aux2))
        Txy=-0.0416667*((x*aux3))
    
    # Txy = wherezero(x**2+y**2,0,Txy)
    Txy = np.imag(Txy)

    if iszero(x**2+y**2):
        Txyz = 0.
    else:
        aux0=(((x**2)+(y**2))*(mysqrt(((x**2)+((y**2)+(z1**2))))))-(z1*(((-3.*\
        z)+(2.*z1))*((mysqrt(((x**2)+((y**2)+(z1**2)))))-z1)))
        Txyz=0.333333*((x*(y*((((x**2)+(y**2))**-2.)*aux0))))
    
    # Txyz = wherezero(x**2+y**2,0,Txyz)
    Txyz = np.imag(Txyz)
    
    return (1/(2*mu))*(3*Qx/(2*np.pi*a**3))*(2*nu*Txy-z*Txyz)


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u3_THertz(x, y, z, a, Qx, mu, nu):
    z1 = z + 1j*a
    
    def mysqrt(x): return np.sqrt((1.+0j)*x)

    if iszero(x**2+y**2):
        Txz = 0.
    else:
        aux0=-3.*(((x**2)+(y**2))*(z*(np.log((z1+(mysqrt(((x**2)+((y**2)+(\
        z1**2))))))))))
        aux1=(2.*(((x**2)+(y**2))*(mysqrt(((x**2)+((y**2)+(z1**2)))))))+((z1*(\
        ((-3.*z)+(2.*z1))*((mysqrt(((x**2)+((y**2)+(z1**2)))))-z1)))+aux0)
        Txz=(0.166667*((x*aux1)))/((x**2)+(y**2))
    
    # Txz = wherezero(x**2+y**2,0,Txz)
    Txz = np.imag(Txz)
    
    if iszero(x**2+y**2):
        Txzz = 0.
    else:
        aux0=(((-2.*z)+z1)*((mysqrt(((x**2)+((y**2)+(z1**2)))))-z1))-(((x**2)+\
        (y**2))*(np.log((z1+(mysqrt(((x**2)+((y**2)+(z1**2)))))))))
        Txzz=(0.5*((x*aux0)))/((x**2)+(y**2))
    
    # Txzz = wherezero(x**2+y**2,0,Txzz)
    Txzz = np.imag(Txzz)
    
    return (1/(2*mu))*(3*Qx/(2*np.pi*a**3))*((1-2*nu)*Txz-z*Txzz)


def get_u1_TPoint(x,y,z, a, Qx, mu, nu):
    rSq = x*x +y*y
    #rSq[rSq < np.finfo(np.float64).tiny] = np.nan
    #z[z < np.finfo(np.float64).tiny] = np.nan
    r = np.sqrt(rSq)
    return (Qx/(4*np.pi*mu))*(1/r+x*x*(r**3)+(1-2*nu)*(1/(r+z)-x*x/(r*(r+z)**2)))


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u1_TSurf(x, y, a, Qx, mu, nu):
    # Test function. Supposed to give the same result as get_u1_THertz
    # for z = 0

    r = np.sqrt(x*x +y*y)
    if r<=a:
        # Johnson 3.91a
        uC = (3*Qx/(64*mu*a**3))*(4*(2-nu)*a**2-(4-3*nu)*x**2-(4-nu)*y**2)
    else:
        # Johnson 3.92a Fixed missing 1/r^2 in final factor
        uC = (3*Qx/(16*np.pi*mu*a**3))*(\
        (2-nu)*((2*a**2-r**2)*np.arcsin(a/r)+a*r*np.sqrt(1-a**2/r**2)) \
        +0.5*nu*(r**2*np.arcsin(a/r)+(2*a**2-r**2)*np.sqrt(1-a**2/r**2)*a/r) \
        *(x**2-y**2)/(r**2))

    return uC


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u2_TSurf(x, y, a, Qx, mu, nu):
    # Test function. Supposed to give the same result as get_u2_THertz
    # for z = 0

    r = np.sqrt(x*x +y*y)
    if r<=a:
        # Johnson 3.91b
        uC = (3*Qx/(64*mu*a**3))*2*nu*x*y
    else:
        # Johnson 3.92b Fixed missing 1/r^2 in final factor
        uC = (3*Qx/(16*np.pi*mu*a**3))*nu*\
        (r**2*np.arcsin(a/r)+(2*a**2-r**2)*np.sqrt(1-a**2/r**2)*a/r) \
        *(x*y/(r**2))

    return uC


@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def get_u3_TSurf(x, y, a, Qx, mu, nu):
    # Test function. Supposed to give the same result as get_u3_THertz
    # for z = 0
    
    # The surface z displacement for a pure x direction traction qx(x,y)
    # is equal to the surface x displacement for a pure z direction traction
    # p(x,y) = -qx(x,y)
    
    # Formular for the latter can be found
    # Johnson 3.41b/3.42b together with ux = x/r*ur
    
    #rSq[rSq < np.finfo(np.float64).tiny] = np.nan
    #z[z < np.finfo(np.float64).tiny] = np.nan
    r = np.sqrt(x*x +y*y)

    if iszero(r):
        return 0
    elif r <= a:
        return x*((1-2*nu)*Qx/(4*mu*np.pi))*(1/r**2)*(1-np.sqrt(a*a-r*r)**3/(a**3))
    else:
        return x*((1-2*nu)*Qx/(4*mu*np.pi))*(1/r**2)


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qx_TSurf(x, y, a, Qx):
    rSq = x*x +y*y
    if rSq < a*a:
        return (3*Qx/(2*np.pi*a**3))*np.sqrt(a*a-rSq)
    else:
        return 0


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qy_TSurf(_x, _y, _a, _Qx):
    return 0


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def get_qz_TSurf(_x, _y, _a, _Qx):
    return 0


if __name__ == "__main__":
   
    spacing = 0.1
    mcVal = 10
    mcValZ = 5
    
    # Make the grid
    x, y, z = np.meshgrid(np.arange(-mcVal, mcVal, spacing),
                          np.arange(-mcVal, mcVal, spacing),
                          np.arange(0, 2*mcValZ, spacing))
    
    # Make the direction data for the arrows
    a = 2  # µm Peak area radius
    F = 1e-2  # µN Indention force

    E = 10000.0  # 50 kPa
    nu = 0.45
    
    Qx = 1e6*F  # µN -> Pa*um²
    mu = E/(2.0*(1.0 + nu))
    
    TH_Params = (x, y, z, a, Qx, mu, nu)
    
    TS_Params = (x[..., -1], y[..., -1], a, Qx, mu, nu)
    
    TD_Params = (x[:, :, (0, -10)], y[:, :, (0, -10)], z[:, :, (0, -10)], a)
    
    ua = get_u1_THertz(*TH_Params)
    va = get_u2_THertz(*TH_Params)
    wa = get_u3_THertz(*TH_Params)
    
    us = get_u1_TSurf(*TS_Params)
    vs = get_u2_TSurf(*TS_Params)
    ws = get_u3_TSurf(*TS_Params)
    
    def hDataInfo(name, dx, dy, dz):
        print("{} data info".format(name))
        print(" type and dimensions:", type(dx[0, 0, 0]), dx.shape)
        print(" no of datapoints", np.prod(dx.shape), np.prod(dy.shape), np.prod(dz.shape))
        print(" no of Nan-points", len(dx[dx==np.nan]), len(dy[dy==np.nan]), len(dz[dz==np.nan]))
        print(" Peak Values:", np.nanmax(dx), np.nanmax(dy), np.nanmax(dz))
        
    hDataInfo("u", ua, va, wa)
    
    # Now plot
    import matplotlib.pyplot as plt
    
    def plotLayer(imgLayer, title, cBarLabel=None, saveFig=True):
        xLen = imgLayer.shape[0]*spacing
        yLen = imgLayer.shape[1]*spacing
        
        extent = -xLen/2, xLen/2, -yLen/2, yLen/2
        plt.title(title)
        plt.xlabel(r'x/$\mu m$')
        plt.ylabel(r'y/$\mu m$')
        plt.imshow(imgLayer, origin='lower', interpolation='bilinear', extent=extent)
        cbar = plt.colorbar()
        if cBarLabel:
            cbar.set_label(cBarLabel)
        
        if saveFig:
            print("DEBUG:", title, "Min:", np.nanmin(imgLayer), "Max:", np.nanmax(imgLayer))
        
        if saveFig:
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
    
    qxTheo = get_qx_TSurf(x[..., -1], y[..., -1], a, Qx)
    plotLayer(qxTheo, 'qx_theo', 'Pa')
    
    plotLayer(qx - qxTheo, 'qx-qx_theo', r'$\mu m$')