#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:27:10 2021

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)

Autograd verification test for the tangential Hertz profile

Do not use the the functions of this file via an module import, as the special treatment for x=0 has
been removed. Instead use tHertzField.py for this

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

# import numpy as np
import autograd.numpy as np

from autograd import grad, elementwise_grad as egrad

# import sys
# sys.setrecursionlimit(10000)


def csqrt(z):
    r = np.absolute(z)
    re = np.real(z)
    im = np.imag(z)
    ph = np.arctan2(im, re)

    return np.sqrt(r)*(np.cos(ph/2)+1j*np.sin(ph/2))

def clog(z):
    r = np.absolute(z)
    re = np.real(z)
    im = np.imag(z)
    ph = np.arctan2(1. * im, 1. * re)

    return np.log(r) + 1j*ph


def get_u1_THertz(x, y, z, a, Qx, mu, nu):
    z1 = z+1j*a

    rsq = x * x + y * y
    
    aux0=(3.*(rsq*(((3.*(x**2))+(y**2))*z1)))+((8.*(((y**2)-(\
    x**2))*(z*(z1**2))))+(6.*((x-y)*((x+y)*(z1**3.)))))
    aux1=(csqrt(((x**2)+((y**2)+(z1**2)))))*((-16.*(rsq*(((2.\
    *(x**2))+(y**2))*z)))+aux0)
    aux2=((3.*(x**2))+(y**2))*(clog((z1+(csqrt(((x**2)+((y**2)+(z1**2))\
    ))))))
    aux3=((rsq**2))*(z1*(((-2.*z)+z1)*(clog((z1+(csqrt(((\
    x**2)+((y**2)+(z1**2))))))))))
    aux4=(6.*(((y**2)-(x**2))*(z1**4.)))+(aux1+((-3.*(((rsq\
    **2))*aux2))+(-12.*aux3)))
    aux5=(6.*(((y**4.)-(x**4.))*(z1**2)))+((8.*((x-y)*((x+y)*(z*(z1**3.)))\
    ))+aux4)
    #Txx=-0.0208333*(((rsq**-2.)*((12.*(((x**4.)-(y**4.))\
    #*(z*z1)))+aux5)))
    Txx=-(((rsq**-2.)*((12.*(((x**4.)-(y**4.))\
    *(z*z1)))+aux5)))/48

    # Txx0=-0.125*(z1*((-4.*z)+(z1+(((4.*z)+(-2.*z1))*(clog((2.*z1)))))))
    # Txx = wherezero(x**2+y**2,Txx0,Txx)

    Txx = np.imag(Txx)
    
    aux0=((x**2)+((y**2)+(2.*(z1*((-2.*z)+z1)))))*(clog((z1+(csqrt(((x**\
    2)+((y**2)+(z1**2))))))))
    Tzz=0.25*(((((-4.*z)+z1)*(csqrt(((x**2)+((y**2)+(z1**2))))))-\
    aux0))

    # Tzz0=0.25*(z1*((-4.*z)+(z1+(((4.*z)+(-2.*z1))*(clog((2.*z1)))))))
    # Tzz = wherezero(x**2+y**2,Tzz0,Tzz)
    
    Tzz = np.imag(Tzz)
    
    aux0=((x**2)*((6.*(y**2))+(((3.*z)+(-2.*z1))*z1)))+((y**2)*((2.*(y**2)\
    )+(z1*((-3.*z)+(2.*z1)))))
    aux1=(rsq**-2.)*((csqrt(((x**2)+((y**2)+(z1**2)))))*((4.*\
    (x**4.))+aux0))
    aux2=(2.*((x-y)*((x+y)*((rsq**-2.)*(z1**3.)))))+(aux1+(-3.\
    *(z*(clog((z1+(csqrt(((x**2)+((y**2)+(z1**2)))))))))))
    #Txxz=0.166667*((3.*(((y**2)-(x**2))*((rsq**-2.)*(z*(\
    #z1**2)))))+aux2)
    Txxz=((3.*(((y**2)-(x**2))*((rsq**-2.)*(z*(\
    z1**2)))))+aux2)/6


    # Txxz0=0.25*(z+((2.*z1)+(-2.*(z*(clog((2.*z1)))))))
    # Txxz = wherezero(x**2+y**2,Txxz0,Txxz)
    
    Txxz = np.imag(Txxz)
    
    return (1/(2*mu))*(3*Qx/(2*np.pi*a**3))*(2*nu*Txx+2*Tzz-z*Txxz)


def get_u2_THertz(x, y, z, a, Qx, mu, nu):
    z1 = z + 1j*a

    rsq = x * x + y * y
    
    aux0=(-8.*(rsq*z))+((3.*(rsq*z1))+((-8.*(z*(\
    z1**2)))+(6.*(z1**3.))))
    aux1=((rsq**-2.)*((csqrt(((x**2)+((y**2)+(z1**2)))))*\
    aux0))+(-3.*(clog((z1+(csqrt(((x**2)+((y**2)+(z1**2)))))))))
    aux2=(8.*((rsq**-2.)*(z*(z1**3.))))+((-6.*((((x**2)+(y**2)\
    )**-2.)*(z1**4.)))+aux1)
    aux3=y*(((12.*(z*z1))/rsq)+(((-6.*(z1**2))/rsq\
    )+aux2))
    #Txy=-0.0416667*((x*aux3))
    Txy = -(x*aux3)/24
    
    # Txy = wherezero(x**2+y**2,0,Txy)
    Txy = np.imag(Txy)

    aux0=(rsq*(csqrt(((x**2)+((y**2)+(z1**2))))))-(z1*(((-3.*\
    z)+(2.*z1))*((csqrt(((x**2)+((y**2)+(z1**2)))))-z1)))
    #Txyz=0.333333*((x*(y*((rsq**-2.)*aux0))))
    Txyz=(x*(y*((rsq**-2.)*aux0)))/3
    
    # Txyz = wherezero(x**2+y**2,0,Txyz)
    Txyz = np.imag(Txyz)
    
    return (1/(2*mu))*(3*Qx/(2*np.pi*a**3))*(2*nu*Txy-z*Txyz)


def get_u3_THertz(x, y, z, a, Qx, mu, nu):
    z1 = z + 1j*a

    rsq = x * x + y * y
    
    aux0=-3.*(rsq*(z*(clog((z1+(csqrt(((x**2)+((y**2)+(\
    z1**2))))))))))
    aux1=(2.*(rsq*(csqrt(((x**2)+((y**2)+(z1**2)))))))+((z1*(\
    ((-3.*z)+(2.*z1))*((csqrt(((x**2)+((y**2)+(z1**2)))))-z1)))+aux0)
    #Txz=(0.166667*((x*aux1)))/rsq
    Txz=(((x*aux1))/rsq)/6

    
    # Txz = wherezero(x**2+y**2,0,Txz)
    Txz = np.imag(Txz)
    
    aux0=(((-2.*z)+z1)*((csqrt(((x**2)+((y**2)+(z1**2)))))-z1))-(((x**2)+\
    (y**2))*(clog((z1+(csqrt(((x**2)+((y**2)+(z1**2)))))))))
    Txzz=(0.5*((x*aux0)))/rsq
    
    # Txzz = wherezero(x**2+y**2,0,Txzz)
    Txzz = np.imag(Txzz)
    
    return (1/(2*mu))*(3*Qx/(2*np.pi*a**3))*((1-2*nu)*Txz-z*Txzz)


def vgrad(fn, i):
    return np.vectorize(grad(fn, i))


if __name__ == "__main__":
    spacing = 10.
    mcVal = 200.5
    mcValZ = 10.

    z = 0.
    
    # Make the grid
    # x, y, z = np.meshgrid(np.arange(-mcVal, mcVal, spacing),
    #                       np.arange(-mcVal, mcVal, spacing),
    #                       np.arange(0, 2*mcValZ, spacing))
    x, y = np.meshgrid(np.arange(-mcVal, mcVal, spacing), np.arange(-mcVal, mcVal, spacing))
    
    # Make the direction data for the arrows
    b = 120  # µm    Dipole Distance
    a = 40   # µm    Peak area radius
    Qx = 4   # Pa*µm²
    
    nu = 0.49
    E = 20000.0 # Pa
    mu = E/(2.0*(1.0 + nu))
    
    # TH_Params = (x, y, z, a, Qx, mu, nu)

    TH_Params_New = (x, y, z, a, Qx, mu, nu)
    
    # TS_Params = (x[..., -1], y[..., -1], a, Qx, mu, nu)
    
    # TD_Params = (x[:, :, (0, -10)], y[:, :, (0, -10)], z[:, :, (0, -10)], a)


    get_ux = vgrad(get_u1_THertz, 0)
    get_uy = vgrad(get_u1_THertz, 1)
    get_uz = vgrad(get_u1_THertz, 2)

    get_vx = vgrad(get_u2_THertz, 0)
    get_vy = vgrad(get_u2_THertz, 1)
    get_vz = vgrad(get_u2_THertz, 2)

    get_wx = vgrad(get_u3_THertz, 0)
    get_wy = vgrad(get_u3_THertz, 1)
    get_wz = vgrad(get_u3_THertz, 2)
    
    ua = get_u1_THertz(*TH_Params_New)
    va = get_u2_THertz(*TH_Params_New)
    wa = get_u3_THertz(*TH_Params_New)

    ux = get_ux(*TH_Params_New)
    vx = get_vx(*TH_Params_New)
    wx = get_wx(*TH_Params_New)

    uy = get_uy(*TH_Params_New)
    vy = get_vy(*TH_Params_New)
    wy = get_wy(*TH_Params_New)

    uz = get_uz(*TH_Params_New)
    vz = get_vz(*TH_Params_New)
    wz = get_wz(*TH_Params_New)

    def hDataInfo(name, dx, dy, dz):
        print("{} data info".format(name))
        print(" type and dimensions:", type(dx[0,0]), dx.shape)
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

    plotLayer(x, 'x', r'$\mu m$')
    plotLayer(y, 'y', r'$\mu m$')

    plotLayer(ua, 'ux_a', r'$\mu m$')
    plotLayer(va, 'uy_a', r'$\mu m$')
    plotLayer(wa, 'uz_a', r'$\mu m$')

    plotLayer(ux, 'ux_x', r'$\mu m$')
    plotLayer(vx, 'uy_x', r'$\mu m$')
    plotLayer(wx, 'uz_x', r'$\mu m$')

    plotLayer(uy, 'ux_y', r'$\mu m$')
    plotLayer(vy, 'uy_y', r'$\mu m$')
    plotLayer(wy, 'uz_y', r'$\mu m$')

    plotLayer(uz, 'ux_z', r'$\mu m$')
    plotLayer(vz, 'uy_z', r'$\mu m$')
    plotLayer(wz, 'uz_z', r'$\mu m$')

    # uy, ux, uz = np.gradient(ua, spacing)
    # vy, vx, vz = np.gradient(va, spacing)
    # wy, wx, wz = np.gradient(wa, spacing)

    divU = ux+vy+wz

    # The surface normal of the plane is directed outwards, hence
    # n = (0,0,-1). Now use \vec{q} = \vec{n}\cdot\boldsymbol{\sigma}
    # qx = -sigma_xz
    # qy = -sigma_yz
    # qz = -sigma_zz

    qx = -mu*(uz+wx)
    qy = -mu*(vz+wy)
    if (0.5-nu) < 0.001:
        qz = - 2 * mu * wz
    else:
        qz = -mu*(wz+wz+2*nu/(1-2*nu)*divU)

    plotLayer(divU, 'divu', '1')

    plotLayer(qx, 'qx', 'Pa')
    plotLayer(qy, 'qy', 'Pa')
    plotLayer(qz, 'qz', 'Pa')