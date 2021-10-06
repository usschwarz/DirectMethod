"""
Fit points using a linear sheme

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import numpy as np
import numba


def fitMatSetup():
    # This functions calculates the fit matrix FitMat
    # that is needed for the above function lsqPolyFit3
    # FitMat = (transp(M).M)^-1.transp(M)
    # where M is the 3d Vandermonde matrix for f given by
    # f(x) = a0 + a1*x + a2*y + a3*z # + a4*x²+ a5*xy + a6*xz + a7*y²+ a8*yz + a9*z²
    # Multiply FitMat onto a vector that contains f(x) for a 3x3x3 region, to get the ai coefficent vector

    x, y, z = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
    x, y, z = x.flatten(), y.flatten(), z.flatten()

    # tpV = M.T
    tpV = np.stack(
        [np.ones_like(x), x, y, z])  # , x**2, x * y, x * z, y**2, y * z, z**2])

    # M is the 3d Vandermonde matrix for f
    # Hence mimimize||b-M.a||_2
    # To find a lsq Fit coefficients, we need to solve
    # transp(M).b =  transp(M).M.a
    # This can be rewritten as:
    # a = FitMat.b where FitMat = (transp(M).M)^-1.transp(M)
    return np.linalg.inv(tpV @ np.transpose(tpV)) @ tpV


def gradientFitArea(f, *varargs, axis=None):
    # This function fits each point using a fitting matrix.
    # A point must have nearest neighbors in each spacial direction.
    # Otherwise center around the nearest neigboring point that has

    # Prolog
    assert (f.ndim == 3)
    if axis is None:
        axis = np.array([0, 1, 2])
    axis = np.atleast_1d(axis)
    if len(varargs) == 0:
        spacing = np.ones(len(axis))
    elif len(varargs) == 1:
        spacing = varargs[0] * np.ones(len(axis))
    else:
        spacing = np.asanyarray(varargs)
        assert (len(spacing) == len(axis))
    # Spacing is only used in the end to rescale the results
    # Until the we assume spacing==1

    # If a dimension contains less then 2 pixels we run into troubles
    assert (f.shape[0] >= 3)
    assert (f.shape[1] >= 3)
    assert (f.shape[2] >= 3)

    @numba.njit
    def mainloop(f, fitMat, i1, i2, i3):
        # Flatten all arrays to get a string of items
        i1 = np.ravel(i1)
        i2 = np.ravel(i2)
        i3 = np.ravel(i3)

        fx = np.empty_like(f)
        fy = np.empty_like(f)
        fz = np.empty_like(f)

        for el in range(len(i1)):
            # j represents the centering element
            curr = np.array([i1[el], i2[el], i3[el]])

            # off represents the offset between the fit center
            # element and the selected element
            off = np.zeros_like(curr)

            # For boundary elements center analysis around a better positioned one
            for i in range(3):
                if curr[i] == 0:
                    curr[i] = 1
                    off[i] = -1
                elif curr[i] == f.shape[i] - 1:
                    curr[i] -= 1
                    off[i] = 1

            # Get support for current fit
            b = (f[curr[0] - 1:curr[0] + 2, curr[1] - 1:curr[1] + 2, curr[2] - 1:curr[2] + 2]).flatten()

            # Get coefficent vector
            # f(x) = a0 + a1*x + a2*y + a3*z #+ a4*x²+ a5*xy + a6*xz + a7*y²+ a8*yz + a9*z²
            a = np.dot(fitMat, b)

            # Use coeficents to determine current derivative
            fx[i1[el], i2[el], i3[el]] = a[1]
            fy[i1[el], i2[el], i3[el]] = a[2]
            fz[i1[el], i2[el], i3[el]] = a[3]

        # Return stuff
        return fx, fy, fz

    # Prepare fit mat
    fitMat = fitMatSetup()

    # Find ux,uy,uz via fitting
    i1, i2, i3 = np.mgrid[0:f.shape[0], 0:f.shape[1], 0:f.shape[2]]
    ux, uy, uz = mainloop(f, fitMat, i1, i2, i3)

    # Rescale and return
    ulist = [ux, uy, uz]
    if len(axis) == 1:
        return ulist[axis[0]] / spacing[0]
    return [ulist[i] / spacing[i] for i in axis]


if __name__ == "__main__":
    rng = np.linspace(-5, 5, 100)
    x, y, z = np.meshgrid(rng, rng, rng, indexing='ij')

    f = np.sin(x) * (y * y + z * z)

    fx, fy, fz = gradientFitArea(f)

    import matplotlib.pyplot as plt

    plt.imshow(f[..., 50])
    plt.show()
    plt.imshow(fx[..., 50])
    plt.show()
    plt.imshow(fy[..., 50])
    plt.show()
    plt.imshow(fz[..., 50])
    plt.show()
