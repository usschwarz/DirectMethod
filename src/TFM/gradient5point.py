# 5 Point gradient gradient function
# Formulae can be found by fitting a 5th order polynomial using the value at 5 equally distanced sampling points
#

import numpy as np


def partD5pSym(x, dim, spacing):
    """
    Calculates the partial derivative on the grid sampled field x using a 5 point formular
    Params:
        x       - grid sampled vectorfield
        dim     - coordinate in which the partial derivative is taken
        spacing - grid spacing in the coordinate direction in which the derivative is taken.

    Returns:
        partial derivative on x in dim direction
    """

    # Temporarily swap concerned axis to the front so we
    d = np.empty_like(x)
    n = x.shape[dim]
    assert (n >= 5)  # Otherwise this will fail

    # Generate Views where the concerned axis is in the top
    # place. This allows fast slice selection
    # Notice that we do this also for d, this increases a likelyhood that the returned
    # array has a normal memory layout.
    xS = x.swapaxes(0, dim)
    dS = d.swapaxes(0, dim)

    dS[2:-2] = xS[0:-4] / 12 - xS[1:-3] * 2 / 3 + xS[3:-1] * 2 / 3 - xS[4:] / 12

    dS[0] = -xS[0] * 25 / 12 + xS[1] * 4 - xS[2] * 3 + xS[3] * 4 / 3 - xS[4] / 4
    dS[1] = -xS[0] / 4 - xS[1] * 5 / 6 + xS[2] * 3 / 2 - xS[3] * 1 / 2 + xS[4] / 12

    dS[-2] = xS[-1] / 4 + xS[-2] * 5 / 6 - xS[-3] * 3 / 2 + xS[-4] * 1 / 2 - xS[-5] / 12
    dS[-1] = xS[-1] * 25 / 12 - xS[-2] * 4 + xS[-3] * 3 - xS[-4] * 4 / 3 + xS[-5] / 4

    d = dS.swapaxes(0, dim)

    return d / spacing


def gradient5point(x, *varargs, axis=None):
    """
    Returns gradient calculated using a 5 point derivative.
    Interface is designed to be equivalent to np.gradient
    """
    if axis is None:
        axis = np.arange(len(x.shape))
    else:
        axis = np.ravel(axis)

    if len(varargs) == 0:
        spacing = np.ones(len(axis))
    elif len(varargs) == 1:
        spacing = varargs[0] * np.ones(len(axis))
    else:
        spacing = np.array(varargs)
        assert (len(spacing) == len(axis))

    a = []
    for i in range(len(axis)):
        a.append(partD5pSym(x, axis[i], spacing[i]))

    if len(a) == 1:
        return a[0]
    else:
        return a
