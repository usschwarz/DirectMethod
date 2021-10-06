"""
Helper routines for output plotting

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from matplotlib_scalebar.scalebar import ScaleBar


def plot_with_arrows(fig, xRng, yRng, fx, fy, fz, cbar_label="Stress [Pa]"):
    """ plot stress field with arrows for fx, fy and cmap for a the third component """
    # print(xRng.shape,yRng.shape)
    # Color Plot
    gs = GridSpec(100, 130)
    ax1 = fig.add_subplot(gs[:100, :100])
    # left right bottom top
    extent = [xRng[0], xRng[-1], yRng[0], yRng[-1]]

    # print(extent)

    ax1.set_xlim((xRng[0], xRng[-1]))
    ax1.set_ylim((yRng[0], yRng[-1]))
    ax1.axis('off')
    if np.sum(np.abs(fz)) == 0.0:  # Yes, only if this is really 0.0
        # Special case of an empty plot
        im = ax1.imshow(
            fz.T, origin="lower", extent=extent, interpolation="bilinear", vmin=-10, vmax=10
        )
    else:
        im = ax1.imshow(fz.T, origin="lower", extent=extent, interpolation="bilinear")

    # scalebar = AnchoredSizeBar(ax1.transData,20,'20 µm','lower right')
    scalebar = ScaleBar(xRng[1] - xRng[0], 'um', location='lower right', box_alpha=0., color='0.8', scale_loc='top')
    ax1.add_artist(scalebar)
    # xLen = xRng[1] - xRng[0]
    # yLen = yRng[1] - yRng[0]

    # patchLock = (xLen/2*0.9,-yLen/2*0.9)
    # rect = patches.Rectangle(patchLock,-20,yLen/2*0.05,linewidth=1,edgecolor='none',facecolor='0.8')
    # ax1..add_patch(rect)

    axes = fig.add_subplot(gs[:100, 103:106])
    cbar = plt.colorbar(im, cax=axes)
    cbar.set_label(cbar_label)  # , fontsize=16)

    # ARROW PLOT #
    px, py, redx, redy = calculate_arrows(xRng, yRng, fx, fy)
    scale = np.sqrt(redx ** 2 + redy ** 2).max() * 20
    # self.scale = np.sqrt(redx ** 2 + redy ** 2).max() * 25.0
    q = ax1.quiver(px, py, redx, redy, color="w", headwidth=4, headlength=6,
                   scale=scale)  # pivot="middle", scale=self.scale)
    ax1size = len(ax1.collections)
    fig.subplots_adjust(bottom=0.13, top=0.92, left=0.11, right=0.88)
    # print("Animation took", time.time() - start, "seconds")
    return ax1


def ceilidiv(x, y):
    """ Calculates ceil(x/y) """
    return (x - 1) // y + 1


def calculate_arrows(xRng, yRng, fx, fy):
    """ Determine error field """
    maxcount = 40
    maxlen = 10

    def _range_interp(rnge, n, leni):
        picka = rnge[:leni:n]
        pickb = rnge[-leni + (n - 1)::n]
        return (picka + pickb) / 2

    def _field_interp(fun, n, len1, len2):
        picka = fun[:len1:n, :len2:n]
        pickb = fun[-len1 + (n - 1)::n, -len2 + (n - 1)::n]
        return (picka + pickb) / 2

    if len(xRng) < maxcount and len(yRng) < maxcount:
        # Number of arrors okay:
        px, py = np.meshgrid(xRng, yRng, indexing="ij")
        return px, py, fx, fy
    else:
        # Trimdown is needed
        scaleX = ceilidiv(len(xRng), maxcount)
        scaleY = ceilidiv(len(yRng), maxcount)
        n = max(scaleX, scaleY)
        len1 = ceilidiv(fx.shape[0], n) * n
        len2 = ceilidiv(fx.shape[1], n) * n
        redx = _field_interp(fx, n, len1, len2)
        redy = _field_interp(fy, n, len1, len2)
        pxrng = _range_interp(xRng, n, len1)
        pyrng = _range_interp(yRng, n, len2)
        px, py = np.meshgrid(pxrng, pyrng, indexing="ij")
        return px, py, redx, redy
