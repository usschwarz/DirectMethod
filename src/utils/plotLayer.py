"""
Small helper function to visualize layers

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""


import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import integrate
from .arrowplot import plot_with_arrows


def plotLayer(
        imgLayer, xSpacing, ySpacing, title, cBarLabel=None,
        saveFig=True, folder="plots", savetitle=None
):
    xLen = imgLayer.shape[0] * xSpacing
    yLen = imgLayer.shape[1] * ySpacing

    extent = -xLen / 2, xLen / 2, -yLen / 2, yLen / 2

    fig = plt.figure()
    ax = fig.gca()

    ax.axis('off')

    if np.sum(np.abs(imgLayer)) == 0.0:  # Yes, only if this is really 0.0
        # Special case of an empty plot
        im = ax.imshow(
            imgLayer.T, origin='lower', interpolation='bilinear', extent=extent, vmin=-10, vmax=10
        )
    else:
        im = ax.imshow(imgLayer.T, origin='lower', interpolation='bilinear', extent=extent)

    patchLock = (xLen / 2 * 0.9, -yLen / 2 * 0.9)

    rect = patches.Rectangle(
        patchLock, -20, yLen / 2 * 0.05, linewidth=1, edgecolor='none', facecolor='0.8'
    )

    # Add the patch to the Axes
    ax.add_patch(rect)

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)

    if cBarLabel:
        cbar.set_label(cBarLabel)

    fig.tight_layout()
    ax.autoscale()

    total = integrate.simps(integrate.simps(imgLayer, dx=ySpacing), dx=xSpacing)
    total2 = np.sum(imgLayer) / (xSpacing * ySpacing)

    str1 = "Peak Information for {}: Min: {} Max: {}".format(title, np.nanmin(imgLayer), np.nanmax(imgLayer))
    print(str1)

    if cBarLabel == "Pa":
        str2 = "Total Surface Force {} Pa µm²$".format(total)
    else:
        str2 = "Surface Integral value {}".format(total)

    print(str2)

    if cBarLabel == "Pa":
        print("Total Surface Force (summed) {} Pa µm²$".format(total2))
    else:
        print("Surface Sum value (summed) {}".format(total2))

    if saveFig:
        if savetitle is None:
            savetitle = title
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig('{}/plot-{}.pdf'.format(folder, savetitle))
        fig.canvas.draw_idle()   # need this if 'transparent=True' to reset colors

        with open('{}/info-{}.txt'.format(folder, savetitle), 'w') as fd:
            # f.write("Peak Information for {}:".format(title), "Min:",np.nanmin(imgLayer), "Max:", np.nanmax(imgLayer))
            print(str1, file=fd)
            print(str2, file=fd)
    # plt.title(title)
    fig.canvas.set_window_title(title)
    plt.show()


def plotLayerWithArrows(
        grid, xcomp, ycomp, zcomp, title, cBarLabel=None,
        saveFig=True, folder="plots", savetitle=None
):
    plt.rcParams.update({'font.size': 25})

    fig = plt.figure()
    plt.axis('off')
    if grid[0][1, 0] > grid[0][0, 0]:
        # 'ij' indexing
        xRng = grid[0][:, 0]
        yRng = grid[1][0, :]
        plot_with_arrows(fig, xRng, yRng, xcomp, ycomp, zcomp, cbar_label=cBarLabel)
    elif grid[1][1, 0] > grid[1][0, 0]:
        # 'xy' indexing
        xRng = grid[0][0, :]
        yRng = grid[1][:, 0]
        plot_with_arrows(fig, xRng, yRng, xcomp.T, ycomp.T, zcomp.T, cbar_label=cBarLabel)

    if saveFig:
        if savetitle is None:
            savetitle = title
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig('{}/plot-{}.pdf'.format(folder, savetitle))
    fig.canvas.set_window_title(title)
    plt.show()
    plt.close()


def plotLayerOld(imgLayer, xSpacing, ySpacing, title, cBarLabel=None, saveFig=True, folder="plots"):
    if saveFig:
        plotLayer4Save(imgLayer, xSpacing, ySpacing, title, cBarLabel, folder="plots")
    plotLayer(imgLayer, xSpacing, ySpacing, title, cBarLabel, saveFig=False, folder="plots")


def plotLayer4Save(imgLayer, xSpacing, ySpacing, title, cBarLabel=None, folder="plots"):
    plt.axis('off')

    if np.sum(np.abs(imgLayer)) == 0.0:  # Yes, only if this is really 0.0
        # Special case of an empty plot
        plt.imshow(imgLayer.T, origin='lower', interpolation='bilinear', vmin=-10, vmax=10)
    else:
        plt.imshow(imgLayer.T, origin='lower', interpolation='bilinear')
    plt.autoscale()
    plt.tight_layout()
    cbar = plt.colorbar()
    if cBarLabel:
        cbar.set_label(cBarLabel)

    total = integrate.simps(integrate.simps(imgLayer, dx=ySpacing), dx=xSpacing)

    str1 = "Peak Information for {}: Min: {} Max: {}".format(title, np.nanmin(imgLayer), np.nanmax(imgLayer))
    print(str1)

    if cBarLabel == "Pa":
        str2 = "Total Surface Force {} Pa µm²$".format(total)
    else:
        str2 = "Surface Integral value {}".format(total)

    print(str2)

    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig('{}/plot-{}.pdf'.format(folder, title), bbox_inches='tight')
    print("Saved to", '{}/plot-{}.pdf'.format(folder, title))

    with open('{}/info-{}.txt'.format(folder, title), 'w') as f:
        print(str1, file=f)
        print(str2, file=f)

    plt.close()
    plt.cla()
