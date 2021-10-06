"""
Helperscript for consistend figure layout
"""

import matplotlib.pyplot as plt


def figure_setup():
    """
    Generates and returns a matplotlib figure using proper configurations

    Returns:
        plt.figure and plt.axes
    """

    # figure settings
    figure_width = 17.526  # cm
    figure_height = 11.43  # cm
    left_margin = 3  # cm
    right_margin = 0.526  # cm
    top_margin = 0.93  # cm
    bottom_margin = 2  # cm

    # Don't change
    left = left_margin / figure_width
    bottom = bottom_margin / figure_height

    width = 1 - (left_margin + right_margin) / figure_width
    height = 1 - (top_margin + bottom_margin) / figure_height

    cm2inch = 1/2.54  # inch per cm

    fig = plt.figure(figsize=(figure_width*cm2inch, figure_height*cm2inch))
    ax = fig.add_axes((left, bottom, width, height))

    return fig, ax

