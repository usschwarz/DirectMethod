""" Single Call implementation of regFTTC in 2D """

import numpy as np

# HACK to allow script executions for unit tests
if __name__ == "__main__":  # This is Python

    # Ensure all of these are loaded in a module like fashon
    import os
    import sys

    # Import parent dir to ensure module-like import
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    # noinspection PyUnresolvedReferences
    from FTTC import tfm
    # noinspection PyUnresolvedReferences
    from FTTC import gcv_block
else:
    from . import tfm
    from . import gcv_block


def plot_gcv_curve(lam, minG, lams, Gs):
    ax1 = plt.gca()
    ax1.plot(lams, Gs)
    ax1.plot([lam], [minG], "*r")
    ax1.plot([lam, lam], [minG / 1000, minG], ":r")
    ax1.set_xlim((lams[0], lams[-1]))
    ax1.set_ylim((minG / 1000, 10 * np.max(Gs)))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\lambda$", fontsize=10)
    ax1.set_ylabel(r"$G(\lambda)$", fontsize=10)
    ax1.text(
        0.75,
        0.90,
        "GCV Function",
        horizontalalignment="center",
        fontsize=10,
        transform=ax1.transAxes,
    )
    for tick in ax1.get_xticklabels():
        tick.set_fontsize(10)
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(10)
    plt.show()


def create_gcv_par(simdata, lamlow, lamhigh, lamcount=20):
    """ Determines FTTC regularization parameter using GCV  """

    pos0, vec0, E, nu, mesh_size, pix_per_mu, lanczos_exp = simdata
    lambdarange = np.logspace(lamlow, lamhigh, lamcount)
    blockU, s, b = tfm.svd_block(pos0, vec0, E, nu, mesh_size, lanczos_exp)
    # Reg gcv can find points outside of this range as well
    reg_min, minG, G, _reg_param = gcv_block.gcv_blockdiag(blockU, s, b, lambdarange, plot=False)
    lam = reg_min
    print('GCV minimum at lambda =', lam)
    # plot_gcv_curve(lam, minG, lambdarange, G)
    return lam


def perform_FTTC(xR, yR, u, v, dm, E, nu, lanczos_exp=1, set_lam=None):
    """ Determine traction field using the 2D regularized FTTC method """
    # We use a subsampling of 4 "Pixels" per input mesh spacing module
    print("Performing fttc")
    mesh_size = 4
    pix_per_mu = mesh_size / dm
    xpixR = np.asanyarray(xR) * pix_per_mu
    ypixR = np.asanyarray(yR) * pix_per_mu

    xpix, ypix = np.meshgrid(xpixR, ypixR, indexing='ij')

    pos0 = np.array([xpix.flatten(), ypix.flatten()])
    vec0 = pix_per_mu * np.array([u.flatten(), v.flatten()])


    # If lam is not known find one using gcv
    if set_lam is None:
        lamguess = 0.2 / E
        lamlow = np.log10(lamguess) - 5.0
        lamhigh = np.log10(lamguess) + 5.0
        lamcount = 50

        simData = pos0, vec0, E, nu, mesh_size, pix_per_mu, lanczos_exp
        lam = create_gcv_par(simData, lamlow, lamhigh, lamcount)
    else:
        lam = set_lam

    # Returns:
    pos, vec, fnorm, f, urec, u, i_max, j_max, energy, force, Ftf, Fturec, _, _ = \
        tfm.do_TFM(pos0, vec0, mesh_size, E, nu, pix_per_mu, lam, lanczos_exp)

    x = np.reshape(pos[0], (i_max, j_max)).T / pix_per_mu
    y = np.reshape(pos[1], (i_max, j_max)).T / pix_per_mu

    # raise RuntimeError
    return (x, y), fnorm, f, urec, u, energy, force, Ftf, Fturec


if __name__ == "__main__":
    # Unit test code following
    xyR = np.linspace(-5, 5)
    dm = xyR[1] - xyR[0]
    x, y = np.meshgrid(xyR, xyR, indexing='ij')
    E = 1e5
    nu = 0.5
    r = np.sqrt(x ** 2 + y ** 2)

    u = np.exp(-x ** 2 - y ** 2 / 0.5)
    v = np.exp(-x ** 2 - y ** 2 / 4)

    print(u.shape)

    import matplotlib.pyplot as plt


    def myPlot(Z):
        plt.imshow(Z.T, interpolation='bilinear',
                   origin='lower', extent=[-5, 5, -5, 5],
                   vmax=abs(Z).max(), vmin=-abs(Z).max())
        plt.show()


    myPlot(u)
    myPlot(v)

    xy, pos, vec, fnorm, f, urec, u, i_max, j_max, energy, force, Ftf, Fturec = \
        perform_FTTC(xyR, xyR, u, v, dm, E, nu)
    print(f.shape, u.shape, urec.shape)
    myPlot(f[0])
    myPlot(f[1])
    myPlot(urec[0])
