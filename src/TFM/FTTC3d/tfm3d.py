# Implementation of 3D FTTC related functions

import numpy as np
from scipy.interpolate import griddata, SmoothBivariateSpline
from scipy.sparse import block_diag

from . import numbafn3d as numbafn


def extract_deformation(deformation):
    """ Extract deformation data into two arrays """
    pos = deformation["pos"]
    vec = deformation["vec"]
    return pos.T, vec.T


def interp_vec2grid(pos, vec, mesh_size, i_max = None, j_max = None):
    """ Interpolate pos and vec data to a finer, regular grid """
    max_corner = np.array([np.max(pos[0]), np.max(pos[1])])
    min_corner = np.array([np.min(pos[0]), np.min(pos[1])])
    if i_max is None and j_max is None:
        i_max = np.round((max_corner[0] - min_corner[0]) / mesh_size)
        j_max = np.round((max_corner[1] - min_corner[1]) / mesh_size)
        i_max -= np.int64(np.mod(i_max, 2))
        j_max -= np.int64(np.mod(j_max, 2))

    i_max = np.int64(i_max)
    j_max = np.int64(j_max)
    X, Y = np.meshgrid(min_corner[0] + np.arange(0.5, i_max, 1) * mesh_size,
                       min_corner[1] + np.arange(0.5, j_max, 1) * mesh_size)
    grid_mat = np.array([X, Y])
    ### INTERPOLATION ###
    u = np.array(
        [griddata((pos[0].ravel(), pos[1].ravel()), vec[0].ravel(), (grid_mat[0], grid_mat[1]), method='cubic'),
         griddata((pos[0].ravel(), pos[1].ravel()), vec[1].ravel(), (grid_mat[0], grid_mat[1]), method='cubic'),
         griddata((pos[0].ravel(), pos[1].ravel()), vec[2].ravel(), (grid_mat[0], grid_mat[1]), method='cubic')])
    u = extrapolate_u(grid_mat, u)

    i_bound_size, j_bound_size = 0, 0
    return grid_mat, u, i_max, j_max, i_bound_size, j_bound_size


def extrapolate_u(grid_mat, u):
    """ Remove nan points by extrapolation """
    grid_positions = np.array([grid_mat[0].flatten(), grid_mat[1].flatten()]).T
    displacements = np.array([u[0].flatten(), u[1].flatten(), u[2].flatten()]).T
    mask = np.ones(displacements.shape[0], dtype=np.bool)
    mask[np.where(~np.isnan(displacements[:, 0]))] = False  # pylint: disable=unsubscriptable-object
    valid_pos = grid_positions[np.where(~mask)]  # pylint: disable=unsubscriptable-object
    invalid_pos = grid_positions[np.where(mask)]  # pylint: disable=unsubscriptable-object
    valid_dis = displacements[np.where(~mask)]  # pylint: disable=unsubscriptable-object
    ### EXTRAPOLATION ###
    try:
        sbs_0 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1], valid_dis[:, 0], kx=3, ky=3)
        sbs_1 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1], valid_dis[:, 1], kx=3, ky=3)
        sbs_2 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1], valid_dis[:, 2], kx=3, ky=3)
    except:
        sbs_0 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1], valid_dis[:, 0], kx=1, ky=1)
        sbs_1 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1], valid_dis[:, 1], kx=1, ky=1)
        sbs_2 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1], valid_dis[:, 2], kx=1, ky=1)

    u[0][np.isnan(u[0])] = sbs_0.ev(invalid_pos[:, 0], invalid_pos[:, 1])
    u[1][np.isnan(u[1])] = sbs_1.ev(invalid_pos[:, 0], invalid_pos[:, 1])
    u[2][np.isnan(u[2])] = sbs_2.ev(invalid_pos[:, 0], invalid_pos[:, 1])
    return u


def calculate_fourier_modes(mesh_size, i_max, j_max, lanczos_exp=1):
    """ Return the frequences associated with each sampling point in the Fourier space in 1/pix """
    kx_vec = 2. * np.pi / i_max / mesh_size * np.append(np.arange(0, (i_max // 2)),
                                                        np.arange(-i_max // 2, 0))  # [0:(i_max//2-1) (-i_max//2:-1)];
    ky_vec = 2. * np.pi / j_max / mesh_size * np.append(np.arange(0, (j_max // 2)),
                                                        np.arange(-j_max // 2, 0))  # [0:(j_max//2-1) (-j_max//2:-1)];
    kx, ky = np.meshgrid(kx_vec, ky_vec)
    lanczosx = np.sinc(kx * mesh_size / np.pi) ** lanczos_exp
    lanczosy = np.sinc(ky * mesh_size / np.pi) ** lanczos_exp
    kx[0, 0] = 1  # otherwise (kx**2 + ky**2)**(-1/2.) will be inf
    ky[0, 0] = 1
    return kx, ky, lanczosx, lanczosy


def calculate_greens_function(E, s, kx, ky, i_max, j_max, mesh_size, pix_per_mu, z = 0.0):
    """
    Calculate Greens function in Fourier space
    k is given in unit [1 / pix].
    z is given in unit [pix]
    """
    V = 2 * (1 + s) / E
    kx_sq = kx ** 2
    ky_sq = ky ** 2
    kabs = np.sqrt(kx_sq + ky_sq)
    kabs_sq = kx_sq + ky_sq
    h = 2. * s + z * kabs

    # otherwise (kx**2 + ky**2)**(-1/2.) will be inf
    kabs[0, 0] = 1.

    GFt = V / (2 * kabs ** 3) * np.exp(-kabs * z) * np.array([
        [2 * kabs_sq - h * kx_sq, -h * kx * ky, (1 - h) * 1j * kabs * kx],
        [-h * kx * ky, 2 * kabs_sq - h * kx_sq, (1 - h) * 1j * kabs * ky],
        [-(1 - 2 * s + kabs * z) * 1j * kabs * kx, -(1 - 2 * s + kabs * z) * 1j * kabs * ky,
         (2 - 2 * s + kabs * z) * kabs_sq]
    ])

    # we assume that all the sources of traction are in the field of view
    GFt[:, :, 0, 0] = 0.0
    return GFt


def calculate_Ginv(GFt, L):
    """ Calculates L2 regularized inverse of FTGmn """
    Ginv = numbafn.calculate_traction_3d(GFt, L ** 2)
    return Ginv


def reg_fourier_TFM_L2(u, Ginv):
    """ Calculate fouier transformed traction field using a L2 regularized FTTC """
    
    def _comp_i(Ftux, Ftuy, Ftuz, Ginv_i):
        Ginv_ix, Ginv_iy, Ginv_iz = Ginv_i
        return Ginv_ix * Ftux + Ginv_iy * Ftuy + Ginv_iz * Ftuz

    Ftux = np.fft.fft2(u[0])
    Ftuy = np.fft.fft2(u[1])
    Ftuz = np.fft.fft2(u[1])
    Ftfx = _comp_i(Ftux, Ftuy, Ftuz, Ginv[0])
    Ftfy = _comp_i(Ftux, Ftuy, Ftuz, Ginv[1])
    Ftfz = _comp_i(Ftux, Ftuy, Ftuz, Ginv[2])
    return Ftfx, Ftfy, Ftfz


def reconstruct_displacement_field(GFt, Ftfx, Ftfy, Ftfz, lanczosx, lanczosy):
    """ Use calculated traction field to find the corresponding (theoretical) deformation field """
    
    def _comp_i(Ftfx, Ftfy, Ftfz, GFt_i):
        GFt_ix, GFt_iy, GFt_iz = GFt_i
        return GFt_ix * Ftfx + GFt_iy * Ftfy + GFt_iz * Ftfz

    Ftux_rec = _comp_i(Ftfx, Ftfy, Ftfz, GFt[0])
    Ftuy_rec = _comp_i(Ftfx, Ftfy, Ftfz, GFt[0])
    Ftuz_rec = _comp_i(Ftfx, Ftfy, Ftfz, GFt[0])
    ux_rec = np.fft.ifft2(lanczosx * Ftux_rec)
    uy_rec = np.fft.ifft2(lanczosy * Ftuy_rec)
    uz_rec = np.fft.ifft2(Ftuz_rec)
    urec = np.array([np.real(ux_rec), np.real(uy_rec), np.real(uz_rec)])
    Fturec = np.array([Ftux_rec, Ftuy_rec, Ftuz_rec])
    return urec, Fturec


def calculate_stress_field(Ftfx, Ftfy, Ftfz, lanczosx, lanczosy, grid_mat, u, i_max, j_max, \
                           i_bound_size, j_bound_size, pix_per_mu, mesh_size):
    """ Reconstruct real space deformation field. Use lanczos filter to suppress Fouier artifacts """
    
    fx = np.fft.ifft2(lanczosx * Ftfx)
    fy = np.fft.ifft2(lanczosy * Ftfy)
    fz = np.fft.ifft2(Ftfz)

    pos = np.array([np.reshape(grid_mat[0], (i_max * j_max)), \
                    np.reshape(grid_mat[1], (i_max * j_max))])
    vec = np.array([np.reshape(u[0], (i_max * j_max)),
                    np.reshape(u[1], (i_max * j_max)),
                    np.reshape(u[2], (i_max * j_max))])

    f = np.array([np.real(fx), np.real(fy), np.real(fz)])
    fnorm = (f[0] ** 2 + f[1] ** 2 + f[2] ** 2) ** 0.5
    if j_bound_size > 0 and i_bound_size > 0:
        energy = calculate_energy(u[:, j_bound_size:-j_bound_size, i_bound_size:-i_bound_size], \
                                  f[:, j_bound_size:-j_bound_size, i_bound_size:-i_bound_size], pix_per_mu, mesh_size)
        force = calculate_total_force(fnorm[j_bound_size:-j_bound_size, i_bound_size:-i_bound_size], pix_per_mu,
                                      mesh_size)
    else:
        energy = calculate_energy(u, f, pix_per_mu, mesh_size)
        force = calculate_total_force(fnorm, pix_per_mu, mesh_size)
    return pos, vec, fnorm, f, energy, force


def calculate_energy(u, f, pix_per_mu, mesh_size):
    """ Determine energy stored in the given traction profile """
    l = mesh_size / pix_per_mu * 1e-6  # nodal distance in the rectangular grid in m**2 -> dA = l**2
    energy = 0.5 * l ** 2 * np.sum(
        u * f) * 1e-6 / pix_per_mu  # u is given in pix -> additional 1e-6 / pix_per_mu, f is given in Pa
    return energy


def calculate_total_force(fnorm, pix_per_mu, mesh_size):
    """ Calculate L2 norm of the generated deformation field """
    l = mesh_size / pix_per_mu * 1e-6  # nodal distance in the rectangular grid in m**2 -> dA = l**2
    total_force = l ** 2 * np.sum(fnorm)
    return total_force


def do_TFM(deformations, nr, mesh_size, E, nu, pix_per_mu, lam, lanczos_exp):
    """ Reconstruct traction forces using a L2 regularized FTTC calculation """
    pos, vec = extract_deformation(deformations[nr])
    grid_mat, u, i_max, j_max, i_bound_size, j_bound_size = interp_vec2grid(pos, vec, mesh_size)
    kx, ky, lanczosx, lanczosy = calculate_fourier_modes(mesh_size, i_max, j_max, lanczos_exp)
    GFt = calculate_greens_function(E, nu, kx, ky, i_max, j_max, mesh_size, pix_per_mu)

    G_inv = calculate_Ginv(GFt, lam)
    Ftfx, Ftfy, Ftfz = reg_fourier_TFM_L2(u, G_inv)

    Ftf = np.array([Ftfx, Ftfy, Ftfz])
    urec, Fturec = reconstruct_displacement_field(GFt, Ftfx, Ftfy, Ftfz, lanczosx, lanczosy)
    pos, vec, fnorm, f, energy, force = calculate_stress_field(Ftfx, Ftfy, Ftfz,
                                                               lanczosx, lanczosy, grid_mat, u, i_max, j_max,
                                                               i_bound_size, j_bound_size, pix_per_mu, mesh_size)
    return pos, vec, fnorm, f, urec, u, i_max, j_max, energy, force, Ftf, Fturec, i_bound_size, j_bound_size


def svd(deformations, framenr, E, nu, mesh_size, pix_per_mu, lanczos_exp):
    """
    Prepare svd representation of the FTTC problem that can be used to quickly calculate
    the GCV function
    """
    pos, vec = extract_deformation(deformations[framenr])
    grid_mat, u, i_max, j_max, i_bound_size, j_bound_size = interp_vec2grid(pos, vec, mesh_size)
    kx, ky, lanczosx, lanczosy = calculate_fourier_modes(mesh_size, i_max, j_max, lanczos_exp)
    GFt = calculate_greens_function(E, nu, kx, ky, i_max, j_max, mesh_size, pix_per_mu)

    # Deleate large data object
    del grid_mat, kx, ky

    Ftu = np.fft.fft2(u).reshape(3, -1).T
    shape = GFt[0, 0].shape

    # Perform flattening in Fourier mode space along the way
    U_h = np.empty((shape[0] * shape[1], 3, 3), dtype=np.complex128)
    s_h = np.empty((shape[0] * shape[1], 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = i * shape[1] + j
            U_h[idx, :], s_h[idx, :], _V = np.linalg.svd(GFt[:, :, i, j])

    # Convert to Output format.
    b = Ftu.flatten()  # = [u(k,l).T]
    s = s_h.flatten()  # = [s(k,l).T]
    sparseU = block_diag(U_h, format="csr")
    return sparseU, s, b
