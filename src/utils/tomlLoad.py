"""
This file is used to load a descriptor toml and return it as a pointList

For reference:
Point List structure
[(x,y,a,Qx,Qy,Qz)]
 x,y     Center of adheasion coordinates (in µm)
 a       Size of traction area (in µm)
 Qx      Total traction force applied in x (tangential) direction (in Pa µm^2)
 Qy      Total traction force applied in y (tangential) direction (in Pa µm^2)
 Qz      Total traction force applied in z (normal)     direction (in Pa µm^2)

@author: Johannes Blumberg (johannes.blumberg@bioquant.uni-heidelberg.de)
"""


import toml
import math


def get_unit_to_Paum2(name):
    """ Get unit conversion factor to Pa*µm^2 for various force units """
    if name == "N":
        return 1e12  # 1 N = 10^(0) N = 10^(0) Pa m^2 = 10^(2*6) Pa µm^2 = 10^12 Pa µm^2
    elif name == "mN":
        return 1e9  # 1 mN = 10^(-3) N = 10^(-3) Pa m^2 = 10^(-3+2*6) Pa µm^2 = 10^9 Pa µm^2
    elif name == "µN":
        return 1e6  # 1 µN = 10^(-6) N = 10^(-6) Pa m^2 = 10^(-6+2*6) Pa µm^2 = 10^6 Pa µm^2
    elif name == "uN":
        return 1e6  # 1 µN = 10^(-6) N = 10^(-6) Pa m^2 = 10^(-6+2*6) Pa µm^2 = 10^6 Pa µm^2
    elif name == "nN":
        return 1e3  # 1 nN = 10^(-9) N = 10^(-9) Pa m^2 = 10^(-9+2*6) Pa µm^2 = 10^3 Pa µm^2
    elif name == "pN":
        return 1e0  # 1 pN = 10^(-12) N = 10^(-12) Pa m^2 = 10^(-12+2*6) Pa µm^2 = 10^0 Pa µm^2
    elif name == "Pa µm^2":
        return 1
    elif name == "Pa µm2":
        return 1
    elif name == "Pa um^2":
        return 1
    elif name == "Pa um2":
        return 1
    elif name == "Paµm2":
        return 1
    elif name == "Paum2":
        return 1
    else:
        raise RuntimeError('Unit "{}" not known'.format(name))


def get_unit_to_Pa(name):
    """ Get unit conversion factor to Pa for various pressure units """
    if name == "Pa":
        return 1
    elif name == "kPa":
        return 1e3  # 1 µN = 10^(-3) N = 10^(-3) Pa m^2 = 10^(-3+2*6) Pa µm^2 = 10^9 Pa µm^2
    elif name == "MPa":
        return 1e6  # 1 µN = 10^(-6) N = 10^(-6) Pa m^2 = 10^(-6+2*6) Pa µm^2 = 10^6 Pa µm^2
    else:
        raise RuntimeError('Unit "{}" not supported'.format(name))


def getFource(x, unit):
    """ Convert number unit tuple to the force to a unitless value in Pa µm^2 """
    if unit is None:
        # Default: µN
        # 1 µN = 10^(-6) N = 10^(-6) Pa m^2 = 10^(-3+2*6) Pa µm^2 = 10^6 Pa µm^2
        return 1e6 * float(x)

    factor = get_unit_to_Paum2(unit)
    return factor * float(x)


def loadDataDescription(tomlpath="description.toml", silent=False):
    """ Load dataset metadata from toml file """
    if not silent:
        print("Loading description", tomlpath)
    desc_obj = toml.load(tomlpath)

    try:
        substrate = desc_obj['substrate']
        fname = desc_obj['dataset']['name']

        E = float(substrate['E'])
        nu = float(substrate['nu'])

        if 'spacing_xy' in desc_obj['image']:
            spacing_xy = float(desc_obj['image']['spacing_xy'])
            spacing_z = float(desc_obj['image']['spacing_z'])
        else:
            micronsPerPixel = float(desc_obj['image']['micronsPerPixel'])
            dLayers = float(desc_obj['image']['layerDistance'])

            spacing_xy = 8 * micronsPerPixel
            spacing_z = 8 * dLayers
    except KeyError:
        print("toml file '{}' couldn't be read successfully.".format(tomlpath))
        raise

    unit = substrate.setdefault('E_unit', 'Pa')
    E *= get_unit_to_Pa(unit)

    return fname, E, nu, spacing_xy, spacing_z


def loadSimulationData(tomlpath="description.toml", silent=False):
    """ Load simulation metadata from toml file """
    if not silent:
        print("Loading description", tomlpath)
    desc_obj = toml.load(tomlpath)

    try:
        simulation = desc_obj['simulation']
        if 'n_points_z' in desc_obj['simulation']:
            n_points_z = float(desc_obj['simulation']['n_points_z'])
            n_points_xy = float(desc_obj['simulation']['n_points_xy'])
        else:
            nLayers = float(desc_obj['simulation']['numberOfLayers'])
            xyPix = float(desc_obj['simulation']['xyPix'])

            if nLayers % 8 == 0:
                n_points_z = nLayers // 8
            else:
                n_points_z = nLayers / 8

            if xyPix % 8 == 0:
                n_points_xy = xyPix // 8
            else:
                n_points_xy = xyPix / 8

    except KeyError:
        print("toml file '{}' couldn't be read successfully.".format(tomlpath))
        raise
    return n_points_xy, n_points_z


def loadMeshSizeRange(tomlpath="description.toml", silent=False):
    """ Load data used for meshsizetest """
    if not silent:
        print("Loading description", tomlpath)
    desc_obj = toml.load(tomlpath)

    try:
        n_points_xy_min = float(desc_obj['meshsizetest']['n_points_xy_min'])
        n_points_xy_max = float(desc_obj['meshsizetest']['n_points_xy_max'])
    except KeyError:
        print("toml file '{}' couldn't be read successfully.".format(tomlpath))
        raise
    return n_points_xy_min, n_points_xy_max


def loadAdheasionSites(tomlpath="description.toml", silent=False):
    """ Load adhesion site description from toml file """
    return _h_loadAdheasionSites(tomlpath, 'adheasion', silent, optional=False)


def _h_loadAdheasionSites(tomlpath, descrname, silent, optional):
    if not silent:
        print("Loading description", tomlpath)
    desc_obj = toml.load(tomlpath)

    pointList = []

    try:
        adheasions = desc_obj[descrname]
    except KeyError:
        if optional:
            return []
        else:
            print("toml file couldn't be read successfully.")
            raise

    for elm in adheasions:
        adheasion = adheasions[elm]
        try:
            atype = adheasion['type']
        except KeyError:
            print("Please specify type of adheasion '{}'".format(elm))
            raise

        unit = adheasion.get('F_unit')

        if atype == "indentor":
            try:
                a = float(adheasion['a'])
                fz = getFource(adheasion['F'], unit)
                pos = adheasion['pos']
                assert (type(pos))
            except KeyError:
                print("Incorrect data for indentor adheasions")
                raise
            x, y = pos
            pointList.append((x, y, a, 0., 0., fz))
        elif atype == "dipole":
            try:
                a = float(adheasion['a'])
                pos = adheasion['pos']
                if 'phi' in adheasion:
                    F = getFource(adheasion['F'], unit)
                    d = float(adheasion['d'])
                    phi = float(adheasion['phi'])
                    if 'FM' in adheasion:
                        fm = getFource(adheasion['FM'], unit)
                    else:
                        fm = 0.0

                    dx = d * math.cos(math.radians(phi))
                    dy = d * math.sin(math.radians(phi))
                    fx = F * math.cos(math.radians(phi)) + fm * math.sin(math.radians(phi))
                    fy = F * math.sin(math.radians(phi)) + fm * math.cos(math.radians(phi))
                else:
                    dx = float(adheasion['dx'])
                    dy = float(adheasion['dx'])
                    dr = math.sqrt(dx * dx + dy * dy)
                    if 'F' in adheasion and (dr != 0.):
                        F = getFource(adheasion['F'], unit)
                        fx = F * dx / dr
                        fy = F * dy / dr
                    else:
                        fx = getFource(adheasion['Fx'], unit)
                        fy = getFource(adheasion['Fy'], unit)
                if 'Fz' in adheasion:
                    Fz = getFource(adheasion['Fz'], unit)
                else:
                    Fz = 0.0
            except KeyError:
                print("Incorrect data for dipole adheasions")
                raise
            x, y = pos
            pointList.append((x + dx / 2, y + dy / 2, a, -fx, -fy, Fz/2))
            pointList.append((x - dx / 2, y - dy / 2, a, fx, fy, Fz/2))
        elif atype == "point":
            try:
                a = float(adheasion['a'])
                if 'phi' in adheasion:
                    phi = float(adheasion['phi'])
                    F = getFource(adheasion['F'], unit)
                    Qx = F * math.cos(math.radians(phi))
                    Qy = F * math.sin(math.radians(phi))
                    Qz = 0.0
                else:
                    F = adheasion['F']
                    Qx, Qy, Qz = F
                    Qx = getFource(Qx, unit)
                    Qy = getFource(Qy, unit)
                    Qz = getFource(Qz, unit)
                #
                pos = adheasion['pos']
                assert (type(pos))
            except KeyError:
                print("Incorrect data for point adheasions")
                raise
            x, y = pos
            pointList.append((x, y, a, Qx, Qy, Qz))
        else:
            raise RuntimeError("Incorrect adheasion type {}".format(atype))
    return pointList


if __name__ == "__main__":
    fname, E, nu, spacing_xy, spacing_z = loadDataDescription("utils/testdescriptor.toml")
    print("fname={}, E = {}, nu = {}, sxy = {}, sz = {}".format(
        fname, E, nu, spacing_xy, spacing_z))

    xyPix, nLayers, scale = loadSimulationData("utils/testdescriptor.toml")
    print("nLayers = {}, xyPix = {}".format(nLayers, xyPix))

    pointList = loadAdheasionSites("utils/testdescriptor.toml")
    print("Adheasion Sites", pointList)
