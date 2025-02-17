import numpy as np
from scipy.constants import speed_of_light

def uma_los(d3d, d2d, dbp, fc, h_b, h_t):
    """
    Compute the path loss in LOS, Urban, Marco setting.
    :param d3d: 3D distance
    :param d2d: 2D distance
    :param dbp: Breakpoint distance
    :param fc: frequency in GHz
    :param h_b: height of basestation
    :param h_t: height of UT
    :return:
    """
    # 38.901 UMa LOS
    PL1 = 28 + 22 * np.log10(d3d) + 20 * np.log10(fc)
    PL2 = 28 + 40 * np.log10(d3d) + 20 * np.log10(fc) - 9 * np.log10(dbp ** 2 + (h_b - h_t) ** 2)
    # PL = np.zeros((d3d.shape))
    PL = PL2  # Default pathloss
    PL[(np.greater_equal(d2d, 10) & np.less_equal(d2d, dbp))] = PL1[(np.greater_equal(d2d, 10) & np.less_equal(d2d,
                                                                                                               dbp))]  # Overwrite if distance is greater than 10 meters or smaller than dbp
    return PL

def uma_nlos(d3d, d2d, dbp, fc, h_b, h_t):
    """
    Compute the path loss in NLOS, Urban, Marco setting.
    :param d3d: 3D distance
    :param d2d: 2D distance
    :param dbp: Breakpoint distance
    :param fc: frequency in GHz
    :param h_b: height of basestation
    :param h_t: height of UT
    :return:
    """
    # 38901 UMa NLOS
    PL_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc) - 0.6 * (h_t - 1.5)
    PL = np.zeros((d3d.shape))
    PL = np.maximum(uma_los(d3d, d2d, dbp, fc, h_b, h_t), PL_nlos)
    return PL

def pathloss_38901(distance, frequency, h_bs=30, h_ut=1.5):
    """
    Simple path loss model for computing RSRP based on distance.
    :param distance: distance between basestation and UE
    :param frequency: frequency in GHz
    :param h_bs: height of basestation
    :param h_ut: height of UT
    :return:
    """
    # Constants
    fc = frequency
    h_b = h_bs  # 30 meters
    h_t = h_ut  # 1.5

    # 2D distance
    d2d = distance

    # 3D distance
    h_e = h_b - h_t  # effective height
    d3d = np.sqrt(d2d ** 2 + h_e ** 2)

    # Breakpoint distance
    dbp = 4 * h_b * h_t * fc * 10e8 / speed_of_light

    loss = uma_nlos(d3d, d2d, dbp, fc, h_b, h_t)
    return loss, uma_los(d3d, d2d, dbp, fc, h_b, h_t)

def pathloss_friis_free_space_model(d, h_b, h_r, f):
    """
    Compute the pathloss in dB, Friis model.
    :param d: is the link distance in km and 
    :param f: is the transmission frequency in MHz
    :return: pathloss in dB
    """
    return 32.45 + 20 * np.log10(d) + 20 * np.log10(f)

def pathloss_ericsson_model(d, h_b, h_r, f):
    """
    Compute the pathloss in dB, Ericsson model.
    :param h_b: is the height of the base station in meters,
    :param h_r: is the height of the receiver in meters,
    :param d: is the link distance in km,
    :param f: is the transmission frequency in MHz, 
    :return: pathloss in dB
    """

    def g(freq):
        return 44.49 * np.log10(freq) - 4.78 * (np.log10(freq) ** 2)

    # for urban environments, (a0, a1, a2, a3) = (36.2, 30.2, 12, 0.1)
    a0 = 36.2
    a1 = 30.2
    a2 = 12
    a3 = 0.1
    line1 = a0 + a1 * np.log10(d) + a2 * np.log10(h_b)
    line2 = + a3 * np.log10(h_b) * np.log10(d)
    line3 = - 3.2 * ((np.log10(11.75 * h_r)) ** 2) + g(f)
    return line1 + line2 + line3