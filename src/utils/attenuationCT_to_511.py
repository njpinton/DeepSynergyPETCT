import numpy as np
import pydicom
from scipy.interpolate import interp1d


def hu_to_511(image, kvp=120):
    """
    https://github.com/villekf/OMEGA/blob/master/source/attenuationCT_to_511.m

    :param image: hu values of either 2d or 3d ct images
    :param kvp: kvp value of ct image (0018, 0060)
    :return: attenuation factor mask for pet
    """

    if image.shape == 3:
        nx, ny, nz = image.shape
    else:
        nx, ny = image.shape

    if kvp == 80:
        a = np.dot([9.3, 3.28, 0.41], 10**-5)
        b = np.array([0.093, 0.093, 0.122])

    elif kvp == 100:
        a = np.dot([9.3, 4, 0.5], 10**-5)
        b = np.array([0.093, 0.093, 0.128])

    elif kvp == 120:
        a = np.dot([9.3, 4.71, 0.589], 10**-5)
        b = np.array([0.093, 0.093, 0.134])

    elif kvp == 140:
        a = np.dot([9.3, 5.59, 0.698], 10**-5)
        b = np.array([0.093, 0.093, 0.142])

    else:
        a1 = np.dot([9.3, 3.28, 0.41], 10**-5)
        b1 = np.array([0.093, 0.093, 0.122])
        a2 = np.dot([9.3, 4, 0.5], 10**-5)
        b2 = np.array([0.093, 0.093, 0.128])
        a3 = np.dot([9.3, 4.71, 0.589], 10**-5)
        b3 = np.array([0.093, 0.093, 0.134])
        a4 = np.dot([9.3, 5.59, 0.698], 10**-5)
        b4 = np.array([0.093, 0.093, 0.142])

        aa = np.array([a1, a2, a3, a4])
        bb = np.array([b1, b2, b3, b4])
        c = np.array([80, 100, 120, 140])

        a = np.zeros(3)
        b = np.zeros(3)

        for kk in range(3):
            a[kk] = interp1d(c, aa[:, kk], kind='slinear')(kvp)
            b[kk] = interp1d(c, bb[:, kk], kind='slinear')(kvp)

    x = np.zeros([4, 2])

    x[0, :] = np.array([-1000.0, b[0] - 1000 * a[0]])
    x[1, :] = np.array([0, b[1]])
    x[2, :] = np.array([1000, b[1] + a[1] * 1000])
    x[3, :] = np.array([3000, b[1] + a[1] * 1000 + a[2] * 2000])

    tarkkuus = 0.1
    vali = np.arange(-1000, 3000, tarkkuus)
    inter = interp1d(x[:, 0], x[:, 1])(vali)

    attenuation_factors = interp1d(vali, inter, fill_value='extrapolate')(np.ndarray.flatten(image))
    attenuation_factors = attenuation_factors.reshape([nx, ny])
    # add attenuation for 3D

    return attenuation_factors


def temp():
    path = '/home/noel/data/dicoms/data/abado lionel/ct/DICOM/EXP00000/EXP0000'
    data = pydicom.dcmread(path)
    print(data.data_element('KVP'))  # 120.0
