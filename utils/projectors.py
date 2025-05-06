#!/usr/bin/env python
import json
import imutils

from tqdm import tqdm
import numpy as np
from scipy.signal import convolve2d
from skimage.transform import radon, iradon


def create_scatter(matr, bkg_event_ratio):
    return (1/(1/bkg_event_ratio - 1))*np.ones(matr.shape)*sum(sum(matr))/sum(sum(np.ones(matr.shape)))


def add_scatter_noise(sino, bkg_event_ratio):
    r = create_scatter(sino, bkg_event_ratio)
    sino = sino + r
    sino = np.random.poisson(sino).astype('float')
    return sino


class Params:
    def __init__(self,
                 phi=np.arange(0, 2*np.pi, 0.05),
                 n_angles=120,
                 vox_size=1,
                 fwhm=5,
                 gpu=0,
                 scale=0.1):
        self.phi = phi
        self.n_angles = n_angles
        self.vox_size = vox_size
        self.fwhm = fwhm
        self.gpu = gpu
        self.scale = scale

    def get_params(self):
        return self


class dotstruct():
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getitem__(self, name):
        return self[name]

    def as_dict(self):
        dic = {}
        for item in self.__dict__.keys():
            dic[item] = self.__dict__.get(item)
        return dic


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def forward_projection(img, params):
    n_psf = round(img.shape[0]/5)
    sigma_mm = params.fwhm/2.3555
    sigma_vox = sigma_mm/params.vox_size
    n_angles = params.n_angles

    h = matlab_style_gauss2D((n_psf, n_psf), sigma=sigma_vox)

    theta = np.linspace(0., 360., n_angles, endpoint=False)
    img = conv2(img, h)

    # img = radon(img, theta) * params.scale
    img = radon2(img, theta) * params.scale
    return img


def radon2(im, theta):
    N = im.shape[0]
    n_theta = len(theta)

    sino = np.zeros([N, n_theta])

    for i in range(n_theta):
        im_rot = imutils.rotate(im, theta[i])
        sino[:, i] = np.transpose(np.sum(im_rot, 1))
        # sino[:, i] = np.sum(im_rot, 1)

    return sino


def bradon2(sino, theta):
    N = sino.shape[0]
    n_theta = len(theta)

    im = np.zeros([N, N])

    for i in range(n_theta):
        r_sino = np.tile(np.transpose(sino[:, i]), [N, 1])
        im = im + imutils.rotate(r_sino, -theta[i])

    im = imutils.rotate(im, 270)

    return im


def back_projection(sinogram, params):
    n_psf = round(sinogram.shape[0]/5)
    sigma_mm = params.fwhm/2.3555
    sigma_vox = sigma_mm/params.vox_size
    n_angles = params.n_angles

    h = matlab_style_gauss2D((n_psf, n_psf), sigma=sigma_vox)
    theta = np.linspace(0., 360., n_angles, endpoint=False)

    # img = iradon(sinogram, theta=theta, filter_name=None)
    img = bradon2(sinogram, theta)
    return conv2(img, h) * params.scale


def mlem(sinogram, img, scatter, params, n_iter=60):
    norm = back_projection(np.ones(sinogram.shape), params)

    for k in range(n_iter):
        y_bar = forward_projection(img, params) + scatter
        ratio = np.nan_to_num(sinogram/y_bar)
        update = np.nan_to_num(back_projection(ratio, params)/norm)
        img = img * update
    return img


def create_train_set(imgs, params, bkg_event_ratio=0.2, step=20, n_iter=60, save=False, save_path='train.json'):
    imgs_recon = dict()
    tmp_dict = dict()
    imgs_recon['params'] = params.__dict__
    # imgs_recon['params']['phi'] = params.phi.tolist()
    imgs_recon['params']['bkg_event_ratio'] = bkg_event_ratio
    imgs_recon['params']['step'] = step
    imgs_recon['params']['n_iter'] = n_iter
    for i, img_slice in enumerate(tqdm(range(10, imgs.shape[0]-10, step))):
        img_raw = get_phantom_slice(imgs, img_slice)
        tmp_dict['raw'] = img_raw.tolist()

        sinogram = forward_projection(img_raw, params)
        sinogram = add_scatter_noise(sinogram, bkg_event_ratio)

        x_init = np.ones(img_raw.shape)

        r = create_scatter(sinogram, bkg_event_ratio)
        x_recon = mlem(sinogram, x_init, r, params, n_iter)
        tmp_dict['recon'] = x_recon.tolist()
        imgs_recon[i] = tmp_dict

    if save:
        with open(save_path, 'w') as file:
            json.dump(imgs_recon, file)

    return imgs_recon


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from img_utils import get_images, get_phantom_slice, display_image

    mri_file_path = 'data/t1_icbm_normal_1mm_pn3_rf20.mnc'
    anat_file_path = 'data/phantom_1.0mm_normal_crisp.mnc'

    mri = get_images(mri_file_path, with_padding=True)
    anat = get_images(anat_file_path, with_padding=True)

    bkg_event_ratio = 0.2
    img_slice = 90
    pet_phantom_2d = get_phantom_slice(anat, img_slice)

    params = Params().get_params()

    sinogram = forward_projection(pet_phantom_2d, params)

    r = create_scatter(sinogram, bkg_event_ratio)

    sinogram_nonoise = sinogram + r
    sinogram_noisy = np.random.poisson(sinogram_nonoise)

    x_init = np.ones(pet_phantom_2d.shape)

    x_recon = mlem(sinogram_noisy, x_init, r, params, n_iter=200)
    plt.figure()
    plt.imshow(x_recon, cmap='gray')

    recon = create_train_set(anat, params, step=100, save=True, n_iter=20)

    plt.imshow(recon[0]['recon'])
