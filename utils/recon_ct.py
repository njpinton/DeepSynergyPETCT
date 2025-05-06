import re
import astra
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras

import utils.data_utils
from lbfgs_vae import get_ref_imgs, get_objf, get_z, get_psnr


class ParamProj:
    """
    Class to encapsulate projection geometry parameters.
    Used for setting up the ASTRA geometry configuration.
    """
    def __init__(self,
                 n_angles=120,
                 vox_size=0.9765625,
                 beam_type='fanflat',
                 det_width=1.2,
                 det_count=750,
                 source_origin=600.,
                 origin_det=600.,
                 proj_type='line_fanflat'):
        self.n_angles = n_angles
        self.vox_size = vox_size
        self.beam_type = beam_type
        self.det_width = det_width
        self.det_count = det_count
        self.source_origin = source_origin
        self.origin_det = origin_det
        self.proj_type = proj_type

    def get_paramProj(self):
        return self


class ParamRecon:
    """
    Class for reconstruction parameters including prior type and iteration controls.
    """
    def __init__(self, I, nb_iter, bckg, prior, beta, delta):
        self.I = I
        self.nb_iter = nb_iter
        self.bckg = bckg
        self.prior = prior
        self.beta = beta
        self.delta = delta

    def get_paramRecon(self):
        return self


class ParamReg:
    """
    Class for regularization parameter groupings (currently unused).
    """
    def __init__(self, alpha, gamma, paramRecon):
        self.alpha = alpha
        self.gamma = gamma
        self.paramRecon = paramRecon

    def get_paramReg(self):
        return self


def load_model(saved_model_dir, model_name):
    """Load a trained VAE decoder model from disk."""
    saved_model_generator_path = f'{saved_model_dir}/{model_name}/decoder'
    return keras.models.load_model(saved_model_generator_path, compile=False)


def forward_projection_astra(img, proj_id):
    """Wrapper for ASTRA forward projection."""
    sino_id, sino = astra.creators.create_sino(img, proj_id)
    astra.data2d.delete(sino_id)
    return sino


def back_projection_astra(sino, proj_id):
    """Wrapper for ASTRA backprojection."""
    bp_id, img = astra.create_backprojection(sino, proj_id)
    astra.data2d.delete(bp_id)
    return img


def get_proj_id(img_size, param_proj):
    """
    Create an ASTRA projection geometry object.
    Arguments:
        img_size: Number of voxels per side (assumes square image).
        param_proj: Instance of ParamProj defining the geometry.
    Returns:
        proj_id: ASTRA projection geometry ID.
    """
    vol_geom = astra.creators.create_vol_geom(img_size)
    proj_geom = astra.create_proj_geom(param_proj.beam_type,
                                       param_proj.det_width / param_proj.vox_size,
                                       param_proj.det_count,
                                       np.linspace(0, 2 * np.pi, param_proj.n_angles, endpoint=False),
                                       param_proj.source_origin / param_proj.vox_size,
                                       param_proj.origin_det / param_proj.vox_size)
    return astra.create_projector(param_proj.proj_type, proj_geom, vol_geom)


def attn_to_hu(mu, mu_water, mu_air, pixel_spacing=0.9765625):
    """Convert attenuation coefficients to Hounsfield Units (HU)."""
    return 1000 * (mu * 10 / pixel_spacing - mu_water) / (mu_water - mu_air)


def hu_to_attn(hu, mu_water, mu_air, pixel_spacing=0.9765625):
    """Convert Hounsfield Units (HU) to attenuation coefficients."""
    return ((hu/1000 * (mu_water - mu_air)) + mu_water) * pixel_spacing / 10


def norm_to_hu(norm, min_, max_):
    """Scale normalized image values to HU."""
    return norm * (max_ - min_) + min_


def recon_ct_pwls_z(x_init, y, beta, proj_id, param_recon, model, pet_image, ct_image, mu_air, mu_water, delay_iter=50, with_fz_bool=True):
    """
    PWLS reconstruction with deep prior from pretrained decoder model.
    Arguments:
        x_init: Initial image estimate.
        y: Noisy sinogram.
        beta: Regularization parameter (prior weight).
        proj_id: ASTRA projector ID.
        param_recon: Reconstruction parameters.
        model: Decoder network (e.g., β-VAE decoder).
        pet_image, ct_image: Ground truth images for deep prior loss.
        mu_air, mu_water: Linear attenuation coefficients.
        delay_iter: Number of iterations to delay inclusion of deep prior.
        with_fz_bool: Flag for enabling deep prior term.
    Returns:
        x: Final reconstructed image.
        fz: Final generated image from latent vector z.
    """
    diff = y - param_recon.bckg
    l = np.where(diff > 0, np.log(param_recon.I / diff), 0)
    w = np.where(diff > 0, np.square(diff) / y, 0)

    N = x_init.shape[0]
    img_one = np.ones([N, N])
    D = back_projection_astra(w * forward_projection_astra(img_one, proj_id), proj_id)

    low, high = -2, 2
    max_iter = 100
    min_, max_ = -1024, 3072
    fz = 0
    w0 = 0.75
    x = x_init
    z = np.random.uniform(low, high, model.input_shape[1]).flatten(order='F')
    ct_image = hu_to_attn(ct_image, mu_water, mu_air)

    if with_fz_bool:
        plt.ion()
        fig1 = plt.figure('frame')

    for i in tqdm(range(param_recon.nb_iter)):
        grad = back_projection_astra(w * (forward_projection_astra(x, proj_id) - l), proj_id)
        x_rec = np.where(D == 0, 0, x - grad / D)

        # Normalize for input to decoder
        x_rec_norm = utils.data_utils.normalise_zero_one(1000 * (x_rec - mu_water) / (mu_water - mu_air), -1024, 3072)

        if i < delay_iter:
            with_fz = False
        else:
            with_fz = with_fz_bool

        if with_fz:
            objf = lambda v: get_objf(v,
                                      model,
                                      x_init.shape[0],
                                      mode_='petct',
                                      w0=w0,
                                      pet_ref_img=utils.data_utils.normalise_zero_one(pet_image, 0, 1e5),
                                      ct_ref_img=x_rec_norm)
            z, _, _ = get_z(objf, z, max_iter, bounds=None)
            fz = model(np.asarray([z]))[1].numpy().reshape(x_rec_norm.shape)
            fz = norm_to_hu(fz, min_, max_)
            fz = hu_to_attn(fz, mu_water, mu_air)

        # PWLS update with regularization
        x = (D * x_rec + beta * fz) / (D + beta)
        x = np.where(x > 0, x, 0)

        if with_fz:
            psnr_x = get_psnr(ct_image, x, True)
            psnr_gen = get_psnr(ct_image, fz, True)
            plt.title(f'PSNR: recon = {psnr_x:.2f}, gen = {psnr_gen:.2f}')

            fig1.clf()
            ax1 = fig1.add_subplot(1, 3, 1); ax1.axis('off'); ax1.imshow(x, cmap='gray')
            ax2 = fig1.add_subplot(1, 3, 2); ax2.axis('off'); ax2.imshow(fz, cmap='gray')
            ax3 = fig1.add_subplot(1, 3, 3); ax3.axis('off'); ax3.imshow(ct_image, cmap='gray')
            plt.pause(0.05)

    return x, fz


def main():
    """
    End-to-end pipeline for PET/CT-informed PWLS reconstruction with β-VAE deep prior.
    """
    saved_model_dir = 'saved_models/betavae'
    model_name = 'beta_vae_petct_lungs256x256_p14476_t500_lr0.0001_ld128_id256_batch256_f32_l6_e1000_mconcatenate_20220802-152713'
    model = load_model(saved_model_dir, model_name)
    img_size = int(re.search(r"_lungs(\d+)", model_name)[1])

    normalize = False
    mode_ = 'petct'
    src_path = f'/home/noel/data/lung-pet-ct-np2/pet_ct_images{img_size}x{img_size}x128_p117.npy'
    pet_image, ct_image = get_ref_imgs(src_path, img_size, mode_, 170, normalize)

    param_proj = ParamProj(n_angles=60,
                           vox_size=0.9765625,
                           beam_type='fanflat',
                           det_width=1.2,
                           det_count=750,
                           source_origin=600,
                           origin_det=600,
                           proj_type='line_fanflat').get_paramProj()

    proj_id = get_proj_id(img_size, param_proj)

    param_recon = ParamRecon(I=1000,
                             nb_iter=100,
                             bckg=0,
                             prior='Huber',
                             beta=1000,
                             delta=0.001).get_paramRecon()

    mu_water = 0.02597       # cm⁻¹ at 80 keV
    mu_air = 2.407E-02 * 1.205E-03

    ct_image_at = hu_to_attn(ct_image, mu_water, mu_air)
    ct_image_at = np.where(ct_image_at < 0, 0, ct_image_at)

    # Simulate noisy sinogram
    y = forward_projection_astra(ct_image_at, proj_id)
    y = param_recon.I * np.exp(-y)
    y = np.random.poisson(y)

    x_init = np.ones(ct_image_at.shape)
    beta = 50000

    # Reconstruction with deep prior
    x, fz = recon_ct_pwls_z(x_init, y, beta, proj_id, param_recon, model, pet_image, ct_image, mu_air, mu_water, delay_iter=75, with_fz_bool=True)
    x_2, _ = recon_ct_pwls_z(x_init, y, beta, proj_id, param_recon, model, pet_image, ct_image, mu_air, mu_water, with_fz_bool=False)

    plt.imshow(ct_image_at - x)


if __name__ == '__main__':
    main()
