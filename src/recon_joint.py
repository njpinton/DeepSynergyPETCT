import os
import numpy as np
from tensorflow import keras
import argparse
import json
from utils.projectors import forward_projection, back_projection, Params
from utils.lbfgs_vae import get_ref_imgs, get_objf, get_z, get_psnr
from utils.recon_ct import (forward_projection_astra, back_projection_astra, attn_to_hu, hu_to_attn, norm_to_hu,
                            ParamRecon, ParamProj, get_proj_id)
from utils.attenuationCT_to_511 import hu_to_511


def load_model(saved_model_dir, model_name):
    """
    Load the trained model for reconstruction.

    Args:
        saved_model_dir (str): Directory where the model is saved.
        model_name (str): Name of the model to load.

    Returns:
        keras.Model: The loaded generator model.
    """
    model_path = os.path.join(saved_model_dir, model_name, 'decoder')
    return keras.models.load_model(model_path, compile=False)


def recon_joint(x_init, sinogram_pet, sinogram_ct, pet_image, ct_image, model, w0, beta, total_iter, params, param_recon, proj_id,
                mu_air, mu_water, max_iter=20, nogen=False):
    """
    Perform joint reconstruction of PET and CT images using the provided model.

    Args:
        x_init (np.array): Initial reconstruction guess.
        sinogram_pet (np.array): PET sinogram data.
        sinogram_ct (np.array): CT sinogram data.
        pet_image (np.array): PET reference image.
        ct_image (np.array): CT reference image.
        model (keras.Model): The trained model used for reconstruction.
        w0 (float): Weight for the PET term in the loss function.
        beta (float): Regularization parameter for CT reconstruction.
        total_iter (int): Total number of iterations.
        params (dict): Parameters for forward/backward projection.
        param_recon (dict): Reconstruction parameters.
        proj_id (np.array): Projection ID for ASTRA toolbox.
        mu_air (float): Linear attenuation coefficient for air.
        mu_water (float): Linear attenuation coefficient for water.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.
        nogen (bool, optional): If True, skips PET reconstruction. Defaults to False.

    Returns:
        tuple: Reconstructed PET image, CT image, and generated PET/CT images.
    """
    norm = back_projection(np.ones(sinogram_pet.shape), params)
    z = np.random.uniform(-3, 3, model.input_shape[1]).flatten(order='F')

    # Initialize CT reconstruction
    diff = sinogram_ct - param_recon['bckg']
    l = np.where(diff > 0, np.log(param_recon['I'] / diff), 0)
    w = np.where(diff > 0, np.square(diff) / sinogram_ct, 0)

    N = x_init.shape[0]
    img_one = np.ones([N, N])
    D1 = back_projection_astra(w * forward_projection_astra(img_one, proj_id), proj_id)

    # Initialize images
    x_pet, x_ct = x_init, x_init

    subiter_init = 100
    for _ in range(subiter_init):
        grad = back_projection_astra(w * (forward_projection_astra(x_ct, proj_id) - l), proj_id)
        x_ct = np.where(D1 == 0, 0, x_ct - grad / D1)
        x_ct = np.where(x_ct > 0, x_ct, 0)

    # Compute linear attenuation for attenuation correction
    mu_511 = hu_to_511(attn_to_hu(x_ct, mu_water, mu_air)) * 0.9765625 / 10
    pmu_511 = forward_projection(mu_511, params)
    lac = np.exp(-pmu_511)
    norm_ac = back_projection(lac, params)

    if nogen:
        norm = norm_ac

    for _ in range(subiter_init):
        fproj = forward_projection(x_pet, params)
        ratio = np.nan_to_num(sinogram_pet / fproj)
        update = np.nan_to_num(back_projection(ratio, params) / norm)
        x_pet = x_pet * update

    if nogen:
        return x_pet, x_ct, None, None

    scale_ct = 96.93775373893472
    for j in range(total_iter):
        # Update z
        x_ct_norm = x_ct * scale_ct
        x_ct_norm = np.where(x_ct_norm > 0, x_ct_norm, 0)
        objf = lambda v: get_objf(v, model, x_init.shape[0], mode_='petct', w0=w0, pet_ref_img=x_pet, ct_ref_img=x_ct_norm)
        z, f, d = get_z(objf, z, max_iter)

        # Generate images
        gen_img_pet = model(np.asarray([z]))[0].numpy().reshape(x_init.shape)
        gen_img_ct = model(np.asarray([z]))[1].numpy().reshape(x_init.shape)

        gen_img_ct = norm_to_hu(gen_img_ct, -1024, 3072)
        gen_img_ct = hu_to_attn(gen_img_ct, mu_water, mu_air)
        gen_img_ct = np.where(gen_img_ct > 0, gen_img_ct, 0)

        # PET reconstruction
        for s1 in range(3):
            fproj = forward_projection(x_pet, params)
            ratio = np.nan_to_num(sinogram_pet / fproj)
            update = np.nan_to_num(back_projection(ratio, params) / norm_ac)
            x_pet = x_pet * update

            # Update gen PET image
            if w0 != 0 and beta != 0:
                x_pet = ((beta * w0 * gen_img_pet - norm_ac) +
                         np.sqrt((beta * w0 * gen_img_pet - norm_ac) ** 2
                                 + 4 * beta * w0 * norm_ac * x_pet)) / (beta * w0 * 2)
            x_pet = np.where(x_pet > 0, x_pet, 0)

        # CT reconstruction
        for s2 in range(5):
            grad = back_projection_astra(w * (forward_projection_astra(x_ct, proj_id) - l), proj_id)
            x_ct = np.where(D1 == 0, 0, x_ct - grad / D1)
            x_ct = np.where(x_ct > 0, x_ct, 0)

            # Update gen CT image
            x_ct = (D1 * x_ct + beta * (1 - w0) * gen_img_ct * scale_ct**2) / (D1 + (1 - w0) * beta * scale_ct**2)
            x_ct = np.where(x_ct > 0, x_ct, 0)

    return x_pet, x_ct, gen_img_pet, gen_img_ct


def parse_args():
    """
    Parse command line arguments from a configuration file.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PET/CT Reconstruction using Generative Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file (JSON)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    return config


def main():
    # Parse arguments from config file
    config = parse_args()

    # Load model
    model = load_model(config['saved_model_dir'], config['model_name'])

    # Load reference images
    ref_imgs = get_ref_imgs(config['src_path'], config['img_size'], config['mode'], config['image_slice_no'], dim=2, normalize=config['normalize'])
    pet_image, ct_image = ref_imgs[0], ref_imgs[1]

    # Set projection parameters
    params = Params(n_angles=config['n_angles'],
                    fwhm=config['fwhm'],
                    scale=config['scale'],
                    gpu=config['gpu'],
                    vox_size=config['vox_size']).get_params()

    param_proj = ParamProj(config['n_angles'],
                           config['vox_size'],
                           config['beam_type'],
                           config['det_width'],
                           config['det_count'],
                           config['source_origin'],
                           config['origin_det'],
                           config['proj_type']).get_paramProj()

    proj_id = get_proj_id(config['img_size'], param_proj)

    # Set reconstruction parameters
    param_recon = ParamRecon(config['i_all'],
                             config['nb_iter'],
                             config['bckg'],
                             config['prior'],
                             config['beta_recon'],
                             config['delta']).get_paramRecon()

    # Initialize sinograms
    sinogram_pet = forward_projection(pet_image, params=params)
    sinogram_pet = np.random.poisson(sinogram_pet)

    sinogram_ct = forward_projection_astra(ct_image, proj_id)
    sinogram_ct = param_recon['I'] * np.exp(-sinogram_ct)
    sinogram_ct = np.random.poisson(sinogram_ct)

    # Initialize reconstruction parameters
    x_init = np.ones(pet_image.shape)

    # Perform reconstruction
    x_pet, x_ct, gen_pet, gen_ct = recon_joint(x_init, sinogram_pet, sinogram_ct, pet_image, ct_image, model,
                                               config['w0'], config['beta'], config['total_iter'], params, param_recon,
                                               proj_id, config['mu_air'], config['mu_water'], config['max_iter'])

    # PSNR Calculation
    psnr_dict_pet = get_psnr(pet_image, x_pet, True)
    psnr_dict_ct = get_psnr(ct_image, x_ct, False)

    print(f"PET PSNR: {psnr_dict_pet:.2f}, CT PSNR: {psnr_dict_ct:.2f}")


if __name__ == "__main__":
    main()
