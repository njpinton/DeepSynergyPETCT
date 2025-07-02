import re

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from skimage.metrics import structural_similarity as ssim
from utils import data_utils as utils


def load_model(saved_model_dir, model_name, t='decoder'):
    # Load generator model
    saved_model_generator_path = f'{saved_model_dir}/{model_name}/{t}'
    return tf.keras.models.load_model(saved_model_generator_path, compile=False)


def get_psnr(original, compressed, max_true=False):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = 1.0
    if max_true:
        max_pixel = original.max()
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def get_ref_imgs(src_path, img_size=256, mode_='petct', image_i=250, dim=2, normalize=True):
    pet_ct_images = np.load(src_path)
    reference_images = None

    if mode_ == 'ct':
        # get ct images
        ct_images = np.expand_dims(pet_ct_images[:, :, :, 1], axis=3)
        reference_images = utils.normalise_zero_one(ct_images, min_=-1024, max_=3072)
        ref_img = reference_images[image_i].reshape(img_size, img_size)
        return [ref_img]

    elif mode_ == 'pet':
        # get pet images
        pet_images = np.expand_dims(pet_ct_images[:, :, :, 0], axis=3)
        pet_images = np.clip(pet_images, 0, 1e5)
        reference_images = utils.normalise_zero_one(pet_images)
        ref_img = reference_images[image_i].reshape(img_size, img_size)
        return [ref_img]

    else:
        ref_img_pet = np.expand_dims(pet_ct_images[:, :, :, 0], axis=3)
        ref_img_ct = np.expand_dims(pet_ct_images[:, :, :, 1], axis=3)
        ref_img_pet = ref_img_pet[image_i].reshape(img_size, img_size)
        ref_img_ct = ref_img_ct[image_i].reshape(img_size, img_size)

        if normalize:
            ref_img_pet = utils.normalise_zero_one(ref_img_pet, min_=0, max_=1e5)
            ref_img_ct = utils.normalise_zero_one(ref_img_ct, min_=-1024, max_=3072)
        return [ref_img_pet, ref_img_ct]


def get_z(objf, x_init, max_iter):
    """Returns x, f , d"""
    return fmin_l_bfgs_b(objf, x_init, fprime=None, factr=1e12, pgtol=1e-5, iprint=-1, maxiter=max_iter)


def get_objf(z_sample, model, img_size=256, mode_='petct', w0=0.0, pet_ref_img=None, ct_ref_img=None):
    global old_dydx, old_diff_total
    gen_imgs = model(np.asarray([z_sample]))

    if mode_ == 'pet':
        pet_gen_img = gen_imgs.numpy().reshape(img_size, img_size)
        diff_total = np.sum(np.abs(pet_ref_img - pet_gen_img))
    elif mode_ == 'ct':
        ct_gen_img = gen_imgs.numpy().reshape(img_size, img_size)
        diff_total = np.sum(np.abs(ct_ref_img - ct_gen_img))
    else:
        pet_gen_img = gen_imgs[0].numpy().reshape(pet_ref_img.shape)
        ct_gen_img = gen_imgs[1].numpy().reshape(ct_ref_img.shape)
        diff_pet = np.square(pet_ref_img - pet_gen_img) * w0
        diff_ct = np.square(ct_ref_img - ct_gen_img) * (1-w0)
        diff_total = np.sum(diff_ct + diff_pet)

    x_tensor = tf.expand_dims(tf.convert_to_tensor(z_sample), 0)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        gen_imgs = model(x_tensor)
        if mode_ == 'petct':
            pet_gen_img = gen_imgs[0]
            ct_gen_img = gen_imgs[1]
            _y = w0 * tf.square(np.expand_dims(pet_ref_img, axis=(0, -1)) - pet_gen_img) + \
                (1 - w0) * tf.square(np.expand_dims(ct_ref_img, axis=(0, -1)) - ct_gen_img)
        elif mode_ == 'ct':
            ct_gen_img = gen_imgs
            _y = tf.abs(np.expand_dims(ct_ref_img, axis=(0, -1)) - ct_gen_img)

        y = tf.reduce_sum(_y)

    dy_dx = tape.gradient(y, x_tensor).numpy().flatten(order='F')
    diff_total = diff_total.flatten(order='F')

    if np.all(dy_dx == 0) or np.any(dy_dx == np.nan):
        dy_dx = old_dydx
        diff_total = old_diff_total

    old_dydx = dy_dx
    old_diff_total = diff_total

    return diff_total, dy_dx


def main():
    # Load model
    saved_model_dir = 'saved_models/betavae'
    model_name = 'image128/beta_vae_petct_lungs128x128_p46925_t1280_lr0.0001_ld64_id128_batch2048_f32_l4_e500_mconcatenate_arelu_20231018-150619'
    model = load_model(saved_model_dir, model_name)

    mode_ = re.search(r"vae_(\w+)_lungs", model_name)[1]
    latent_dim = int(re.search(r"_ld(\d+)", model_name)[1])
    img_size = int(re.search(r"_lungs(\d+)", model_name)[1])

    # Load reference images
    file_path = '/home/noel/data/dicoms-nobed-np-2/pet_ct_48205x128x128x2.npy'
    pet_ct_images = np.load(file_path)
    ct_images = pet_ct_images[:, :, :, 1]
    scale_ct = 1 / ct_images.max()

    ref_img_pet = pet_ct_images[0, :, :, 0]
    ref_img_pet = utils.normalise_zero_one(ref_img_pet, min_=0, max_=1e5)

    ref_img_ct = ct_images[0, :, :] * scale_ct
    ref_img_ct = np.where(ref_img_ct > 0, ref_img_ct, 0)

    ref_imgs = [ref_img_pet, ref_img_ct]

    w0 = 0.5

    low = -3
    high = 3
    x_init = np.random.uniform(low, high, latent_dim).flatten(order='F')
    max_iter = 1000

    objf = lambda x: get_objf(x, model, img_size, mode_=mode_, w0=w0, pet_ref_img=ref_imgs[0], ct_ref_img=ref_imgs[1])

    z, f, d = fmin_l_bfgs_b(objf, x_init, fprime=None, factr=1e12, pgtol=1e-12, iprint=-1, maxiter=max_iter)

    display_image_from_z(z, model, ref_imgs, slice_no=None, mode_=mode_)


def display_image_from_z(x, model, ref_imgs, slice_no, save_path=None, save_bool=False, mode_='petct'):
    gen_imgs = model(np.asarray([x]))
    if mode_ != 'petct':
        plt.imshow(gen_imgs.numpy().reshape(ref_imgs.shape))
    else:
        fig = plt.figure(figsize=(7.5, 5))
        rows = 1
        cols = 2
        ax1 = fig.add_subplot(rows, cols, 1)
        gen_pet_img = gen_imgs[0].numpy().reshape(ref_imgs[0].shape)
        vmax_pet = max(ref_imgs[0].max() * 1.05, gen_pet_img.max() * 1.05)
        psnr_pet = get_psnr(ref_imgs[0], gen_pet_img)
        ssim_pet = ssim(ref_imgs[0], gen_pet_img, data_range=1.0)
        ax1.text(0, 0, f'SSIM: {ssim_pet:.4f} \nPSNR: {psnr_pet:.4f}', color='green', fontsize=15, transform=ax1.transAxes, verticalalignment='bottom')
        plt.imshow(gen_pet_img[slice_no].reshape(ref_imgs[0].shape), vmax=vmax_pet)
        plt.axis('off')
        plt.title('Gen 1', fontsize=20)
        fig.add_subplot(rows, cols, 2)
        plt.imshow(ref_imgs[0][slice_no].reshape(ref_imgs[0].shape), vmax=vmax_pet)
        plt.axis('off')
        plt.title('Ref 1', fontsize=20)

        if save_bool:
            fig.savefig(f'{save_path}_1.png')

        fig2 = plt.figure(figsize=(7.5, 5))
        rows = 1
        cols = 2
        ax3 = fig2.add_subplot(rows, cols, 1)
        gen_ct_img = gen_imgs[1].numpy().reshape(ref_imgs[1].shape)
        vmax_ct = max(ref_imgs[1].max() * 1.05, gen_ct_img.max() * 1.05)
        psnr_ct = get_psnr(ref_imgs[1], gen_ct_img)
        ssim_ct = ssim(ref_imgs[1], gen_ct_img, data_range=1.0)
        ax3.text(0, 0, f'SSIM: {ssim_ct:.4f} \nPSNR: {psnr_ct:.4f}', color='green', fontsize=15, transform=ax3.transAxes, verticalalignment='bottom')
        plt.imshow(gen_ct_img[slice_no].reshape(ref_imgs[1].shape), vmax=vmax_ct)
        plt.axis('off')
        plt.title('Gen 2', fontsize=20)
        fig2.add_subplot(rows, cols, 2)
        plt.imshow(ref_imgs[1][slice_no].reshape(ref_imgs[1].shape), vmax=vmax_ct)
        plt.axis('off')
        plt.title('Ref 2', fontsize=20)

        if save_bool:
            fig2.savefig(f'{save_path}_2.png')

    plt.show()

if __name__ == "__main__":
    main()
