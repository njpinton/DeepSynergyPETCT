import os
import re
import gc
import glob
import math
from pathlib import Path
from datetime import datetime

import numpy as np
from tensorflow import keras
from skimage.transform import resize
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import filters


def get_order(file):
    file_pattern = re.compile(r'.*?(\d+).*?')
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


def get_pet_ct_data(limit=50, image_size=64):
    dir_path_src = '/home/noel/data/lung-pet-ct-np1'
    file_list = sorted(glob.glob(os.path.join(dir_path_src, '*')), key=get_order)

    ct_list = [file for file in file_list if '_ct_' in file]
    pet_list = [file for file in file_list if '_pet_' in file]

    pet_ct_images = np.empty((limit, 128, image_size, image_size, 2))
    for idx2, pet_image_path in enumerate(tqdm(pet_list[:limit])):
        pet_image = np.load(pet_image_path)
        if image_size != 512:
            pet_image = resize(pet_image, (128, image_size, image_size))  # resize to save space
        # pet_images = np.append(pet_images, pet_image, axis=0)
        pet_ct_images[idx2, :, :, :, 0] = pet_image

    # ct_images = np.empty((0, image_size, image_size))
    for idx1, ct_image_path in enumerate(tqdm(ct_list[:limit])):
        ct_image = np.load(ct_image_path)
        if image_size != 512:
            ct_image = resize(ct_image, (128, image_size, image_size))  # resize to save space
        # ct_images = np.append(ct_images, ct_image, axis=0)
        pet_ct_images[idx1, :, :, :, 1] = ct_image

    return pet_ct_images


def normalise_zero_one(image, min_=None, max_=None):
    """Image normalisation. Normalises image to fit [0, 1] range.
    https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need"""

    """
    for pet, min=0 max=clip #1e5
    for ct, min=-1024, max=3072
    """
    image = image.astype(np.float32)

    if not min_:
        min_ = np.min(image)
    if not max_:
        max_ = np.max(image)

    minimum = min_
    maximum = max_

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def save_fig(img, fname):
    fig = plt.figure(frameon=False)
    # fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    fig.savefig(fname, bbox_inches='tight',
                transparent=True,
                pad_inches=0)


def get_edge(image):
    edge_roberts = filters.roberts(image)
    edge_sobel = filters.sobel(image)
    edge_scharr = filters.scharr(image)
    edge_prewitt = filters.prewitt(image)

    fig, axes = plt.subplots(ncols=5, sharex=True, sharey=True,
                             figsize=(12, 4))

    axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
    axes[0].set_title('Roberts Edge Detection')

    axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
    axes[1].set_title('Sobel Edge Detection')

    axes[2].imshow(edge_scharr, cmap=plt.cm.gray)
    axes[2].set_title('Scharr Edge Detection')

    axes[3].imshow(edge_prewitt, cmap=plt.cm.gray)
    axes[3].set_title('Prewitt Edge Detection')

    axes[4].imshow(image, cmap=plt.cm.gray)
    axes[4].set_title('Original Image')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_mnist_pairs(n=5):
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = x_train

    mnist_edge = np.empty(mnist_digits.shape)
    for idx in range(mnist_digits.shape[0]):
        mnist_edge[idx] = filters.roberts(mnist_digits[idx])

    fig1, axes1 = plt.subplots(ncols=n, nrows=n,
                               sharex=True, sharey=True,
                               figsize=(8, 8))

    fig2, axes2 = plt.subplots(ncols=n, nrows=n,
                              sharex=True, sharey=True,
                              figsize=(8, 8))

    for row1, row2 in zip(axes1, axes2):
        for ax1, ax2 in zip(row1, row2):
            idx1 = np.random.randint(mnist_digits.shape[0])
            ax1.imshow(filters.gaussian(mnist_digits[idx1], sigma=1.5))
            ax2.imshow(mnist_edge[idx1])
            ax1.axis('off')

    plt.imshow()


