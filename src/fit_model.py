"""
Train β-VAE on PET/CT Patches
-----------------------------

This script trains a β-VAE or dual-branch β-VAE (for PET/CT image patches) using TensorFlow/Keras.
It supports configurable architecture parameters and training hyperparameters via CLI.

Example usage:
    python train_vae.py \
        --patches_path ./data/patches_petct.npz \
        --image_size 64 \
        --latent_dim 128 \
        --inter_dim 64 \
        --batch_size 1024 \
        --filters 32 \
        --layers_n 3 \
        --epoch 200 \
        --mode petct \
        --loss_type H \
        --merge_type concatenate

Requirements:
    - TensorFlow 2.x
    - NumPy
    - Keras
    - Custom model definitions in models/betavae.py
"""

import os
import argparse
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf
import keras

from models.betavae import build_dual_betavae, build_betavae


def input_fn(x_train1: np.ndarray,
             x_train2: np.ndarray,
             img_size: int = 64,
             batch_size: int = 32,
             buffer_size: int = 32,
             shuffle: bool = True) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset for training with two image modalities (e.g., PET and CT).

    Args:
        x_train1: First modality image data, shape (N, H, W).
        x_train2: Second modality image data, shape (N, H, W).
        img_size: Target image size (assumes square images).
        batch_size: Batch size for training.
        buffer_size: Buffer size for dataset shuffling.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A tf.data.Dataset that yields dictionaries with two modality inputs.
    """
    x_train1 = x_train1.reshape([-1, img_size, img_size, 1])
    x_train2 = x_train2.reshape([-1, img_size, img_size, 1])

    def generator():
        for i1, i2 in zip(x_train1, x_train2):
            yield {"input_1": i1, "input_2": i2}

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types={"input_1": tf.float32, "input_2": tf.float32})
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset.batch(batch_size)


def create_callbacks(model_name: str, epoch: int) -> Tuple[tf.keras.callbacks.TensorBoard, tf.keras.callbacks.ModelCheckpoint, str]:
    """
    Constructs TensorBoard and model checkpoint callbacks.

    Args:
        model_name: Name for the experiment/model to be used in log paths.
        epoch: Number of training epochs (used in checkpoint filename).

    Returns:
        Tuple containing TensorBoard callback, checkpoint callback, and a time tag string.
    """
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{model_name}_{date_time}"
    checkpoint_path = f"tmp/{model_name}_{date_time}/checkpoint_{epoch}"

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='total_loss',
        mode='min',
        save_best_only=True,
        save_freq=20
    )
    return tensorboard_cb, checkpoint_cb, date_time


def build_model(mode: str,
                input_shape: Tuple[int, int, int],
                inter_dim: int,
                latent_dim: int,
                filters: int,
                layers_n: int,
                g: float,
                c_max: float,
                c_stop_iter: int,
                loss_type: str,
                merge_type: str):
    """
    Initializes a β-VAE model with the appropriate architecture.

    Args:
        mode: Mode of the model: "petct" for dual-modality, otherwise single.
        input_shape: Shape of input images (H, W, C).
        inter_dim: Intermediate representation dimension.
        latent_dim: Latent space dimension.
        filters: Initial number of convolutional filters.
        layers_n: Number of convolutional layers.
        g: Weight of total correlation loss.
        c_max: Max capacity for KL divergence.
        c_stop_iter: Iteration to stop increasing KL capacity.
        loss_type: 'B' for β-VAE, 'H' for Higgins’ β-VAE.
        merge_type: Strategy to merge dual branches ("concatenate", "add", "average").

    Returns:
        A compiled Keras model.
    """
    builder = build_dual_betavae if mode == 'petct' else build_betavae
    model = builder(input_shape, inter_dim, latent_dim, filters, layers_n,
                    g, c_max, c_stop_iter, loss_type, merge_type)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    return model


def save_model_components(model, save_dir: str):
    """
    Saves the encoder and decoder sub-models to disk.

    Args:
        model: Trained VAE model with encoder and decoder attributes.
        save_dir: Path to save model components.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.encoder.save(os.path.join(save_dir, "encoder"))
    model.decoder.save(os.path.join(save_dir, "decoder"))


def prepare_data(patches_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads PET/CT image data from a .npz archive.

    Args:
        patches_path: Path to .npz file containing (N, H, W, 2) images.

    Returns:
        Tuple of (PET images, scaled CT images, scale factor).
    """
    data = np.load(patches_path)['arr_0']
    pet = data[..., 0]
    ct = data[..., 1]
    scale_ct = 1 / ct.max()
    return pet, ct * scale_ct, scale_ct


def train_experiment(config: Dict):
    """
    Runs a full training cycle for a β-VAE model based on provided configuration.

    Args:
        config: Dictionary of training hyperparameters and paths.
    """
    pet, ct, scale = prepare_data(config['patches_path'])

    print(f'[INFO] CT scale factor = {scale}')
    print(f'[INFO] PET max = {pet.max()}, CT max = {ct.max()}')

    input_shape = (config['image_size'], config['image_size'], 1)
    model = build_model(config['mode'], input_shape, config['inter_dim'], config['latent_dim'],
                        config['filters'], config['layers_n'], config['g'],
                        config['c_max'], config['c_stop_iter'],
                        config['loss_type'], config['merge_type'])
    model.optimizer.lr.assign(config['learning_rate'])

    model_id = (
        f'beta_vae_{config["mode"]}_lungs{config["image_size"]}x{config["image_size"]}_'
        f'p{config["num_patients"]}_lr{config["learning_rate"]}_ld{config["latent_dim"]}_'
        f'id{config["inter_dim"]}_batch{config["batch_size"]}_f{config["filters"]}_l{config["layers_n"]}_'
        f'e{config["epoch"]}_m{config["merge_type"]}_g{config["g"]}{config["loss_type"]}_cmax{config["c_max"]}'
    )
    tensorboard_cb, checkpoint_cb, time_tag = create_callbacks(model_id, config['epoch'])

    dataset = input_fn(pet, ct, img_size=config['image_size'], batch_size=config['batch_size'])
    model.fit(dataset, epochs=config['epoch'], verbose=2, callbacks=[tensorboard_cb, checkpoint_cb])

    save_model_components(model, f"saved_models/betavae/{model_id}_{time_tag}")


def parse_args() -> Dict:
    """
    Parses command-line arguments and returns them as a dictionary.

    Returns:
        Dictionary of parsed hyperparameters and configuration options.
    """
    parser = argparse.ArgumentParser(description="Train PET/CT β-VAE Model")
    parser.add_argument('--patches_path', type=str, required=True, help='Path to .npz file of PET/CT patches')
    parser.add_argument('--image_size', type=int, default=64, help='Input image size (H=W)')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent space dimension')
    parser.add_argument('--inter_dim', type=int, default=128, help='Intermediate representation dimension')
    parser.add_argument('--batch_size', type=int, default=2048, help='Training batch size')
    parser.add_argument('--filters', type=int, default=64, help='Number of initial conv filters')
    parser.add_argument('--layers_n', type=int, default=4, help='Number of convolutional layers')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--g', type=float, default=1.0, help='Weight on total correlation loss term')
    parser.add_argument('--c_max', type=float, default=50.0, help='Max capacity for KL divergence')
    parser.add_argument('--loss_type', choices=['B', 'H'], default='H', help='Loss type: B for β-VAE, H for Higgins')
    parser.add_argument('--mode', choices=['petct', 'pet', 'ct'], default='petct', help='Model input mode')
    parser.add_argument('--merge_type', choices=['concatenate', 'add', 'average'], default='concatenate', help='Merge method for dual input')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_patients', type=int, default=316, help='Number of patient samples')

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['c_stop_iter'] = args_dict['epoch'] * 10  # Derived parameter
    return args_dict


if __name__ == "__main__":
    config = parse_args()
    train_experiment(config)
