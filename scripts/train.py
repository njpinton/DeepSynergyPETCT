"""Train the two-branch beta-VAE on paired PET/CT patches.

    python scripts/train.py --data data/train.npz --vox-mm 3.9 --patch 32 --stride 8 \
        --latent-dim 32 --filters 32 --layers 3 --epochs 30 --out models/phantom

Thesis setting (512x512 slices, 0.977 mm pixels):
    --patch 64 --stride 16 --latent-dim 64 --inter-dim 128 --filters 64 --layers 4 \
    --loss-type B --gamma 5 --c-max 100 --batch-size 1500 --epochs 500
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from deepsynergy.data import make_training_patches
from deepsynergy.model_io import save_model
from deepsynergy.models import build_dual_betavae


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help=".npz with pet and ct_hu arrays")
    ap.add_argument("--vox-mm", type=float, required=True, help="pixel size in mm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--inter-dim", type=int, default=128)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--merge", choices=["concatenate", "add"], default="concatenate")
    ap.add_argument("--loss-type", choices=["H", "B"], default="H")
    ap.add_argument("--gamma", type=float, default=1.0, help="KL weight")
    ap.add_argument("--c-max", type=float, default=50.0, help="max KL capacity (loss B)")
    ap.add_argument("--recon-loss", choices=["bce", "mse"], default="bce")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    data = np.load(args.data)
    p_pet, p_ct, meta = make_training_patches(data["pet"], data["ct_hu"], args.vox_mm, args.patch, args.stride)
    print(f"{len(p_pet)} patch pairs, pet_scale={meta['pet_scale']:.4g}, ct_scale={meta['ct_scale']:.4g}")

    idx = np.random.default_rng(args.seed).permutation(len(p_pet))
    n_val = int(len(idx) * args.val_frac)

    def dataset(ix, shuffle):
        ds = tf.data.Dataset.from_tensor_slices({"input_1": p_pet[ix, ..., None], "input_2": p_ct[ix, ..., None]})
        if shuffle:
            ds = ds.shuffle(len(ix), seed=args.seed)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = int(np.ceil((len(idx) - n_val) / args.batch_size))
    vae = build_dual_betavae(patch=args.patch, latent_dim=args.latent_dim, inter_dim=args.inter_dim,
                             filters=args.filters, n_layers=args.layers, merge=args.merge,
                             gamma=args.gamma, loss_type=args.loss_type, c_max=args.c_max,
                             c_stop_iter=args.epochs * steps_per_epoch, recon_loss=args.recon_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(args.lr))
    hist = vae.fit(dataset(idx[n_val:], True), validation_data=dataset(idx[:n_val], False) if n_val else None,
                   epochs=args.epochs, verbose=2)

    meta.update(vars(args) | {"data": str(args.data), "out": str(args.out)})
    save_model(vae, args.out, meta)
    (args.out / "history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}))
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
