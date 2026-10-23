"""Simulate PET/CT data from a ground-truth pair and reconstruct it with
(1) independent MLEM / PWLS and (2) the synergistic multi-branch prior.

    python scripts/reconstruct.py --config configs/phantom.json --model models/phantom \
        --data data/test.npz --index 0 --out results/phantom

Outputs: metrics.json, images.npz and a comparison figure (comparison.png).
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deepsynergy.config import Config  # noqa: E402
from deepsynergy.experiment import simulate  # noqa: E402
from deepsynergy.metrics import report  # noqa: E402
from deepsynergy.model_io import load_prior  # noqa: E402
from deepsynergy.recon import ReconResult, joint_reconstruct  # noqa: E402


def save_figure(path, rows):
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.4 * len(rows)))
    for r, (name, ref, imgs) in enumerate(rows):
        vmax = ref.max()
        for c, (title, img) in enumerate([("ground truth", ref)] + imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray" if name == "CT" else "hot", vmin=0, vmax=vmax)
            ax.set_title(f"{name}: {title}" + ("" if c == 0 else f"\nPSNR {report(ref, img)['psnr']:.2f} dB"),
                         fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help=".npz with pet and ct_hu arrays")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pet-scale", type=float, help="only for legacy models without meta.json")
    ap.add_argument("--ct-scale", type=float, help="only for legacy models without meta.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = Config.load(args.config)
    rng = np.random.default_rng(cfg.seed)
    data = np.load(args.data)
    pet_true, ct_hu = data["pet"][args.index].astype(np.float64), data["ct_hu"][args.index].astype(np.float64)

    sim = simulate(cfg, pet_true, ct_hu, rng)
    r = cfg.recon
    init = ReconResult(pet=sim.pet.mlem(np.ones_like(pet_true), r.init_iters), ct=sim.ct_init)
    # baselines get the same number of data updates as the joint reconstruction
    base = ReconResult(pet=sim.pet.mlem(init.pet, r.outer_iters * r.pet_subiters),
                       ct=sim.ct.pwls(init.ct, r.outer_iters * r.ct_subiters))

    prior = load_prior(args.model, args.pet_scale, args.ct_scale)
    joint = joint_reconstruct(sim.pet, sim.ct, prior, r, init=init, rng=rng)

    metrics = {
        "pet": {"mlem": report(pet_true, base.pet), "joint": report(pet_true, joint.pet),
                "generated": report(pet_true, joint.pet_gen)},
        "ct": {"pwls": report(sim.mu_true, base.ct), "joint": report(sim.mu_true, joint.ct),
               "generated": report(sim.mu_true, joint.ct_gen)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    cfg.save(args.out / "config.json")
    np.savez_compressed(args.out / "images.npz", pet_true=pet_true, mu_true=sim.mu_true,
                        pet_mlem=base.pet, ct_pwls=base.ct, pet_joint=joint.pet, ct_joint=joint.ct,
                        pet_gen=joint.pet_gen, ct_gen=joint.ct_gen)
    save_figure(args.out / "comparison.png", [
        ("PET", pet_true, [("MLEM", base.pet), ("joint", joint.pet), ("generated", joint.pet_gen)]),
        ("CT", sim.mu_true, [("PWLS", base.ct), ("joint", joint.ct), ("generated", joint.ct_gen)]),
    ])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
