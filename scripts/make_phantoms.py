"""Generate paired synthetic PET/CT thorax phantoms.

Output: .npz with ``pet`` (N, n, n) normalised activity and ``ct_hu`` (N, n, n) in HU.
Your own data can be used with every other script if saved in the same format.

    python scripts/make_phantoms.py --n 400 --size 128 --seed 0 --out data/train.npz
"""

import argparse
from pathlib import Path

import numpy as np

from deepsynergy.data import make_phantoms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pet, ct_hu = make_phantoms(args.n, args.size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pet=pet.astype(np.float32), ct_hu=ct_hu.astype(np.float32))
    print(f"saved {args.n} phantom pairs ({args.size}x{args.size}) to {args.out}")


if __name__ == "__main__":
    main()
