"""Generate the synthetic pose dataset.

Each sample: uniformly sampled continuous pose (x, y, z, phi) -> projected/occluded
feature vector (42) + noise -> label = index of the pose bin (0..4799).

Usage:  python src/generate_dataset.py [--n 400000] [--seed 0]
"""
import argparse
import numpy as np
import config as C
from aircraft_model import sample_poses, render_features, pose_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000, help="total samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    poses = sample_poses(args.n, rng).astype(np.float32)
    X = np.concatenate([render_features(poses[i:i + 50_000], rng)
                        for i in range(0, args.n, 50_000)])
    y = pose_to_label(poses).astype(np.int16)

    # 70 / 15 / 15 split
    perm = rng.permutation(args.n)
    n_tr, n_va = int(0.70 * args.n), int(0.15 * args.n)
    splits = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    save = {}
    for name, idx in splits.items():
        save[f"X_{name}"], save[f"y_{name}"], save[f"pose_{name}"] = X[idx], y[idx], poses[idx]
    np.savez_compressed(C.DATASET_PATH, **save)

    vis = X.reshape(-1, C.NUM_FEATURES, 3)[:, :, 2]
    print(f"Saved {C.DATASET_PATH}")
    print(f"  samples: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"  input dim: {X.shape[1]}, classes: {C.NUM_CLASSES}, distinct labels seen: {len(np.unique(y))}")
    print(f"  mean visible features per image: {vis.sum(1).mean():.2f} / {C.NUM_FEATURES}")


if __name__ == "__main__":
    main()
