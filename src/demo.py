"""Live-demo script: a 10 s formation-approach trajectory (as in the paper's Fig. 6).

For each frame we (1) simulate the detected feature vector with noise/occlusion,
(2) run the trained network, (3) compare predicted vs actual pose.

Outputs: console table, results/demo_trajectory.png, results/demo_frame.png

Usage:  python src/demo.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import aircraft_model as A
from model import load_model


def make_trajectory(n=100, T=10.0):
    t = np.linspace(0, T, n)
    x = -90 + 5.0 * t              # closing from 90 m to 40 m behind the leader
    y = 25 - 1.0 * t               # lateral offset drifting 25 -> 15 m
    z = np.full_like(t, 15.0)      # constant vertical offset
    phi = 15.0 * np.sin(2 * np.pi * t / T)   # gentle roll oscillation
    return t, np.stack([x, y, z, phi], axis=1)


def main():
    model = load_model()
    rng = np.random.default_rng(42)
    t, pose = make_trajectory()
    feats = A.render_features(pose, rng)                      # simulated detections
    with torch.no_grad():
        pred_label = model(torch.from_numpy(feats)).argmax(1).numpy()
    pred = A.label_to_pose(pred_label)
    true_label = A.pose_to_label(pose)

    # ---- console table -------------------------------------------------------
    print(f"{'t[s]':>5} | {'ACTUAL x,y,z,roll':^30} | {'PREDICTED x,y,z,roll':^30} | pos err[m]")
    for i in range(0, len(t), 10):
        a, p = pose[i], pred[i]
        print(f"{t[i]:5.1f} | {a[0]:7.1f} {a[1]:6.1f} {a[2]:6.1f} {a[3]:6.1f}   | "
              f"{p[0]:7.1f} {p[1]:6.1f} {p[2]:6.1f} {p[3]:6.1f}   | {np.linalg.norm(a[:3] - p[:3]):6.1f}")
    err = pred - pose
    print(f"\nLabel accuracy on trajectory: {(pred_label == true_label).mean():.2%}")
    print(f"Mean |error|: x={np.abs(err[:, 0]).mean():.1f} m, y={np.abs(err[:, 1]).mean():.1f} m, "
          f"z={np.abs(err[:, 2]).mean():.1f} m, roll={np.abs(err[:, 3]).mean():.1f} deg")
    print(f"Mean 3-D position error: {np.linalg.norm(err[:, :3], axis=1).mean():.1f} m")

    # ---- plot 1: actual vs predicted over time ---------------------------------
    labels = ["x (forward) [m]", "y (lateral) [m]", "z (vertical) [m]", "roll [deg]"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    for k, a in enumerate(ax.ravel()):
        a.plot(t, pose[:, k], "k-", lw=2, label="actual")
        a.step(t, pred[:, k], "r-", lw=1.2, where="mid", label="predicted (NN)")
        a.set(title=labels[k], xlabel="time [s]"); a.grid(alpha=0.3)
    ax[0, 0].legend()
    plt.suptitle("Formation approach: actual vs predicted relative pose")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "demo_trajectory.png", dpi=150); plt.close()

    # ---- plot 2: what the network sees at one frame ----------------------------
    i = len(t) // 2
    f = feats[i].reshape(C.NUM_FEATURES, 3)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vis = f[:, 2] > 0.5
    ax.scatter(f[vis, 0], -f[vis, 1], c="tab:blue", label="detected")
    for k in np.where(vis)[0]:
        ax.annotate(A.FEATURE_NAMES[k], (f[k, 0], -f[k, 1]), fontsize=7)
    ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), title=f"Detected features at t={t[i]:.1f}s "
           f"({vis.sum()}/14 visible)\nnormalised pixel coords", aspect="equal")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "demo_frame.png", dpi=150); plt.close()
    print("Saved demo plots to", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
