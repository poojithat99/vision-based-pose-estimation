"""Evaluate the trained PoseNet on the held-out test set.

Metrics
-------
* Top-1 / Top-5 classification accuracy over the 4800 pose labels
* "Within one bin" accuracy (predicted bin differs by <=1 in every pose axis)
* Pose error: predicted bin-centre pose vs. TRUE continuous pose
    - per-axis MAE (x, y, z in m, roll in deg), 3-D position error (mean/median/95%)
* Quantisation floor: error obtained if the label were classified perfectly
  (true bin centre vs. true pose) -> the best any 4800-label classifier can do.

Usage:  python src/evaluate.py
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from aircraft_model import label_to_pose, label_to_indices
from model import load_model


def predict_logits(model, X, bs=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs])))
    return torch.cat(out)


def pose_errors(pred_pose, true_pose):
    e = pred_pose - true_pose
    pos = np.linalg.norm(e[:, :3], axis=1)
    return e, pos


def summarize(e, pos):
    return {
        "MAE_x_m": float(np.abs(e[:, 0]).mean()),
        "MAE_y_m": float(np.abs(e[:, 1]).mean()),
        "MAE_z_m": float(np.abs(e[:, 2]).mean()),
        "MAE_roll_deg": float(np.abs(e[:, 3]).mean()),
        "pos_err_mean_m": float(pos.mean()),
        "pos_err_median_m": float(np.median(pos)),
        "pos_err_p95_m": float(np.percentile(pos, 95)),
    }


def main():
    d = np.load(C.DATASET_PATH)
    X, y, pose = d["X_test"], d["y_test"].astype(np.int64), d["pose_test"]
    model = load_model()
    logits = predict_logits(model, X)
    pred = logits.argmax(1).numpy()
    top5 = logits.topk(5, dim=1).indices.numpy()

    top1 = float((pred == y).mean())
    top5_acc = float((top5 == y[:, None]).any(1).mean())
    within1 = float((np.abs(label_to_indices(pred) - label_to_indices(y)) <= 1).all(1).mean())

    pred_pose = label_to_pose(pred)
    e_pred, pos_pred = pose_errors(pred_pose, pose)
    e_floor, pos_floor = pose_errors(label_to_pose(y), pose)

    # per-axis label accuracy (is each of the 4 bin indices correct?)
    axis_acc = (label_to_indices(pred) == label_to_indices(y)).mean(0)

    results = {
        "num_test_samples": int(len(y)),
        "top1_accuracy": top1,
        "top5_accuracy": top5_acc,
        "within_one_bin_accuracy": within1,
        "per_axis_bin_accuracy": dict(zip(["x", "y", "z", "roll"], map(float, axis_acc))),
        "model_pose_error": summarize(e_pred, pos_pred),
        "quantisation_floor_error": summarize(e_floor, pos_floor),
    }
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # ---- Plot 1: error histograms (model vs. quantisation floor) ----------
    names = ["x [m]", "y [m]", "z [m]", "roll [deg]"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for k in range(4):
        lim = np.percentile(np.abs(e_pred[:, k]), 99.5)
        bins = np.linspace(-lim, lim, 60)
        ax[k].hist(e_pred[:, k], bins=bins, alpha=0.6, label="model prediction")
        ax[k].hist(e_floor[:, k], bins=bins, alpha=0.6, label="perfect-label floor")
        ax[k].set(title=f"Error in {names[k]}", xlabel="predicted - true")
    ax[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "error_histograms.png", dpi=150); plt.close()

    # ---- Plot 2: actual vs predicted scatter ------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 4000, replace=False)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for k in range(4):
        ax[k].scatter(pose[idx, k], pred_pose[idx, k], s=3, alpha=0.4)
        lo, hi = C.POSE_LOW[k], C.POSE_HIGH[k]
        ax[k].plot([lo, hi], [lo, hi], "r--", lw=1)
        ax[k].set(title=f"{names[k]}: actual vs predicted", xlabel="actual", ylabel="predicted")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "actual_vs_predicted.png", dpi=150); plt.close()

    # ---- Plot 3: accuracy vs number of visible features -------------------
    nvis = X.reshape(-1, C.NUM_FEATURES, 3)[:, :, 2].sum(1).astype(int)
    ks = np.arange(nvis.min(), nvis.max() + 1)
    accs = [(pred[nvis == k] == y[nvis == k]).mean() if (nvis == k).sum() > 50 else np.nan for k in ks]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(ks, accs); ax.set(xlabel="# visible features", ylabel="top-1 accuracy",
                             title="Accuracy vs. visible features")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "accuracy_vs_visibility.png", dpi=150); plt.close()
    print("Saved plots to", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
