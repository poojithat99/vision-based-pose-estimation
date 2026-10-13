"""Train PoseNet with cross-entropy + Adam. Saves best model (by val accuracy).

Usage:  python src/train.py [--epochs 30] [--batch 512] [--lr 1e-3]
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from model import PoseNet


def accuracy(model, X, y, device, bs=8192):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pred = model(X[i:i + bs].to(device)).argmax(1).cpu()
            correct += (pred == y[i:i + bs]).sum().item()
    return correct / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from models/pose_net.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(C.DATASET_PATH)
    Xtr, ytr = torch.from_numpy(d["X_train"]), torch.from_numpy(d["y_train"].astype(np.int64))
    Xva, yva = torch.from_numpy(d["X_val"]), torch.from_numpy(d["y_val"].astype(np.int64))
    print(f"device={device}  train={len(Xtr)}  val={len(Xva)}")

    model = PoseNet().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "train_acc": [], "val_acc": []}
    best = 0.0
    hist_path = C.RESULTS_DIR / "training_history.json"
    if args.resume and C.MODEL_PATH.exists():
        model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))
        best = accuracy(model, Xva, yva, device)
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
        print(f"resumed from checkpoint, val_acc {best:.4f}")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total, n = 0.0, 0
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        tr_acc = accuracy(model, Xtr[:50_000], ytr[:50_000], device)
        va_acc = accuracy(model, Xva, yva, device)
        sched.step(1 - va_acc)
        hist["train_loss"].append(total / n)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), C.MODEL_PATH)
        hist_path.write_text(json.dumps(hist, indent=2))
        print(f"epoch {ep:3d}  loss {total / n:.4f}  train_acc {tr_acc:.4f}  val_acc {va_acc:.4f}")

    print(f"Best val accuracy: {best:.4f}  -> saved {C.MODEL_PATH}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(hist["train_loss"]); ax[0].set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set(title="Top-1 accuracy (4800 classes)", xlabel="epoch"); ax[1].legend()
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
