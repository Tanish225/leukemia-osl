"""Train / evaluate the ResNet18 + OSL leukemia classifier (and the baselines from the paper).

  python train.py --data_dir data --mode holdout            # 70/30 split (paper's 2nd protocol)
  python train.py --data_dir data --mode cv --folds 10      # 10-fold CV (paper's 1st protocol)
  python train.py --data_dir data --head fc                 # same backbone with a plain FC head (ablation)
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.dataset import CLASSES, ImageCache, SubsetDS, get_transforms, load_samples
from src.model import BACKBONES, build_model
from src.utils import (compute_metrics, metrics_from_counts, plot_confusion, plot_history, plot_roc,
                       save_json, set_seed)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model", default="resnet18", choices=BACKBONES)
    p.add_argument("--head", default="osl", choices=["osl", "fc"])
    p.add_argument("--mode", default="holdout", choices=["holdout", "cv"])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.30)
    p.add_argument("--val_frac", type=float, default=0.15, help="part of TRAIN used to pick the best epoch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true", help="skip ImageNet weights (offline / quick test)")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, probs, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += ce(out, y).item(); n += len(y)
        probs.append(torch.softmax(out, 1)[:, 1].cpu()); ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy(), loss_sum / max(n, 1)


def fit(model, train_ds, val_ds, a, device):
    tl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                    drop_last=len(train_ds) > a.batch_size)
    vl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ce = nn.CrossEntropyLoss()
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_state, best = None, (-1.0, 1e9)
    for ep in range(a.epochs):
        model.train(); tot, correct, ls = 0, 0, 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); loss = ce(out, y); loss.backward(); opt.step()
            ls += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); tot += len(y)
        sched.step()
        yv, pv, vloss = predict_loader(model, vl, device)
        vacc = 100 * ((pv >= 0.5).astype(int) == yv).mean()
        hist["train_loss"].append(ls / tot); hist["train_acc"].append(100 * correct / tot)
        hist["val_loss"].append(vloss); hist["val_acc"].append(float(vacc))
        if (vacc, -vloss) > (best[0], -best[1]):
            best = (vacc, vloss); best_state = copy.deepcopy(model.state_dict())
        print(f"  epoch {ep + 1:3d}/{a.epochs}  train loss {ls / tot:.4f} acc {100 * correct / tot:5.1f}%  "
              f"| val loss {vloss:.4f} acc {vacc:5.1f}%")
    model.load_state_dict(best_state)
    return model, hist


def run_split(cache, labels, tr_idx, te_idx, a, device):
    tr_idx, va_idx = train_test_split(tr_idx, test_size=a.val_frac, stratify=labels[tr_idx], random_state=a.seed)
    tr = SubsetDS(cache, labels, tr_idx, get_transforms(a.img_size, True))
    va = SubsetDS(cache, labels, va_idx, get_transforms(a.img_size, False))
    te = SubsetDS(cache, labels, te_idx, get_transforms(a.img_size, False))
    model = build_model(a.model, a.head, 2, a.hidden, a.dropout, not a.no_pretrained).to(device)
    model, hist = fit(model, tr, va, a, device)
    yt, pt, _ = predict_loader(model, DataLoader(te, batch_size=a.batch_size), device)
    return model, hist, yt, pt


def main(argv=None):
    a = get_args(argv)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out or f"runs/{a.model}_{a.head}_{a.mode}"); out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | model: {a.model}+{a.head} | mode: {a.mode} | output: {out}")

    paths, labels = load_samples(a.data_dir)
    print(f"Loaded {len(paths)} images  (healthy={int((labels == 0).sum())}, leukemia={int((labels == 1).sum())})")
    cache = ImageCache(paths, a.img_size)
    idx = np.arange(len(labels)); t0 = time.time()

    if a.mode == "holdout":
        tr, te = train_test_split(idx, test_size=a.test_size, stratify=labels, random_state=a.seed)
        model, hist, yt, pt = run_split(cache, labels, tr, te, a, device)
        m = compute_metrics(yt, pt)
        plot_history(hist, out / "training_curves.png")
        plot_confusion(m, out / "confusion_matrix.png"); plot_roc(yt, pt, out / "roc.png", m["auc"])
        torch.save({"state_dict": model.state_dict(), "model": a.model, "head": a.head, "hidden": a.hidden,
                    "dropout": a.dropout, "img_size": a.img_size, "classes": CLASSES}, out / "best_model.pt")
        result = {"mode": "holdout", "args": vars(a), "metrics": m}
    else:
        folds = min(a.folds, int(np.bincount(labels).min()))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=a.seed)
        all_y, all_p, per_fold = [], [], []
        for k, (tr, te) in enumerate(skf.split(idx, labels), 1):
            print(f"\n=== Fold {k}/{folds} ===")
            _, _, yt, pt = run_split(cache, labels, tr, te, a, device)
            per_fold.append(compute_metrics(yt, pt)); all_y.append(yt); all_p.append(pt)
        yt, pt = np.concatenate(all_y), np.concatenate(all_p)
        m = compute_metrics(yt, pt)     # pooled over folds (TN/FP/FN/TP sum to the dataset size, like Tables 8-11)
        m["accuracy_mean_over_folds"] = float(np.mean([f["accuracy"] for f in per_fold]))
        m["accuracy_std_over_folds"] = float(np.std([f["accuracy"] for f in per_fold]))
        plot_confusion(m, out / "confusion_matrix.png"); plot_roc(yt, pt, out / "roc.png", m["auc"])
        result = {"mode": "cv", "folds": folds, "args": vars(a), "metrics": m, "per_fold": per_fold}

    result["train_time_sec"] = round(time.time() - t0, 1)
    save_json(result, out / "metrics.json")
    print("\n=========== FINAL TEST RESULTS ===========")
    for k in ["accuracy", "precision", "recall", "specificity", "f1", "auc", "TN", "FP", "FN", "TP"]:
        v = m[k]; print(f"{k:12s}: {v:.4f}" if isinstance(v, float) else f"{k:12s}: {v}")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
