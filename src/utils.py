import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def metrics_from_counts(tn, fp, fn, tp, auc=float("nan")):
    e = 1e-12
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / (tp + fp + e)
    rec = tp / (tp + fn + e)          # recall = sensitivity
    spec = tn / (tn + fp + e)
    f1 = 2 * prec * rec / (prec + rec + e)
    return dict(accuracy=100 * acc, precision=100 * prec, recall=100 * rec, specificity=100 * spec,
                f1=f1, auc=auc, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thr).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum()); tp = int(((y_true == 1) & (y_pred == 1)).sum())
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) == 2 else float("nan")
    return metrics_from_counts(tn, fp, fn, tp, auc)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_confusion(m, path, title="Confusion matrix"):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["healthy", "leukemia"]); ax.set_yticklabels(["healthy", "leukemia"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, y_prob, path, auc):
    if len(set(np.asarray(y_true).tolist())) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}"); ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
