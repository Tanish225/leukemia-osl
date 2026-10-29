"""Reproduce the paper's comparison table: every backbone with FC head vs OSL head (same split & settings).
   python compare.py --data_dir data --mode holdout --epochs 30
Pass extra flags after the known ones, e.g. --no_pretrained."""
import argparse, json, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data")
ap.add_argument("--mode", default="holdout")
ap.add_argument("--epochs", default="30")
ap.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v2", "resnet50"])
a, extra = ap.parse_known_args()

rows = []
for m in a.models:
    for h in ["fc", "osl"]:
        out = f"runs/compare/{m}_{h}"
        subprocess.run([sys.executable, "train.py", "--data_dir", a.data_dir, "--model", m, "--head", h,
                        "--mode", a.mode, "--epochs", a.epochs, "--out", out] + extra, check=True)
        rows.append((f"{m}-{h}", json.load(open(Path(out) / "metrics.json"))["metrics"]))
lines = ["method,accuracy,precision,recall,specificity,f1,auc"]
print(f"\n{'Method':20s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'Spec':>7s} {'F1':>7s} {'AUC':>7s}")
for n, m in rows:
    print(f"{n:20s} {m['accuracy']:7.2f} {m['precision']:7.2f} {m['recall']:7.2f} {m['specificity']:7.2f} {m['f1']:7.4f} {m['auc']:7.4f}")
    lines.append(f"{n},{m['accuracy']:.2f},{m['precision']:.2f},{m['recall']:.2f},{m['specificity']:.2f},{m['f1']:.4f},{m['auc']:.4f}")
Path("runs/compare").mkdir(parents=True, exist_ok=True)
Path("runs/compare/summary.csv").write_text("\n".join(lines))
print("\nSaved runs/compare/summary.csv")
