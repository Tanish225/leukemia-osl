"""Classify one image or a whole folder with a trained checkpoint.
   python predict.py --ckpt runs/resnet18_osl_holdout/best_model.pt --input some_image.jpg
   python predict.py --ckpt runs/resnet18_osl_holdout/best_model.pt --input folder_of_images/
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from src.dataset import IMG_EXT, get_transforms
from src.model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--input", required=True)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
model = build_model(ck["model"], ck["head"], 2, ck["hidden"], ck["dropout"], pretrained=False)
model.load_state_dict(ck["state_dict"]); model.eval()
tf = get_transforms(ck["img_size"], False)
inp = Path(a.input)
files = sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT) if inp.is_dir() else [inp]
with torch.no_grad():
    for f in files:
        x = tf(Image.open(f).convert("RGB").resize((ck["img_size"],) * 2)).unsqueeze(0)
        p = torch.softmax(model(x), 1)[0]
        print(f"{f.name:40s} -> {ck['classes'][int(p.argmax())]:9s}  (leukemia prob = {p[1]:.3f})")
print("\nResearch/educational use only - not a medical diagnosis.")
