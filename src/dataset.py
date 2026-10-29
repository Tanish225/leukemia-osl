"""Dataset loading with automatic label inference.

Works with these layouts (any depth, images are found recursively):
  1) ALL-IDB original names:  Im001_1.jpg  -> '_1' = leukemia (ALL), '_0' = healthy
  2) Class folders:           .../healthy/*.png   .../all/*.png   (also normal/hem/non_aml  vs  aml/leukemia/blast/cancer ...)
  3) Class in the file name:  healthy_01.jpg, AML_12.jpg
Check what was inferred with:   python -m src.dataset --data_dir data
Label 0 = healthy / negative, label 1 = leukemia / positive.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = ["healthy", "leukemia"]
NEG = {"hem", "healthy", "normal", "negative", "neg", "nonaml", "nonall", "benign", "noncancer", "non", "control"}
POS = {"all", "aml", "leukemia", "leukaemia", "leukemic", "blast", "blasts", "sick", "positive", "pos",
       "cancer", "malignant", "abnormal"}
IGNORE_PART = re.compile(r"all[_\-\s]?idb|^(data|dataset|train|test|val|images?|img|im|raw)$", re.I)
ALLIDB_NAME = re.compile(r"^im\d+_([01])\.", re.I)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _tokens(s):
    return set(re.split(r"[^a-z0-9]+", s.lower())) - {""}


def _label_from_tokens(toks):
    if toks & NEG:
        return 0
    if toks & POS:
        return 1
    return None


def infer_label(path: Path, root: Path):
    m = ALLIDB_NAME.match(path.name)
    if m:
        return int(m.group(1))
    for part in reversed(path.relative_to(root).parts[:-1]):   # nearest folder first
        if IGNORE_PART.search(part):
            continue
        lab = _label_from_tokens(_tokens(part))
        if lab is not None:
            return lab
    return _label_from_tokens(_tokens(path.stem))


def load_samples(data_dir):
    root = Path(data_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"No images found under '{root}'. Did you download/extract the dataset there?")
    paths, labels, unknown = [], [], []
    for p in files:
        lab = infer_label(p, root)
        if lab is None:
            unknown.append(p)
        else:
            paths.append(p)
            labels.append(lab)
    if unknown:
        ex = "\n  ".join(str(u) for u in unknown[:5])
        raise SystemExit(f"Could not infer the class of {len(unknown)} image(s), e.g.\n  {ex}\n"
                         "Rename/arrange files as described in the README (class folders such as healthy/ and all/).")
    cnt = Counter(labels)
    if len(cnt) < 2:
        raise SystemExit(f"Only one class found {dict(cnt)} - need both healthy and leukemia images.")
    return paths, np.array(labels)


class ImageCache:
    """Loads every image ONCE, resized to img_size x img_size (datasets are small, so this fits in RAM)."""
    def __init__(self, paths, img_size):
        self.images = []
        for p in paths:
            with Image.open(p) as im:
                self.images.append(im.convert("RGB").resize((img_size, img_size), Image.BILINEAR))


def get_transforms(img_size, train):
    if train:
        return T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
                          T.ColorJitter(0.15, 0.15, 0.1, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class SubsetDS(Dataset):
    def __init__(self, cache, labels, idx, transform):
        self.cache, self.labels, self.idx, self.tf = cache, labels, np.asarray(idx), transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(self.cache.images[j]), int(self.labels[j])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    a = ap.parse_args()
    paths, labels = load_samples(a.data_dir)
    print(f"Found {len(paths)} images")
    for k, name in enumerate(CLASSES):
        print(f"  {name:9s} (label {k}): {(labels == k).sum()}")
    print("Example:", paths[0].name, "->", CLASSES[labels[0]])
