# Acute Leukemia Detection — ResNet18 + Orthogonal Softmax Layer (OSL)

Re-implementation of *"An Efficient Detection and Classification of Acute Leukemia Using Transfer Learning and
Orthogonal Softmax Layer-Based Model"* (Das, Sahoo, Meher — IEEE/ACM TCBB 2023) in PyTorch.

**Idea in one line:** take an ImageNet-pretrained ResNet18, and replace its normal fully-connected classifier with
`FC → Dropout → ReLU → Dropout → ReLU → OSL`. OSL is a classifier layer whose weights are multiplied by a fixed 0/1
mask so each class uses its own separate set of features → class weight vectors are **orthogonal** → less overfitting
on tiny medical datasets.

```
leukemia-osl/
├── train.py            # train + evaluate (holdout 70/30 or k-fold CV)
├── predict.py          # classify new images with a saved model
├── compare.py          # reproduces the paper's "FC vs OSL" comparison table
├── download_data.py    # pulls the Google Drive dataset into ./data
├── src/osl.py          # the Orthogonal SoftMax Layer  (the core of the paper)
├── src/model.py        # ResNet18 (+ResNet50/MobileNetV2/VGG16) with FC or OSL head
├── src/dataset.py      # image loading + automatic label detection
├── src/utils.py        # metrics, ROC / confusion-matrix plots
└── tests/test_osl.py   # checks the weights really are orthogonal
```

---

## Step-by-step

### Step 0 — What you need
Python 3.9+ . A GPU is nice but **not required** (the datasets are tiny; CPU works, just slower).
Easiest free GPU: **Google Colab** (see the Colab section at the bottom).

### Step 1 — Create an environment and install
```bash
cd leukemia-osl
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
(For an NVIDIA GPU on your PC, install the CUDA build of PyTorch first from pytorch.org, then the rest.)

### Step 2 — Check the code works
```bash
python tests/test_osl.py          # should print: all tests passed
```

(OR)

```bash
PYTHONPATH=. python tests/test_osl.py         # should print: all tests passed
```

### Step 3 — Get the dataset into `data/`
**Option A (automatic):**
```bash
python download_data.py
```
**Option B (manual, if A fails — Drive often limits folder downloads):** open the Drive link in your browser →
right-click the folder → *Download* → unzip everything inside `leukemia-osl/data/`.

Any folder structure is fine; images are found recursively. Labels are detected automatically from:
* ALL-IDB file names — `Im001_1.jpg` (`_1` = leukemia/ALL, `_0` = healthy)
* class folders — `healthy / normal / hem / non_aml`  vs  `all / aml / leukemia / blast`
* or the class word inside the file name

### Step 4 — Verify the labels (important!)
```bash
python -m src.dataset --data_dir data
```
You should see something like `healthy: 59`, `leukemia: 49` for ALL-IDB1.
If it says it can't infer labels, rename/arrange files into `data/healthy/` and `data/leukemia/`.

### Step 5 — Train the proposed model (ResNet18 + OSL)
Quick run (70% train / 30% test, same as the paper's second protocol):
```bash
python train.py --data_dir data --mode holdout --epochs 30
```
No internet / want a 1-minute smoke test? add `--no_pretrained --epochs 2`.

Paper's main protocol, 10-fold cross-validation (slower):
```bash
python train.py --data_dir data --mode cv --folds 10 --epochs 30
```

### Step 6 — Read the results
Everything is saved in `runs/resnet18_osl_holdout/` :
| file | what |
|---|---|
| `metrics.json` | accuracy, precision, recall, specificity, F1, AUC, TN/FP/FN/TP |
| `confusion_matrix.png`, `roc.png`, `training_curves.png` | plots for your report |
| `best_model.pt` | trained weights (holdout mode) |

### Step 7 — Reproduce the comparison (OSL vs normal FC head)
```bash
python compare.py --data_dir data --mode holdout --epochs 30
```
Trains ResNet18 / MobileNetV2 / ResNet50 each with a plain FC head and with OSL, prints a table and saves
`runs/compare/summary.csv` (this mirrors Tables 12–15 of the paper). To also test the ablation only:
`python train.py --head fc` vs `python train.py --head osl`.

### Step 8 — Predict on new images
```bash
python predict.py --ckpt runs/resnet18_osl_holdout/best_model.pt --input path/to/image_or_folder
```

---

## Useful options (`python train.py -h` for all)
| flag | default | meaning |
|---|---|---|
| `--model` | resnet18 | resnet18 / resnet50 / mobilenet_v2 / vgg16 |
| `--head` | osl | `osl` (paper) or `fc` (baseline) |
| `--epochs` | 30 | |
| `--batch_size` | 16 | lower to 8 if you run out of memory |
| `--lr` | 1e-4 | Adam learning rate |
| `--img_size` | 224 | try 256/320 — ALL-IDB1 images are huge (2592×1944), so cells get small at 224 |
| `--seed` | 42 | change it to see run-to-run variation |

## How the code maps to the paper
* **Fig. 1 architecture** → `src/model.py` (`_make_head`, `head="osl"`)
* **Eq. (10) `Y = softmax((M ∘ W) X)`** → `src/osl.py` (`self.weight * self.mask`; softmax is inside `CrossEntropyLoss`)
* **Mask M** → fixed buffer, feature *i* is wired to class `i mod n_classes`; never changes in training or testing
* **Tables 8–11 (TN/FP/FN/TP)** → CV mode sums the confusion matrix over folds, so the counts add up to the dataset size
* **70/30 protocol, Tables 12–19** → `--mode holdout`

## Troubleshooting
* `No images found` → data not inside `data/`, or Step 3 failed.
* `Could not infer the class` → arrange files into `healthy/` and `leukemia/` folders.
* `CUDA out of memory` → `--batch_size 8`.
* gdown "Too many files / quota" → use the manual download (Step 3, option B).
* Download of pretrained weights fails (offline) → use `--no_pretrained` (accuracy will be much lower).

## Honest notes (good to mention in a report/viva)
* The datasets are tiny (ALL-IDB1 = 108 images, ASH = 80), so results swing a lot with the split/seed. Run
  several seeds and report mean ± std; don't expect to hit exactly 99.39% as in the paper.
* A small part of the training set is held out to choose the best epoch, so the test set is never used for tuning
  (the paper doesn't describe this; it is the safer choice).
* AUC here is computed from predicted probabilities (the paper's AUC values look like they come from hard labels).
* C-NMC 2019 has many cells per patient — split by *patient* if you use it, otherwise the score is inflated.
* This is a student/research project, **not a medical device or a diagnosis tool**.

## Running on Google Colab (free GPU)
1. Runtime → Change runtime type → **T4 GPU**.
2. Upload the project folder (or the zip) to Colab, then run:
```python
!unzip -q leukemia-osl.zip && cd leukemia-osl && pip install -q gdown
%cd leukemia-osl
!python download_data.py          # or mount Google Drive and copy the images into data/
!python -m src.dataset --data_dir data
!python train.py --data_dir data --mode holdout --epochs 30
```
