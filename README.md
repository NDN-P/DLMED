# Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning)

One model with a **shared encoder** learns two tasks together from a T1-weighted contrast-enhanced MRI slice:

1. **Segmentation** of the tumor region (U-Net decoder, metric: Dice / IoU)
2. **Classification** of tumor type: meningioma / glioma / pituitary (metric: Precision / Recall / Macro-F1, confusion matrix)

Main question: *does learning where the tumor is help classify its type (and vice versa)?*

**Results: see [results/RESULTS.md](results/RESULTS.md).**

## Dataset

**Figshare Brain Tumor Dataset** (Jun Cheng): 3,064 slices from 233 patients, `.mat` (MATLAB v7.3) files containing
`image`, `tumorMask`, `label` (1 = meningioma, 2 = glioma, 3 = pituitary) and `PID` (patient ID).

- Download: <https://figshare.com/articles/dataset/brain_tumor_dataset/1512427> (4 zip files)
- Do **not** use the Kaggle "Brain Tumor MRI" re-uploads: they have no masks and no patient IDs.
- Citation: Cheng J. et al., ["Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition"](https://doi.org/10.1371/journal.pone.0140381), PLoS ONE 10(10), 2015.
- Acquisition sites: Nanfang Hospital and General Hospital, Tianjin Medical University (two hospitals).

Put all extracted `.mat` files under `data/raw/` (sub-folders are fine). The `data/` folder is git-ignored.
On Colab, keep data and runs in a shared Google Drive folder.

## Setup

**Colab:** use `notebooks/colab_train.ipynb`. Kaggle needs its own data paths; the Colab Drive-mount cell does not run there.

**Local machine with an NVIDIA GPU (Windows):**

Run from the `DLMED` folder in **PowerShell**:

```powershell
python -m venv "$env:USERPROFILE/.venvs/dlmed"
$dlmedPython = "$env:USERPROFILE/.venvs/dlmed/Scripts/python.exe"
& $dlmedPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $dlmedPython -m pip install -r requirements.txt
& $dlmedPython -c "import torch; print('GPU available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- On Guy's laptop the environment already exists: start with the `$dlmedPython = ...` line; no reinstall is needed.
- Use `& $dlmedPython` for every Python command below. In a new PowerShell window, set that variable again.
- `requirements-lock.txt` records the Windows / Python 3.14.7 environment verified in the 2026-10-10 review.
  To install those exact package versions instead of the portable minimums, use
  `& $dlmedPython -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`.
- RTX 50-series GPUs need the CUDA 12.8 build (`cu128`) as above; older GPUs also work with it.
- If DataLoader workers cause errors on Windows, add `--workers 0`.

## Usage

```powershell
# 1. Convert .mat -> PNG (splits/splits.csv is already committed and is kept as is)
& $dlmedPython src/prepare_data.py --raw_dir data/raw/mat --out_dir data/processed

# 2. Short trial in a separate folder (best epoch picked on validation)
& $dlmedPython src/train.py --tasks both --epochs 2 --name C_demo_seed0 --out_dir runs/demo

# 3. Inspect the trial on validation; reserve test for the final experiments
& $dlmedPython src/evaluate.py --run runs/demo/C_demo_seed0 --split val

# 4. Rebuild the published tables/figures from the original saved runs (Guy's laptop)
& $dlmedPython src/analyze.py --runs_dir runs --out_dir results

# 5. Regression checks
& $dlmedPython -m unittest discover -s tests -v
```

**Existing run folders are protected.** A repeated name exits without modifying that run. Choose another
`--name` or a new `--out_dir`; there is no automatic overwrite or resume.

To repeat the full protocol, install **Git for Windows** for `bash`, and use new folders:

```powershell
# Let the Bash scripts find the same Python environment
$env:PATH = "$(Split-Path -Parent $dlmedPython);$env:PATH"
bash scripts/tune_lr.sh --out_dir runs/reproduced/tuning --workers 4
# Choose lr from your validation results first; 3e-4 was the original choice
bash scripts/run_experiments.sh --out_dir runs/reproduced/final --lr 3e-4 --workers 4
& $dlmedPython src/analyze.py --runs_dir runs/reproduced/final --out_dir runs/reproduced/results
```

Code, summary tables and figures are on GitHub. Data and the 12 trained checkpoints are git-ignored and stay on Guy's
laptop; teammates who want to run evaluation need those files or must download the dataset and train anew.
The main model's weights **are** in the repo (see Inference below).

## Inference (one image)

```powershell
& $dlmedPython src/predict.py --image data/raw/mat/858.mat     # or any .png/.jpg MRI slice
```

Prints the tumor type with class probabilities and saves `<image>_pred.png`: predicted tumor in red, ground
truth in green (shown and scored with Dice only for Figshare `.mat` input). Example:
[`results/figures/8_inference_example.png`](results/figures/8_inference_example.png).

The default weights `weights/C_mtl_equal_seed0.pt` (49 MB) are the multi-task C model, seed 0, stored in fp16.
On the test set they give the same results as the original fp32 checkpoint (accuracy 0.9518, Macro-F1 0.9375,
Dice 0.7666). Preprocessing matches validation/test: grayscale → resize 256 → /255 → normalize (0.449, 0.226).

## What to submit (course requirement)

| Required | File(s) |
|---|---|
| Architecture code | [`src/model.py`](src/model.py) |
| Inference code | [`src/predict.py`](src/predict.py) (one image), [`src/evaluate.py`](src/evaluate.py) (whole test set) |
| Model weights | [`weights/C_mtl_equal_seed0.pt`](weights/) |
| Results | [`results/`](results/): `RESULTS.md`, tables, figures |

Training code is not required, but it stays in the repo so the experiments can be reproduced.

## Data split (separate patient groups)

Slices from the same patient are near-duplicates, so the split is done **by patient**
(`StratifiedGroupKFold`, grouped by the conservative `patient` column, stratified by tumor type):
about 70 / 15 / 15 % train / val / test. No recorded ID or conservative group crosses the committed splits.

The dataset reports 233 patients and contains 233 distinct PID strings. Eighteen groups of glioma IDs have
letter-suffixed variants (e.g. `MR040240`, `MR040240B`, `MR040240C`, `MR040240D`). They may be related scans;
their shared identity has **not** been confirmed by the dataset authors. Fourteen such groups crossed the initial
raw-ID split. We conservatively keep each group together to reduce possible leakage:
**209 groups → train 151 / val 29 / test 29 (2,194 / 434 / 436 images)**.
This is a grouping rule, not a corrected count of real patients. The `patients` columns in
`splits/data_summary.csv` count these groups; the CSV schema is kept for compatibility.
**Everyone uses the committed `splits/splits.csv`.** The test set is only used after all settings are chosen on validation.

## Experiments

All experiments use the same model (`src/model.py`: U-Net, ResNet-34 ImageNet encoder, classification head on the
deepest encoder features), the same data, augmentation policy, epochs and seed labels. Single-task baselines switch the other loss off.

| ID | Command | Purpose |
|---|---|---|
| A | `--tasks cls` | classification only |
| B | `--tasks seg` | segmentation only |
| C | `--tasks both` | multi-task, equal loss weights |
| D | `--tasks both --weighting uncertainty` | multi-task, learned loss weights (Kendall et al., 2018) |

Fixed weights can also be tried, e.g. `--tasks both --w_seg 1 --w_cls 0.5`.

Settings (lr chosen on validation from 1e-4 / 3e-4 / 1e-3, see `results/tuning_lr_validation.csv`): image 256x256, batch 16, AdamW lr 3e-4, weight decay 1e-4, cosine schedule, 30 epochs, mixed precision on GPU.
Segmentation loss = BCE + Dice; classification loss = cross-entropy.
Augmentation (train only, applied to image and mask together): horizontal flip, affine (scale ±10 %, shift ±5 %, rotate ±15°), brightness/contrast.

**Original results vs corrected training:** the published 12 runs were trained before augmentation seeding was fixed.
Their checkpoints and scores were verified on 2026-10-10 and are preserved. Their seed labels do **not** guarantee
identical augmentation draws or an exact retraining replay; per-seed differences are descriptive comparisons.
New runs seed Albumentations, each worker and the training sampler, and save `environment.txt`.
Keep software versions, worker count and hardware fixed when checking repeatability; bitwise equality across
GPUs or library versions is not guaranteed. New runs must be reported separately from the original results.

## Outputs (per run, in `runs/<name>/`)

| File | Content |
|---|---|
| `config.json` | all settings of the run |
| `environment.txt` | Python / CUDA version and installed package versions (new runs only) |
| `history.csv` | train losses and validation metrics per epoch |
| `best.pt` | checkpoint of the best validation epoch |
| `test_metrics.json` | test metrics for the trained task(s), incl. per-class F1 and per-tumor-type Dice |
| `test_predictions.csv` | per-image prediction, class probabilities, Dice, IoU (for error analysis) |
| `confusion_matrix.png`, `misclassified.png`, `seg_best_worst.png` | figures for the slides |

## Project structure

```
src/prepare_data.py   .mat -> PNG, metadata, patient-level split
src/dataset.py        PyTorch dataset + augmentation
src/model.py          shared-encoder multi-task model
src/losses.py         task losses and multi-task weighting
src/metrics.py        Dice / IoU / classification metrics
src/engine.py         inference loop shared by training and evaluation
src/train.py          training (one experiment)
src/evaluate.py       test-set evaluation and figures
src/predict.py        inference on one image (tumor type + mask overlay)
weights/              main model weights (C, seed 0, fp16)
src/summarize.py      comparison table across runs and seeds
src/analyze.py        final tables + slide figures -> results/
scripts/tune_lr.sh    learning-rate choice on validation
scripts/run_experiments.sh   A-D x 3 seeds, train + test evaluation
notebooks/colab_train.ipynb
splits/               shared split files (committed)
results/              final results: RESULTS.md, tables, figures (committed)
```

## Team

Who reviews what, slide order and Q&A preparation: see [TASKS.md](TASKS.md).

Topic 2 classification materials: [classification/README.md](classification/README.md) — method notes, reproducible A/C/D result tables, slide 9/12 copy, Thai speaking notes and Q&A. Regenerate tables with `python classification/report.py` (standard library only; uses the recorded result CSVs).

