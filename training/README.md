# Training — Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning)

Training code of the DLMED project ([github.com/NDN-P/DLMED](https://github.com/NDN-P/DLMED)), written in the
layout of slide chapter 10 and the class Colab (`Training`): **`Net.py` + `TrainingCNN.py` → `model.pt`**.

One model with a shared encoder learns two tasks together from one T1-weighted contrast-enhanced MRI slice:

1. **Segmentation** of the tumor region (U-Net decoder; Dice / IoU)
2. **Classification** of the tumor type: meningioma / glioma / pituitary (macro-F1, confusion matrix)

## Files

| File | Content |
|---|---|
| `Net.py` | structure of the model: ResNet-34 encoder (ImageNet) + U-Net decoder + classification head |
| `TrainingCNN.py` | training script, steps 1–5 below |
| `Training.ipynb` | the same code as a Colab notebook (`Net` is included in a cell) |
| `prepare_data.py` | converts the Figshare `.mat` files to PNG images and masks |
| `splits.csv` | shared train / val / test split **by patient** (2,194 / 434 / 436 images), from the DLMED repo |
| `requirements.txt` | packages |

Steps in `TrainingCNN.py` (same order as the class Colab):

1. Custom `Dataset` (image, mask, label) + `DataLoader`s; augmentation with `torchvision.transforms.v2`
2. Define the network (`Net`)
3. Define the loss function and optimizer
4. Train the network, evaluate on the validation set every epoch
5. Save the model of the best validation epoch (`model.pt`)

## Data source

**Figshare Brain Tumor Dataset** (Jun Cheng): 3,064 slices from 233 patients, `.mat` files with `image`,
`tumorMask`, `label` (1 = meningioma, 2 = glioma, 3 = pituitary) and `PID` (patient ID).

- Download: <https://figshare.com/articles/dataset/brain_tumor_dataset/1512427> (4 zip files)
- Citation: Cheng J. et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation
  and Partition", PLoS ONE 10(10), 2015.

Extract all `.mat` files under `training/data/raw/` (sub-folders are fine).

## How to use

Run everything from inside this folder.

```bash
pip install -r requirements.txt

# 1. Convert .mat -> PNG (splits.csv is kept as is)
python prepare_data.py --raw_dir data/raw --out_dir data/processed

# 2. Train (30 epochs; a GPU is recommended)
python TrainingCNN.py
```

Colab: open `Training.ipynb`, set Runtime → GPU, and put `processed.zip` (the zipped `data/processed` folder)
in the Google Drive folder named in the "Google Colab only" cell.

Outputs, written next to the script:

| File | Content |
|---|---|
| `model.pt` | weights (`state_dict`) of the best validation epoch |
| `history.csv` | train loss and validation metrics per epoch |
| `training_curves.png` | loss and validation macro-F1 / Dice curves |
| `sample_batch.png` | one augmented training batch with the tumor mask in red |

To test the trained model, copy `model.pt` to `../inference/` and use `TestingCNN.py` there.

## Settings

Set at the top of `TrainingCNN.py`. Defaults are the main model of the project (experiment C):

| Setting | Value |
|---|---|
| Image size / batch size | 256 × 256 / 16 |
| Optimizer | AdamW, lr 3e-4 (chosen on validation from 1e-4 / 3e-4 / 1e-3), weight decay 1e-4 |
| Schedule / epochs | cosine / 30 |
| Segmentation loss | Binary Cross Entropy + Dice |
| Classification loss | Cross Entropy |
| Multi-task loss | `w_seg * L_seg + w_cls * L_cls` |
| Augmentation (train only) | horizontal flip, affine (rotate ±15°, shift ±5 %, scale ±10 %), brightness / contrast |
| Normalization | grayscale, mean 0.449, std 0.226 |
| Best epoch | highest validation (macro-F1 + Dice) / 2 |

The four experiments of the project use the same code with different loss weights:

| ID | Setting | Purpose |
|---|---|---|
| A | `w_seg, w_cls = 0.0, 1.0` | classification only |
| B | `w_seg, w_cls = 1.0, 0.0` | segmentation only |
| C | `w_seg, w_cls = 1.0, 1.0` | multi-task, equal loss weights (main model) |
| D | `weighting = 'uncertainty'` | multi-task, learned loss weights (Kendall et al., 2018) |

## Notes

- `Net` has the same layers and layer names as the repo's model (`segmentation_models_pytorch` U-Net with a
  ResNet-34 encoder and classification head), so weights are interchangeable. Here it is written with plain
  `torch.nn` and `torchvision.models`, as in the slides.
- Augmentation uses `torchvision.transforms.v2` instead of Albumentations (same policy), so a new run will not
  reproduce the published runs number for number.
- The test set is not used in training. The best epoch is picked on the validation set only.
