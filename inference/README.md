# Inference — Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning)

Inference code of the DLMED project ([github.com/NDN-P/DLMED](https://github.com/NDN-P/DLMED)), written in the
layout of slide chapter 10 and the class Colab (`inference`): **`Net.py` + `model.pt` → `TestingCNN.py`**.

For one MRI slice the model gives the **tumor type** (meningioma / glioma / pituitary, with probabilities) and
the **tumor mask**.

## Files

| File | Content |
|---|---|
| `Net.py` | structure of the model (same file as in `../training/`) |
| `TestingCNN.py` | 1. load the network, 2. test with one image, 3. (optional) test with the whole test set |
| `inference.ipynb` | the same code as a Colab notebook (`Net` is included in a cell) |
| `model.pt` | trained weights (`state_dict`). Not committed: when it is missing, `../weights/C_mtl_equal_seed0.pt` (multi-task model C, seed 0, fp16, 49 MB) is used, or downloaded from the DLMED repo on the first run |
| `requirements.txt` | packages |

## How to use

Run from inside this folder.

```bash
pip install -r requirements.txt

python TestingCNN.py path/to/image
```

The image is a Figshare `.mat` file or any `.png` / `.jpg` MRI slice. Output format (the numbers below only
show the layout; they are not results):

```
Predicted: glioma 98.12%
  meningioma  0.015
  glioma      0.981
  pituitary   0.004
Predicted tumor area: 5321 pixels
Dice vs ground truth: 0.871
Saved 858_pred.png (red = predicted tumor, green = ground truth)
```

`<image>_pred.png` is saved in this folder: predicted tumor in red. For a `.mat` input the expert mask is drawn
in green and the Dice score is printed.

Colab: open `inference.ipynb`. It downloads the weights from the DLMED repo and asks for an image to upload.

### Whole test set (optional)

When the processed dataset exists at `../training/data/processed` (see `../training/README.md`), step 3 also
runs the 436 test images of `../training/splits.csv`, saves the predictions to `out.csv` and prints:

- classification: accuracy, precision / recall / F1 per class and macro, confusion matrix
- segmentation: Dice (mean ± std, and per tumor type) and IoU

Reference for `model.pt` on the test set (DLMED repo): accuracy 0.9518, macro-F1 0.9375, Dice 0.7666.

## Settings

Set at the top of `TestingCNN.py`:

| Setting | Default | Meaning |
|---|---|---|
| `model_path` | `./model.pt` | weights; a repo checkpoint `runs/<name>/best.pt` also works |
| `img_path` | first command-line argument | image to test |
| `annotations_file`, `img_dir` | `../training/splits.csv`, `../training/data/processed` | test set for step 3 |
| `image_size` | 256 | must be the size used in training |

Pre-processing is the same as validation / test in training: grayscale → resize 256 × 256 → [0, 1] →
normalize (mean 0.449, std 0.226). The mask is thresholded at 0.5 and resized back to the original image size.

To use a model you trained yourself, copy `../training/model.pt` to this folder as `model.pt`.
