# %% [markdown]
# # Inference - Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning)
#
# TestingCNN: the file that calls `Net` and tests the trained model (slide: Testing Phase, uses `model.pt`).
#
# 1. Load a Convolutional Neural Network (`Net` + `model.pt`)
# 2. Test the network with an image: tumor type (with probabilities) + predicted tumor mask
# 3. (optional) Test the network with the whole test set and measure the performance
#
# Run from inside this folder: `python TestingCNN.py path/to/image`
# (a Figshare `.mat` file, or any `.png` / `.jpg` MRI slice)

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image

# %% [markdown]
# ### Settings

# %%
model_path = './model.pt'                      # trained weights (multi-task model C, seed 0)
img_path = '../training/data/raw/1.mat'        # image to test: .mat (Figshare) or .png / .jpg

# Optional test-set evaluation (step 3): runs only when this data folder exists
annotations_file = '../training/splits.csv'    # columns: id, label, ..., split
img_dir = '../training/data/processed'         # images/<id>.png and masks/<id>.png (from prepare_data.py)

image_size = 256
MEAN, STD = 0.449, 0.226                       # same normalization as in training

in_notebook = 'ipykernel' in sys.modules
if len(sys.argv) > 1 and not in_notebook:
    img_path = sys.argv[1]                     # python TestingCNN.py path/to/image
if not os.path.exists(model_path) and os.path.exists('../weights/C_mtl_equal_seed0.pt'):
    model_path = '../weights/C_mtl_equal_seed0.pt'   # inside the DLMED repo: the committed weights (same model)

# %% [markdown]
# - Google Colab only: get the trained weights from the DLMED repo and upload the image to test.

# %%
if 'google.colab' in sys.modules:
    import urllib.request
    from google.colab import files

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            'https://github.com/NDN-P/DLMED/raw/main/weights/C_mtl_equal_seed0.pt', model_path)
    img_path = next(iter(files.upload()))      # choose an MRI slice (.mat / .png / .jpg)

# %% [markdown]
# ## 1. Load a Convolutional Neural Network

# %%
from Net import Net, classes

model = Net(num_classes=len(classes), pretrained=False)
state_dict = torch.load(model_path, map_location='cpu')
state_dict = state_dict.get('model', state_dict)   # a DLMED-repo checkpoint (runs/<name>/best.pt) also works
model.load_state_dict(state_dict)

# %% [markdown]
# ## 2. Test the network with an image
# Same pre-processing as the validation / test set in training: grayscale -> resize 256 -> [0, 1] -> normalize.

# %%
transform = transforms.Compose([
    transforms.ToImage(),                                  # Convert to image format
    transforms.Resize((image_size, image_size), antialias=False),
    transforms.ToDtype(torch.float32, scale=True),         # Convert image dtype, range [0, 1]
    transforms.Normalize(mean=(MEAN,), std=(STD,))])

# %%
def load_image(path):
    """Returns (grayscale uint8 image, ground-truth mask or None). Only Figshare .mat files have a mask."""
    if path.endswith('.mat'):
        import h5py
        with h5py.File(path, 'r') as f:
            # MATLAB stores column-major, so h5py returns transposed arrays
            image = np.array(f['cjdata']['image'], dtype=np.float32).T
            gt = np.array(f['cjdata']['tumorMask'], dtype=np.uint8).T
        lo, hi = image.min(), image.max()
        return ((image - lo) / max(hi - lo, 1e-8) * 255).astype(np.uint8), gt
    return np.array(Image.open(path).convert('L')), None


def draw_overlay(image, mask, gt=None):
    """Grayscale image with the predicted tumor in red and the ground-truth contour in green."""
    out = np.stack([image] * 3, axis=-1).astype(np.float32)
    out[mask > 0] = 0.65 * out[mask > 0] + 0.35 * np.array([255, 0, 0])
    if gt is not None:
        g = np.pad(gt > 0, 1)
        inside = g[1:-1, 1:-1] & g[:-2, 1:-1] & g[2:, 1:-1] & g[1:-1, :-2] & g[1:-1, 2:]
        e = np.pad((gt > 0) & ~inside, 1)   # border pixels of the mask, made 2-3 px thick
        out[e[1:-1, 1:-1] | e[:-2, 1:-1] | e[2:, 1:-1] | e[1:-1, :-2] | e[1:-1, 2:]] = (0, 255, 0)
    return out.astype(np.uint8)


if not os.path.exists(img_path):
    raise SystemExit(f'Image not found: {img_path}\nUsage: python TestingCNN.py path/to/image (.mat / .png / .jpg)')

original, gt = load_image(img_path)
image = transform(Image.fromarray(original))

# Check if GPU (cuda) is available (mps = Apple Silicon GPU)
device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Move our model to device (GPU or CPU)
model.to(device)
model.eval()

image = image.to(device)
image = image.unsqueeze(0)

with torch.no_grad():
    seg_logits, cls_logits = model(image)

# Task 2: classification
outputs = cls_logits.squeeze(0)
score = torch.softmax(outputs, 0)
_, predicted = torch.max(score, 0)

print(f'Predicted: {classes[predicted]} {score[predicted] * 100:.02f}%')
for i, name in enumerate(classes):
    print(f'  {name:<11} {score[i]:.3f}')

# Task 1: segmentation (threshold 0.5, then back to the original image size)
mask = (torch.sigmoid(seg_logits) > 0.5).float()
mask = F.interpolate(mask, size=original.shape, mode='nearest')[0, 0].cpu().numpy().astype(np.uint8)
print(f'Predicted tumor area: {int(mask.sum())} pixels')
if gt is not None:
    inter = int((mask & (gt > 0)).sum())
    print(f'Dice vs ground truth: {2 * inter / max(int(mask.sum()) + int((gt > 0).sum()), 1):.3f}')

out_path = os.path.splitext(os.path.basename(img_path))[0] + '_pred.png'
overlay = draw_overlay(original, mask, gt)
Image.fromarray(overlay).save(out_path)
print(f'Saved {out_path} (red = predicted tumor' + (', green = ground truth)' if gt is not None else ')'))

if in_notebook:
    plt.imshow(overlay)
    plt.title(f'{classes[predicted]} {score[predicted] * 100:.02f}%')
    plt.axis('off')
    plt.show()

# %% [markdown]
# ## 3. Test the network with the test set (optional)
# Runs only when the processed dataset is available. `model.eval()` and `torch.no_grad()`: the model does not
# learn from the test data. Predictions are saved to `out.csv`; metrics follow the course guideline
# (classification: Precision / Recall / F1 / confusion matrix, segmentation: Dice / IoU).

# %%
if os.path.isdir(os.path.join(img_dir, 'images')):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from torchvision import tv_tensors

    test = pd.read_csv(annotations_file, dtype={'id': str})
    test = test[test['split'] == 'test'].reset_index(drop=True)

    rows = []
    with torch.no_grad():
        for img_id, label in zip(test['id'], test['label']):
            # loading the test image and its ground-truth mask
            image = Image.open(os.path.join(img_dir, 'images', f'{img_id}.png')).convert('L')
            gt = Image.open(os.path.join(img_dir, 'masks', f'{img_id}.png')).convert('L')
            image, gt = transform(tv_tensors.Image(image), tv_tensors.Mask(gt))
            gt = (gt.as_subclass(torch.Tensor)[0] > 127).to(device)

            # generating predictions
            seg_logits, cls_logits = model(image.as_subclass(torch.Tensor).to(device).unsqueeze(0))
            score = torch.softmax(cls_logits.squeeze(0), 0).cpu()
            pred = torch.sigmoid(seg_logits)[0, 0] > 0.5

            inter, total = (pred & gt).sum().item(), pred.sum().item() + gt.sum().item()
            rows.append({
                'id': img_id, 'label': int(label), 'pred': int(score.argmax()),
                **{f'prob_{name}': score[k].item() for k, name in enumerate(classes)},
                'dice': 2 * inter / total if total else 1.0,
                'iou': inter / (total - inter) if total else 1.0,
            })

    # saving the file
    out = pd.DataFrame(rows)
    out.to_csv('out.csv', index=False)

    precision, recall, f1, _ = precision_recall_fscore_support(
        out['label'], out['pred'], labels=list(range(len(classes))), zero_division=0)
    print(f'\nTest set: {len(out)} images')
    print(f'[classification] accuracy {accuracy_score(out["label"], out["pred"]):.4f}  '
          f'macro precision {precision.mean():.4f}  macro recall {recall.mean():.4f}  macro F1 {f1.mean():.4f}')
    for k, name in enumerate(classes):
        print(f'  {name:<11} precision {precision[k]:.4f}  recall {recall[k]:.4f}  F1 {f1[k]:.4f}')
    print('Confusion matrix (rows = true, columns = predicted):')
    print(confusion_matrix(out['label'], out['pred'], labels=list(range(len(classes)))))
    print(f'[segmentation]   Dice {out["dice"].mean():.4f} (+/- {out["dice"].std(ddof=0):.4f})  IoU {out["iou"].mean():.4f}')
    for k, name in enumerate(classes):
        print(f'  {name:<11} Dice {out.loc[out["label"] == k, "dice"].mean():.4f}')
    print('Saved out.csv')
