# %% [markdown]
# # Training - Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning)
#
# TrainingCNN: the file that calls `Net` and trains the model (slide: Training Phase -> `model.pt`).
#
# One model with a **shared encoder** learns two tasks together from a T1-weighted contrast-enhanced MRI slice:
# **segmentation** of the tumor (mask) and **classification** of the tumor type (meningioma / glioma / pituitary).
#
# Steps (same order as the class Colab and slide chapter 10):
#
# 1. Creating a custom Dataset and DataLoaders (training + validation set)
# 2. Define a Convolutional Neural Network (`Net`)
# 3. Define a loss function and optimizer
# 4. Train the network, evaluate on the validation set every epoch
# 5. Save the model (`model.pt`) - the best validation epoch
#
# Run from inside this folder: `python TrainingCNN.py`

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

# %% [markdown]
# ### Settings
# Data comes from `prepare_data.py` (Figshare `.mat` -> PNG). `splits.csv` is the shared patient-level split.
# The values below are the settings of the main model in the DLMED repo (experiment C: multi-task, equal loss weights).

# %%
img_dir = './data/processed'        # contains images/<id>.png and masks/<id>.png
annotations_file = './splits.csv'   # columns: id, label, pid, patient, split (train / val / test)
model_path = './model.pt'           # output: trained weights

image_size = 256
batch_size = 16
num_epochs = 30
learning_rate = 3e-4                # chosen on validation from 1e-4 / 3e-4 / 1e-3
weight_decay = 1e-4
seed = 0

# Loss weights of the two tasks: total loss = w_seg * L_seg + w_cls * L_cls
#   A classification only : w_seg = 0, w_cls = 1
#   B segmentation only   : w_seg = 1, w_cls = 0
#   C multi-task (equal)  : w_seg = 1, w_cls = 1   <- main model
#   D multi-task (learned): weighting = 'uncertainty' (Kendall et al., CVPR 2018)
w_seg, w_cls = 1.0, 1.0
weighting = 'fixed'                 # 'fixed' or 'uncertainty'

classes = ('meningioma', 'glioma', 'pituitary')   # label 0 / 1 / 2
# Grayscale average of the ImageNet mean / std (the encoder is ImageNet-pretrained)
MEAN, STD = 0.449, 0.226

num_workers = 2 if sys.platform.startswith('linux') else 0   # Colab / Linux: 2 loader processes
in_notebook = 'ipykernel' in sys.modules

torch.manual_seed(seed)
np.random.seed(seed)

# %% [markdown]
# - Google Colab only: mount Google Drive and copy the dataset to the Colab disk (reading thousands of small
#   files from Drive is slow). `processed.zip` = the zipped `data/processed` folder made by `prepare_data.py`.

# %%
if 'google.colab' in sys.modules:
    import shutil
    import urllib.request
    from google.colab import drive

    drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/brain_tumor'   # <- change to your shared folder
    if not os.path.isdir(img_dir):
        shutil.unpack_archive(f'{DRIVE}/processed.zip', './data')   # -> ./data/processed/images, masks
    if not os.path.exists(annotations_file):
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/NDN-P/DLMED/main/splits/splits.csv', annotations_file)

# %% [markdown]
# ## 1. Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: `__init__`, `__len__` and `__getitem__`.
# The MRI images are stored in `img_dir/images`, the tumor masks in `img_dir/masks`, and the labels
# (tumor type) together with the train / val / test split in the CSV file `annotations_file`.
#
# Each sample is **(image, mask, label)** because the model has two tasks. Image and mask are wrapped as
# `tv_tensors.Image` / `tv_tensors.Mask`, so every geometric transform is applied to both in the same way.

# %%
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, split, transform=None, target_transform=None):
        img_labels = pd.read_csv(annotations_file, dtype={'id': str})
        self.img_labels = img_labels[img_labels['split'] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_labels.loc[idx, 'id']
        label = int(self.img_labels.loc[idx, 'label'])

        # Load image and mask (grayscale, uint8)
        image = Image.open(os.path.join(self.img_dir, 'images', f'{img_id}.png')).convert('L')
        mask = Image.open(os.path.join(self.img_dir, 'masks', f'{img_id}.png')).convert('L')
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        if self.target_transform:
            label = self.target_transform(label)

        image = image.as_subclass(torch.Tensor)                  # (1, H, W) float32, normalized
        mask = (mask.as_subclass(torch.Tensor) > 127).float()    # (1, H, W) 0 = background, 1 = tumor
        return image, mask, label

# %% [markdown]
# - Data transformation and augmentation (`torchvision.transforms.v2`)
#
# Train only: horizontal flip, affine (rotate ±15°, shift ±5 %, scale ±10 %), brightness / contrast.
# Validation / test: resize and normalize only.

# %%
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size), antialias=False),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomAffine(
        degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.BILINEAR)], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
    transforms.ToDtype(torch.float32, scale=True),        # image -> float32 in [0, 1] (mask unchanged)
    transforms.Normalize(mean=(MEAN,), std=(STD,))])

transform = transforms.Compose([
    transforms.Resize((image_size, image_size), antialias=False),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=(MEAN,), std=(STD,))])

# %% [markdown]
# - Preparing your data for training with DataLoaders (training and validating set)
#
# The split is **by patient** (slices of one patient are near-duplicates), so no patient is in two sets.

# %%
train_dataset = CustomImageDataset(annotations_file, img_dir, split='train', transform=train_transform)
val_dataset = CustomImageDataset(annotations_file, img_dir, split='val', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(f'train {len(train_dataset)} images, val {len(val_dataset)} images')

# %% [markdown]
# - Show sample images from the DataLoader (tumor mask in red)

# %%
def imshow(img, path=None):
    npimg = img.numpy()
    plt.figure(figsize=(12, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    if path:
        plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show() if in_notebook else plt.close()


# get some random training images
dataiter = iter(train_dataloader)
images, masks, labels = next(dataiter)

n_show = min(4, len(images))
gray = (images[:n_show] * STD + MEAN).clamp(0, 1)                                      # unnormalize
tumor = masks[:n_show] * 0.35
overlay = torch.cat([gray * (1 - tumor) + tumor, gray * (1 - tumor), gray * (1 - tumor)], dim=1)  # red where mask = 1
imshow(torchvision.utils.make_grid(overlay), path='sample_batch.png')
# print labels
print(' '.join(f'{classes[labels[j]]:10s}' for j in range(n_show)))

# %% [markdown]
# ## 2. Define a Convolutional Neural Network
# `Net` (file `Net.py`): ResNet-34 encoder shared by a U-Net decoder (segmentation) and a classification head.

# %%
from Net import Net

# ImageNet-pretrained encoder
model = Net(num_classes=len(classes), pretrained=True)
print(model)

# %% [markdown]
# ## 3. Define a Loss function and optimizer
# - Segmentation loss = Binary Cross Entropy + Dice loss
# - Classification loss = Cross Entropy
# - Multi-task loss = weighted sum of the two

# %%
def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


class MultiTaskLoss(nn.Module):
    """weighting='fixed':       L = w_seg * L_seg + w_cls * L_cls
    weighting='uncertainty': L = sum_i exp(-s_i) * L_i + s_i, with learnable s_i (Kendall et al., 2018)"""

    def __init__(self, weighting='fixed', w_seg=1.0, w_cls=1.0):
        super().__init__()
        self.weighting = weighting
        self.w_seg, self.w_cls = w_seg, w_cls
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        if weighting == 'uncertainty':
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, seg_logits, cls_logits, masks, labels):
        loss_seg = self.bce(seg_logits, masks) + dice_loss(seg_logits, masks)
        loss_cls = self.ce(cls_logits, labels)
        if self.weighting == 'uncertainty':
            s = self.log_vars
            loss = torch.exp(-s[0]) * loss_seg + s[0] + torch.exp(-s[1]) * loss_cls + s[1]
        else:
            loss = self.w_seg * loss_seg + self.w_cls * loss_cls
        return loss, loss_seg, loss_cls


criterion = MultiTaskLoss(weighting, w_seg, w_cls)
optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()),
                        lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# %% [markdown]
# - Evaluating performance (validation set): macro-F1 for classification, Dice / IoU for segmentation.
#   `model.eval()` + `torch.no_grad()`: the model must not learn from this data.

# %%
def dice_iou_per_image(seg_logits, masks, threshold=0.5):
    pred = (torch.sigmoid(seg_logits) > threshold).float()
    inter = (pred * masks).sum(dim=(1, 2, 3))
    p, t = pred.sum(dim=(1, 2, 3)), masks.sum(dim=(1, 2, 3))
    empty = (p + t) == 0   # prediction and ground truth both empty counts as perfect
    dice = torch.where(empty, torch.ones_like(inter), 2 * inter / (p + t).clamp(min=1))
    iou = torch.where(empty, torch.ones_like(inter), inter / (p + t - inter).clamp(min=1))
    return dice.cpu(), iou.cpu()


def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, dices, ious, losses = [], [], [], [], []
    with torch.no_grad():
        for inputs, masks, labels in dataloader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            seg_logits, cls_logits = model(inputs)
            loss, _, _ = criterion(seg_logits, cls_logits, masks, labels)
            _, predicted = torch.max(cls_logits, 1)
            dice, iou = dice_iou_per_image(seg_logits, masks)
            losses.append(loss.item())
            y_true += labels.tolist()
            y_pred += predicted.tolist()
            dices.append(dice)
            ious.append(iou)
    return {
        'loss': float(np.mean(losses)),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'dice': torch.cat(dices).mean().item(),
        'iou': torch.cat(ious).mean().item(),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
    }


def selection_score(val):
    """Validation score used to pick the best epoch (only the trained task(s) count)."""
    if weighting == 'fixed' and w_seg == 0:
        return val['macro_f1']
    if weighting == 'fixed' and w_cls == 0:
        return val['dice']
    return (val['macro_f1'] + val['dice']) / 2

# %% [markdown]
# ## 4. Train the network

# %%
# Check if GPU (cuda) is available (mps = Apple Silicon GPU)
device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('device:', device)

# Move our model to device (GPU or CPU)
model.to(device)
criterion.to(device)

use_amp = device.type == 'cuda'   # mixed precision on GPU: faster, less memory
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

history = []
best_score = -1.0

for epoch in range(num_epochs):  # loop over the dataset multiple times
    model.train()
    running_loss, running_seg, running_cls = 0.0, 0.0, 0.0

    for batch_idx, (inputs, masks, labels) in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, masks, labels]
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
            seg_logits, cls_logits = model(inputs)
        loss, loss_seg, loss_cls = criterion(seg_logits.float(), cls_logits.float(), masks, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # statistics
        running_loss += loss.item()
        running_seg += loss_seg.item()
        running_cls += loss_cls.item()
    scheduler.step()

    # evaluate on the validation set
    n_batches = len(train_dataloader)
    val = evaluate(model, val_dataloader, device)
    score = selection_score(val)
    history.append({
        'epoch': epoch + 1,
        'train_loss': running_loss / n_batches,
        'train_loss_seg': running_seg / n_batches,
        'train_loss_cls': running_cls / n_batches,
        'val_loss': val['loss'], 'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1'],
        'val_dice': val['dice'], 'val_iou': val['iou'], 'val_score': score,
    })
    pd.DataFrame(history).to_csv('history.csv', index=False)

    # Is it accepted? keep the model of the best validation epoch
    improved = score > best_score
    if improved:
        best_score = score
        # Save model
        torch.save(model.state_dict(), model_path)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / n_batches:.4f}, '
          f'Val Loss: {val["loss"]:.4f}, Val F1: {val["macro_f1"]:.4f}, Val Dice: {val["dice"]:.4f}'
          f'{"  * saved" if improved else ""}')

print('Finished Training')

# %% [markdown]
# - Training curves

# %%
history = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history['epoch'], history['train_loss'], label='train loss')
axes[0].plot(history['epoch'], history['val_loss'], label='validation loss')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].legend()
axes[1].plot(history['epoch'], history['val_macro_f1'], label='validation macro-F1 (classification)')
axes[1].plot(history['epoch'], history['val_dice'], label='validation Dice (segmentation)')
axes[1].set_xlabel('epoch')
axes[1].legend()
fig.tight_layout()
fig.savefig('training_curves.png', dpi=120)
plt.show() if in_notebook else plt.close(fig)

# %% [markdown]
# - The saved model (`model.pt` = best validation epoch): load it back and check it on the validation set.
#   The test set is not used here; test it with `inference/TestingCNN.py` after all settings are chosen.

# %%
model.load_state_dict(torch.load(model_path, map_location=device))
val = evaluate(model, val_dataloader, device)
print(f'Best model on validation: accuracy {val["accuracy"]:.4f}, macro-F1 {val["macro_f1"]:.4f}, '
      f'Dice {val["dice"]:.4f}, IoU {val["iou"]:.4f}')
print('Confusion matrix (rows = true, columns = predicted):', classes)
print(val['confusion_matrix'])
print(f'Saved {model_path}')
