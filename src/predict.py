"""Inference on one MRI image: predict the tumor type and draw the predicted tumor mask.

Usage:
    python src/predict.py --image data/raw/mat/1.mat
    python src/predict.py --image my_slice.png --weights best.pt --out result.png

Input: a Figshare .mat file (its ground-truth mask is drawn too, in green) or any grayscale/RGB image file.
Weights: default weights/C_mtl_equal_seed0.pt (committed, fp16 copy of the C run, seed 0; same test results);
         any runs/<name>/best.pt from a multi-task run also works.
Output: prints class probabilities and saves <image>_pred.png (red = predicted tumor, green = ground truth).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import MEAN, STD
from model import build_model
from prepare_data import CLASS_NAMES, read_mat, to_uint8


def load_image(path):
    """Returns (uint8 grayscale image, ground-truth mask or None)."""
    if path.suffix == ".mat":
        image, mask, _, _ = read_mat(path)
        return to_uint8(image), mask
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Cannot read image: {path}")
    return image, None


@torch.no_grad()
def predict(model, image, size, device):
    """Same preprocessing as validation/test: resize, scale to [0,1], normalize."""
    x = cv2.resize(image, (size, size)).astype(np.float32) / 255.0
    x = torch.from_numpy((x - MEAN) / STD)[None, None].to(device)  # (1, 1, H, W)
    seg_logits, cls_logits = model(x)
    probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
    mask = (torch.sigmoid(seg_logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, image.shape[::-1], interpolation=cv2.INTER_NEAREST)  # back to original size
    return probs, mask


def draw(image, mask, gt=None):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    red = out.copy()
    red[mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(red, 0.35, out, 0.65, 0)
    if gt is not None:
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help=".mat from the Figshare dataset, or a .png/.jpg")
    ap.add_argument("--weights", default="weights/C_mtl_equal_seed0.pt")
    ap.add_argument("--out", default=None, help="output image (default: <image>_pred.png next to the input)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg["tasks"] != "both":
        raise SystemExit(f"{args.weights} was trained for '{cfg['tasks']}' only; use a multi-task (C or D) checkpoint")
    model = build_model(cfg["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    path = Path(args.image)
    image, gt = load_image(path)
    probs, mask = predict(model, image, cfg["size"], device)

    k = int(probs.argmax())
    print(f"Predicted tumor type: {CLASS_NAMES[k]} ({probs[k]:.1%})")
    for i, name in CLASS_NAMES.items():
        print(f"  {name:<11} {probs[i]:.3f}")
    print(f"Predicted tumor area: {int(mask.sum())} pixels")
    if gt is not None:
        inter = int((mask & (gt > 0)).sum())
        dice = 2 * inter / max(int(mask.sum()) + int((gt > 0).sum()), 1)
        print(f"Dice vs ground truth: {dice:.3f}")

    out = Path(args.out) if args.out else path.with_name(f"{path.stem}_pred.png")
    cv2.imwrite(str(out), draw(image, mask, gt))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
