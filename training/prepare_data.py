"""Convert the Figshare brain tumor dataset (.mat, MATLAB v7.3) to PNG for TrainingCNN.py.

Usage (run from inside this folder):
    python prepare_data.py --raw_dir data/raw --out_dir data/processed

Outputs:
    <out_dir>/images/<id>.png   grayscale image, min-max scaled to uint8, original size
    <out_dir>/masks/<id>.png    tumor mask, 0 / 255
    <out_dir>/metadata.csv      id, label, class_name, pid, patient, height, width, tumor_area

The train / val / test split is the committed `splits.csv` (by patient, from the DLMED repo) and is kept as is.
It is only created here when the file does not exist (or with --overwrite_split).
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = {0: 'meningioma', 1: 'glioma', 2: 'pituitary'}  # dataset labels 1/2/3 -> 0/1/2


def read_mat(path):
    with h5py.File(path, 'r') as f:
        d = f['cjdata']
        # MATLAB stores column-major, so h5py returns transposed arrays
        image = np.array(d['image'], dtype=np.float32).T
        mask = np.array(d['tumorMask'], dtype=np.uint8).T
        label = int(np.array(d['label']).squeeze()) - 1
        pid = ''.join(chr(int(c)) for c in np.array(d['PID']).flatten()).strip('\x00 ')
    return image, mask, label, pid


def to_uint8(image):
    lo, hi = image.min(), image.max()
    return ((image - lo) / max(hi - lo, 1e-8) * 255).astype(np.uint8)


def patient_group(pid):
    """Some glioma IDs come with letter suffixes (MR040240, MR040240B, ...). Keep each group together."""
    return re.sub(r'^(MR\d+)[A-Z]$', r'\1', pid)


def split_by_patient(meta, test_frac, val_frac, seed):
    """Stratified (by tumor type) and grouped (by patient) train / val / test split."""
    from sklearn.model_selection import StratifiedGroupKFold

    meta = meta.copy()
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(meta, meta['label'].values, meta['patient'].values))
    meta['split'] = 'train'
    meta.loc[meta.index[test_idx], 'split'] = 'test'

    tv = meta.iloc[trainval_idx]
    sgkf = StratifiedGroupKFold(n_splits=round((1 - test_frac) / val_frac), shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(tv, tv['label'].values, tv['patient'].values))
    meta.loc[tv.index[val_idx], 'split'] = 'val'

    # No patient group may appear in more than one split.
    groups = {s: set(meta.loc[meta.split == s, 'patient']) for s in ('train', 'val', 'test')}
    assert not (groups['train'] & groups['val']), 'patient leak train/val'
    assert not (groups['train'] & groups['test']), 'patient leak train/test'
    assert not (groups['val'] & groups['test']), 'patient leak val/test'
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default='data/raw', help='folder with the extracted 1.mat ... 3064.mat')
    ap.add_argument('--out_dir', default='data/processed')
    ap.add_argument('--splits', default='splits.csv')
    ap.add_argument('--test_frac', type=float, default=0.15)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--overwrite_split', action='store_true', help='remake the split file if it exists')
    args = ap.parse_args()

    raw, out, split_csv = Path(args.raw_dir), Path(args.out_dir), Path(args.splits)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)

    files = sorted((p for p in raw.rglob('*.mat') if p.stem.isdigit()), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f'No numbered .mat files found under {raw}')

    rows, seen = [], set()
    for i, path in enumerate(files, 1):
        if path.stem in seen:  # same file extracted twice
            continue
        seen.add(path.stem)
        image, mask, label, pid = read_mat(path)
        Image.fromarray(to_uint8(image)).save(out / 'images' / f'{path.stem}.png')
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(out / 'masks' / f'{path.stem}.png')
        rows.append({
            'id': path.stem, 'label': label, 'class_name': CLASS_NAMES[label], 'pid': pid,
            'height': image.shape[0], 'width': image.shape[1], 'tumor_area': int((mask > 0).sum()),
        })
        if i % 500 == 0:
            print(f'  {i}/{len(files)}')

    meta = pd.DataFrame(rows)
    meta.insert(4, 'patient', meta.pid.map(patient_group))
    meta.to_csv(out / 'metadata.csv', index=False)
    print(f'Converted {len(meta)} images from {meta.pid.nunique()} patient IDs '
          f'({meta.patient.nunique()} patient groups) -> {out}')

    if split_csv.exists() and not args.overwrite_split:
        print(f'Keeping existing {split_csv} (the shared split). Use --overwrite_split to remake it.')
        return
    meta = split_by_patient(meta, args.test_frac, args.val_frac, args.seed)
    meta[['id', 'label', 'pid', 'patient', 'split']].to_csv(split_csv, index=False)
    print(meta.groupby(['split', 'class_name']).size().unstack().to_string())


if __name__ == '__main__':
    main()
