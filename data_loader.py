"""
data_loader.py  –  walks Dataset/ and builds feature arrays
"""
import os, glob, re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import config
from feature_extraction import extract_ml_features, extract_dl_features


def _speaker_label(folder_name: str) -> str:
    num = re.search(r"\d+", folder_name)
    return f"Speaker_{int(num.group()):02d}" if num else folder_name


def collect_wav_paths(dataset_root: str = config.DATASET_ROOT):
    """Return list of (wav_path, label_str) tuples."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset folder '{dataset_root}' not found. "
            "Set DATASET_ROOT in config.py"
        )
    records = []
    for spk_dir in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, spk_dir)
        if not os.path.isdir(full):
            continue
        label = _speaker_label(spk_dir)
        for wav in sorted(glob.glob(os.path.join(full, "*.wav"))):
            records.append((wav, label))
    return records


def load_ml_dataset(dataset_root: str = config.DATASET_ROOT, progress_cb=None):
    records = collect_wav_paths(dataset_root)
    X, y_raw = [], []
    for i, (wav, label) in enumerate(records):
        try:
            X.append(extract_ml_features(wav))
            y_raw.append(label)
        except Exception as e:
            print(f"[WARN] {wav}: {e}")
        if progress_cb:
            progress_cb(i + 1, len(records))
    X  = np.array(X)
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)
    return X_tr, X_te, y_tr, y_te, le


def load_dl_dataset(dataset_root: str = config.DATASET_ROOT, progress_cb=None):
    records = collect_wav_paths(dataset_root)
    X, y_raw = [], []
    for i, (wav, label) in enumerate(records):
        try:
            feat = extract_dl_features(wav)
            X.append(feat[np.newaxis])        # [1, n_mels, T]
            y_raw.append(label)
        except Exception as e:
            print(f"[WARN] {wav}: {e}")
        if progress_cb:
            progress_cb(i + 1, len(records))
    X  = np.array(X)
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)
    return X_tr, X_te, y_tr, y_te, le
