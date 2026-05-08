"""
evaluate.py  –  accuracy + per-class 2×2 confusion matrix
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import config
from dl_models import get_device


def _per_class_cm(y_true, y_pred, n_classes):
    out = {}
    for c in range(n_classes):
        pos  = (y_true == c).astype(int)
        pred = (y_pred == c).astype(int)
        out[c] = {
            "TP": int(((pos==1)&(pred==1)).sum()),
            "TN": int(((pos==0)&(pred==0)).sum()),
            "FP": int(((pos==0)&(pred==1)).sum()),
            "FN": int(((pos==1)&(pred==0)).sum()),
        }
    return out


def aggregate_cm(cm_2x2: dict):
    agg = {"TP":0,"TN":0,"FP":0,"FN":0}
    for v in cm_2x2.values():
        for k in agg: agg[k] += v[k]
    return agg


def evaluate_ml(pipes, X_te, y_te, le):
    results = {}
    for name, pipe in pipes.items():
        y_pred = pipe.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        results[name] = {
            "accuracy": acc,
            "cm_2x2":  _per_class_cm(y_te, y_pred, len(le.classes_)),
            "y_pred":  y_pred, "y_true": y_te, "classes": le.classes_,
        }
    return results


def evaluate_dl(model, X_te, y_te, le, name):
    device = get_device()
    model.eval()
    Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits, _ = model(Xte)
    y_pred = logits.argmax(1).cpu().numpy()
    acc    = accuracy_score(y_te, y_pred)
    return {name: {
        "accuracy": acc,
        "cm_2x2":  _per_class_cm(y_te, y_pred, len(le.classes_)),
        "y_pred":  y_pred, "y_true": y_te, "classes": le.classes_,
    }}
