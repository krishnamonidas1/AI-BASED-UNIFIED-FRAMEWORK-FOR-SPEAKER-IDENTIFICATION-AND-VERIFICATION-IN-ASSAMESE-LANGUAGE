"""
utils.py  –  shared helpers used by App.py and train.py
"""
import os, pickle
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

import config
from feature_extraction import extract_ml_features, extract_dl_features
from ml_models          import load_ml_models, load_label_encoder, predict_ml
from dl_models          import (XVectorNet, ECAPATDNNNet,
                                load_dl_model, get_embedding, get_device)
from data_loader        import collect_wav_paths


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_ml():
    """Returns (pipes, le) or (None, None)."""
    try:
        pipes = load_ml_models()
        le    = load_label_encoder()
        return pipes, le
    except Exception:
        return None, None


def load_all_dl():
    """Returns (xvec, ecapa, le) or (None, None, None)."""
    try:
        with open(config.DL_LABEL_PATH, "rb") as f:
            le = pickle.load(f)
        num_spk = len(le.classes_)
        xvec  = load_dl_model(XVectorNet,    config.DL_XVEC_PATH,  config.N_MELS, num_spk)
        ecapa = load_dl_model(ECAPATDNNNet,  config.DL_ECAPA_PATH, config.N_MELS, num_spk)
        return xvec, ecapa, le
    except Exception:
        return None, None, None


def load_gallery(model_name: str):
    """Load speaker gallery dict {label: embedding}."""
    path = config.GALLERY_XVEC_PATH if model_name == "X-Vector" \
           else config.GALLERY_ECAPA_PATH
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Identification
# ─────────────────────────────────────────────────────────────────────────────

def identify_ml(pipes, le, wav_path, model_name="SVM"):
    feat  = extract_ml_features(wav_path)
    preds = predict_ml({model_name: pipes[model_name]}, feat)
    out   = preds[model_name]
    spk   = le.inverse_transform([out["label_idx"]])[0]
    conf  = float(np.max(out["probas"])) * 100
    return spk, conf, out["probas"], le.classes_


def identify_dl(model, le, wav_path):
    device = get_device()
    model.eval()
    spec = extract_dl_features(wav_path)
    t    = torch.tensor(spec[np.newaxis, np.newaxis],
                        dtype=torch.float32).to(device)
    with torch.no_grad():
        logits, _ = model(t)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx   = int(probs.argmax())
    spk   = le.inverse_transform([idx])[0]
    return spk, float(probs[idx])*100, probs, le.classes_


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_ml(pipes, le, wav_path, claimed_speaker, threshold=config.ML_VERIFY_THRESH):
    feat = extract_ml_features(wav_path)
    try:
        idx = list(le.classes_).index(claimed_speaker)
    except ValueError:
        return {n: {"decision":"REJECT","score":0.0} for n in pipes}
    results = {}
    x = feat.reshape(1, -1)
    for name, pipe in pipes.items():
        score    = float(pipe.predict_proba(x)[0][idx])
        decision = "ACCEPT" if score >= threshold else "REJECT"
        results[name] = {"decision": decision, "score": round(score, 4)}
    return results


def verify_dl(model, gallery, wav_path, claimed_speaker,
              threshold=config.COSINE_THRESHOLD):
    if gallery is None or claimed_speaker not in gallery:
        return {"decision": "REJECT", "score": 0.0}
    emb_query   = get_embedding(model, extract_dl_features(wav_path))
    emb_gallery = gallery[claimed_speaker]
    score = float(cosine_similarity(
        emb_query.reshape(1,-1), emb_gallery.reshape(1,-1))[0,0])
    return {"decision": "ACCEPT" if score >= threshold else "REJECT",
            "score": round(score, 4)}


def majority_vote(verdicts: dict):
    decisions = [v["decision"] for v in verdicts.values()]
    return "ACCEPT" if decisions.count("ACCEPT") > len(decisions)//2 else "REJECT"
