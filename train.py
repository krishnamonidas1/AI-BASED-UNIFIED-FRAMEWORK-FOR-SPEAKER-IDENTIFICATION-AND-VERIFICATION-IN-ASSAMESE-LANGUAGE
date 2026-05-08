"""
train.py  –  Run from terminal to train all models
Usage:
    python train.py --mode all        # Train ML + DL + build gallery
    python train.py --mode ml         # Train only ML models
    python train.py --mode dl         # Train only DL models
    python train.py --mode gallery    # Build DL speaker gallery only
    python train.py --mode eval       # Evaluate all trained models
"""
import argparse, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data_loader        import load_ml_dataset, load_dl_dataset, collect_wav_paths
from ml_models          import train_and_save_ml, load_ml_models, load_label_encoder
from dl_models          import (XVectorNet, ECAPATDNNNet, get_device,
                                save_dl_model, get_embedding)
from evaluate           import evaluate_ml, evaluate_dl, aggregate_cm
from feature_extraction import extract_dl_features

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# DL training loop
# ─────────────────────────────────────────────────────────────────────────────
def _train_dl(model, X_tr, y_tr, X_te, y_te, name,
              epochs=config.DL_EPOCHS, batch=config.DL_BATCH, lr=config.DL_LR):
    device = get_device()
    model  = model.to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch, shuffle=True, drop_last=False)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    yte_t = torch.tensor(y_te, dtype=torch.long).to(device)

    print(f"\n  Training {name} for {epochs} epochs …")
    for ep in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()*len(yb)
            correct    += (logits.argmax(1)==yb).sum().item()
            total      += len(yb)
        sched.step()
        model.eval()
        with torch.no_grad():
            lg, _ = model(Xte_t)
            val_acc = (lg.argmax(1)==yte_t).float().mean().item()
        if ep % 5 == 0 or ep == epochs:
            print(f"    Epoch {ep:3d}/{epochs}  "
                  f"loss={total_loss/total:.4f}  "
                  f"train_acc={correct/total:.3f}  val_acc={val_acc:.3f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Build speaker gallery (mean embeddings per speaker)
# ─────────────────────────────────────────────────────────────────────────────
def build_gallery(model, le, name):
    print(f"\n  Building gallery for {name} …")
    records = collect_wav_paths()
    gallery = {}
    for wav, label in records:
        try:
            emb = get_embedding(model, extract_dl_features(wav))
            gallery.setdefault(label, []).append(emb)
        except Exception as e:
            print(f"    [WARN] {wav}: {e}")
    gallery = {spk: np.mean(embs, axis=0) for spk, embs in gallery.items()}
    path = config.GALLERY_XVEC_PATH if name == "X-Vector" \
           else config.GALLERY_ECAPA_PATH
    with open(path, "wb") as f:
        pickle.dump(gallery, f)
    print(f"    Gallery saved → {path}  ({len(gallery)} speakers)")
    return gallery


# ─────────────────────────────────────────────────────────────────────────────
# Save confusion matrix PNG to results/
# ─────────────────────────────────────────────────────────────────────────────
def save_accuracies(scores: dict):
    """Merge new scores into results/accuracies.json — read by App.py sidebar."""
    import json
    path = os.path.join(config.RESULTS_DIR, "accuracies.json")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(scores)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Accuracies saved → {path}")


def save_cm_plot(cm_2x2, name, classes):
    from evaluate import aggregate_cm
    agg = aggregate_cm(cm_2x2)
    arr = np.array([[agg["TP"], agg["FN"]], [agg["FP"], agg["TN"]]])
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred +","Pred −"],
                yticklabels=["Act +","Act −"], ax=ax)
    ax.set_title(f"{name}  acc={agg['TP']/(agg['TP']+agg['FN']+1e-9):.2f}")
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"cm_{name.replace(' ','_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    CM saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all","ml","dl","gallery","eval"])
    parser.add_argument("--epochs", type=int, default=config.DL_EPOCHS)
    parser.add_argument("--dataset", default=config.DATASET_ROOT)
    args = parser.parse_args()
    config.DL_EPOCHS     = args.epochs
    config.DATASET_ROOT  = args.dataset

    t0 = time.time()

    # ── ML ────────────────────────────────────────────────────────────────────
    if args.mode in ("all", "ml"):
        print("\n════════ ML Training ════════")
        print("  Loading dataset & extracting features …")
        X_tr, X_te, y_tr, y_te, le = load_ml_dataset()
        print(f"  Train: {len(y_tr)}  Test: {len(y_te)}  Classes: {len(le.classes_)}")
        pipes = train_and_save_ml(X_tr, y_tr, le)

        print("\n  Evaluating ML …")
        results = evaluate_ml(pipes, X_te, y_te, le)
        ml_scores = {}
        for mname, r in results.items():
            print(f"    {mname:20s}  acc={r['accuracy']*100:.2f}%")
            save_cm_plot(r["cm_2x2"], mname, le.classes_)
            ml_scores[mname] = round(r["accuracy"] * 100, 2)
        save_accuracies(ml_scores)

    # ── DL ────────────────────────────────────────────────────────────────────
    if args.mode in ("all", "dl"):
        print("\n════════ DL Training ════════")
        print("  Loading dataset & extracting features …")
        X_tr, X_te, y_tr, y_te, le = load_dl_dataset()
        num_spk = len(le.classes_)
        print(f"  Train: {len(y_tr)}  Test: {len(y_te)}  Speakers: {num_spk}")

        with open(config.DL_LABEL_PATH, "wb") as f:
            pickle.dump(le, f)

        xvec  = _train_dl(XVectorNet(config.N_MELS, num_spk),
                           X_tr, y_tr, X_te, y_te, "X-Vector")
        save_dl_model(xvec, config.DL_XVEC_PATH)

        ecapa = _train_dl(ECAPATDNNNet(config.N_MELS, num_spk),
                           X_tr, y_tr, X_te, y_te, "ECAPA-TDNN")
        save_dl_model(ecapa, config.DL_ECAPA_PATH)

        print("\n  Evaluating DL …")
        r_xvec  = evaluate_dl(xvec,  X_te, y_te, le, "X-Vector")
        r_ecapa = evaluate_dl(ecapa, X_te, y_te, le, "ECAPA-TDNN")
        dl_scores = {}
        for name, r in {**r_xvec, **r_ecapa}.items():
            print(f"    {name:20s}  acc={r['accuracy']*100:.2f}%")
            save_cm_plot(r["cm_2x2"], name, le.classes_)
            dl_scores[name] = round(r["accuracy"] * 100, 2)
        save_accuracies(dl_scores)

        # Auto-build gallery after DL training
        build_gallery(xvec,  le, "X-Vector")
        build_gallery(ecapa, le, "ECAPA-TDNN")

    # ── Gallery only ──────────────────────────────────────────────────────────
    if args.mode == "gallery":
        print("\n════════ Building Gallery ════════")
        from utils import load_all_dl
        xvec, ecapa, le = load_all_dl()
        if xvec is None:
            print("  [ERROR] DL models not found. Train DL first.")
            return
        build_gallery(xvec,  le, "X-Vector")
        build_gallery(ecapa, le, "ECAPA-TDNN")

    # ── Eval only ─────────────────────────────────────────────────────────────
    if args.mode == "eval":
        print("\n════════ Evaluation ════════")
        eval_scores = {}
        try:
            pipes = load_ml_models()
            le    = load_label_encoder()
            _, X_te, _, y_te, _ = load_ml_dataset()
            results = evaluate_ml(pipes, X_te, y_te, le)
            for mname, r in results.items():
                print(f"  {mname:20s}  acc={r['accuracy']*100:.2f}%")
                eval_scores[mname] = round(r["accuracy"] * 100, 2)
        except Exception as e:
            print(f"  ML eval skipped: {e}")

        try:
            from utils import load_all_dl
            xvec, ecapa, le = load_all_dl()
            _, X_te, _, y_te, _ = load_dl_dataset()
            for name, model in [("X-Vector", xvec), ("ECAPA-TDNN", ecapa)]:
                r = evaluate_dl(model, X_te, y_te, le, name)
                print(f"  {name:20s}  acc={r[name]['accuracy']*100:.2f}%")
                eval_scores[name] = round(r[name]["accuracy"] * 100, 2)
        except Exception as e:
            print(f"  DL eval skipped: {e}")

        if eval_scores:
            save_accuracies(eval_scores)

    print(f"\n✅ Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()