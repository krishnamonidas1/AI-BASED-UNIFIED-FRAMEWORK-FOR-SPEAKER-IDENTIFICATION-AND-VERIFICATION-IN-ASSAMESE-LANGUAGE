"""
utils_confusion.py  –  Plot and save confusion matrices for all trained models.
Run standalone: python utils_confusion.py
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from ml_models   import load_ml_models, load_label_encoder
from dl_models   import XVectorNet, ECAPATDNNNet, load_dl_model
from data_loader import load_ml_dataset, load_dl_dataset
from evaluate    import evaluate_ml, evaluate_dl, aggregate_cm


def plot_cm_2x2(agg: dict, title: str, save_path: str):
    arr = np.array([[agg["TP"], agg["FN"]],
                    [agg["FP"], agg["TN"]]])
    labels = [["TP", "FN"], ["FP", "TN"]]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Positive", "Predicted Negative"],
                yticklabels=["Actual Positive",    "Actual Negative"],
                ax=ax, linewidths=1, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})

    # Add TP/FN/FP/TN labels inside cells
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.82, labels[i][j],
                    ha="center", va="center", fontsize=9,
                    color="gray", alpha=0.7)

    total  = sum(agg.values())
    denom  = agg["TP"] + agg["FN"]
    acc    = agg["TP"] / denom if denom else 0
    ax.set_title(f"{title}\nAcc={acc:.2%}  N={total}", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_per_speaker(cm_2x2: dict, classes, title: str, save_path: str):
    """Bar chart: TP rate (sensitivity) per speaker."""
    sensitivities = []
    for idx in range(len(classes)):
        cm = cm_2x2[idx]
        d  = cm["TP"] + cm["FN"]
        sensitivities.append(cm["TP"] / d if d else 0)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors  = ["#3b82f6" if s >= 0.8 else "#f59e0b" if s >= 0.5 else "#ef4444"
               for s in sensitivities]
    ax.bar(classes, [s*100 for s in sensitivities], color=colors, edgecolor="white")
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, label="80% line")
    ax.set_ylabel("Sensitivity / Recall (%)")
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{title} – Per-Speaker Sensitivity")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ── ML ────────────────────────────────────────────────────────────────────
    print("\nML Confusion Matrices …")
    try:
        pipes = load_ml_models()
        le    = load_label_encoder()
        _, X_te, _, y_te, _ = load_ml_dataset()
        results = evaluate_ml(pipes, X_te, y_te, le)
        for name, r in results.items():
            agg = aggregate_cm(r["cm_2x2"])
            safe = name.replace(" ", "_")
            plot_cm_2x2(agg, name,
                        os.path.join(config.RESULTS_DIR, f"cm_2x2_{safe}.png"))
            plot_per_speaker(r["cm_2x2"], le.classes_, name,
                             os.path.join(config.RESULTS_DIR,
                                          f"per_speaker_{safe}.png"))
    except Exception as e:
        print(f"  ML skipped: {e}")

    # ── DL ────────────────────────────────────────────────────────────────────
    print("\nDL Confusion Matrices …")
    try:
        with open(config.DL_LABEL_PATH, "rb") as f:
            le_dl = pickle.load(f)
        num_spk = len(le_dl.classes_)
        _, X_te, _, y_te, _ = load_dl_dataset()
        for ModelCls, path, name in [
            (XVectorNet,   config.DL_XVEC_PATH,  "X-Vector"),
            (ECAPATDNNNet, config.DL_ECAPA_PATH,  "ECAPA-TDNN"),
        ]:
            model = load_dl_model(ModelCls, path, config.N_MELS, num_spk)
            r     = evaluate_dl(model, X_te, y_te, le_dl, name)
            agg   = aggregate_cm(r[name]["cm_2x2"])
            safe  = name.replace("-", "_")
            plot_cm_2x2(agg, name,
                        os.path.join(config.RESULTS_DIR, f"cm_2x2_{safe}.png"))
            plot_per_speaker(r[name]["cm_2x2"], le_dl.classes_, name,
                             os.path.join(config.RESULTS_DIR,
                                          f"per_speaker_{safe}.png"))
    except Exception as e:
        print(f"  DL skipped: {e}")

    print("\n✅ All confusion matrix plots saved to results/")


if __name__ == "__main__":
    main()
