"""
ml_models.py  –  SVM, Random Forest, Gradient Boosting
Each saved separately into models/
"""
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config

MODEL_DEFS = {
    "SVM": (SVC(kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=config.RANDOM_STATE),
            config.ML_SVM_PATH),
    "Random Forest": (RandomForestClassifier(
                        n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
                      config.ML_RF_PATH),
    "Gradient Boosting": (GradientBoostingClassifier(
                            n_estimators=200, learning_rate=0.1,
                            max_depth=5, random_state=config.RANDOM_STATE),
                          config.ML_GB_PATH),
}


def build_pipelines():
    return {name: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            for name, (clf, _) in MODEL_DEFS.items()}


def train_and_save_ml(X_tr, y_tr, le):
    pipes = build_pipelines()
    for name, pipe in pipes.items():
        print(f"  [ML] Training {name} …", flush=True)
        pipe.fit(X_tr, y_tr)
        path = MODEL_DEFS[name][1]
        with open(path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"       Saved → {path}")
    with open(config.LABEL_ENC_PATH, "wb") as f:
        pickle.dump(le, f)
    return pipes


def load_ml_models():
    pipes = {}
    for name, (_, path) in MODEL_DEFS.items():
        with open(path, "rb") as f:
            pipes[name] = pickle.load(f)
    return pipes


def load_label_encoder():
    with open(config.LABEL_ENC_PATH, "rb") as f:
        return pickle.load(f)


def predict_ml(pipes: dict, feat: np.ndarray):
    """Returns {model_name: {label_idx, probas}}"""
    x = feat.reshape(1, -1)
    return {name: {"label_idx": int(p.predict(x)[0]),
                   "probas":    p.predict_proba(x)[0]}
            for name, p in pipes.items()}
