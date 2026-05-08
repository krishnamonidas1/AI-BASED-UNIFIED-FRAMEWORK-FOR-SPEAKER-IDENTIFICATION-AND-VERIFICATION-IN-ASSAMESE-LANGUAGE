import os

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_ROOT  = "Dataset"
NUM_SPEAKERS  = 14
SAMPLE_RATE   = 16000

# ── Feature Extraction ────────────────────────────────────────────────────────
N_MFCC        = 40
N_MELS        = 80
HOP_LENGTH    = 160
WIN_LENGTH    = 400
N_FFT         = 512
FIXED_FRAMES  = 300

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
DL_EPOCHS     = 30
DL_BATCH      = 16
DL_LR         = 1e-3
EMBEDDING_DIM = 192

# ── Folder Structure ──────────────────────────────────────────────────────────
MODELS_DIR       = "models"            # ML model pickles
DEEP_MODELS_DIR  = "deep_models"       # DL .pt weights
PRETRAINED_DIR   = "pretrained_models" # speaker gallery embeddings
RESULTS_DIR      = "results"           # eval outputs
FEATURES_DIR     = "Features"          # cached numpy arrays

for d in [MODELS_DIR, DEEP_MODELS_DIR, PRETRAINED_DIR, RESULTS_DIR, FEATURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── ML paths ──────────────────────────────────────────────────────────────────
ML_SVM_PATH    = os.path.join(MODELS_DIR, "svm.pkl")
ML_RF_PATH     = os.path.join(MODELS_DIR, "random_forest.pkl")
ML_GB_PATH     = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
LABEL_ENC_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ── DL paths ──────────────────────────────────────────────────────────────────
DL_XVEC_PATH   = os.path.join(DEEP_MODELS_DIR, "xvector.pt")
DL_ECAPA_PATH  = os.path.join(DEEP_MODELS_DIR, "ecapa_tdnn.pt")
DL_LABEL_PATH  = os.path.join(DEEP_MODELS_DIR, "dl_label_encoder.pkl")

# ── Gallery (for DL verification) ─────────────────────────────────────────────
GALLERY_XVEC_PATH  = os.path.join(PRETRAINED_DIR, "gallery_xvector.pkl")
GALLERY_ECAPA_PATH = os.path.join(PRETRAINED_DIR, "gallery_ecapa.pkl")

# ── Verification thresholds ───────────────────────────────────────────────────
COSINE_THRESHOLD = 0.75
ML_VERIFY_THRESH = 0.50
