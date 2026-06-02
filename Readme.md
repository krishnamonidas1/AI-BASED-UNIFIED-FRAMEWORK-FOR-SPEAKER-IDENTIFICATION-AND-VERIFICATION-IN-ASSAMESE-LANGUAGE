# Assamese Speaker Identification & Verification System

## Folder Structure
```
SPEAKER_ID/
├── Dataset/                  ← Put your dataset here
│   ├── SPEAKER 1/
│   │   └── ASS_S1_SEN1(1).wav ...
│   └── Speaker 14/ ...
├── models/                   ← Auto-created: ML model pickles
├── deep_models/              ← Auto-created: DL .pt weights
├── pretrained_models/        ← Auto-created: speaker galleries
├── results/                  ← Auto-created: confusion matrix PNGs
├── Features/                 ← Auto-created: cached arrays
├── App.py                    ← Streamlit UI
├── train.py                  ← Terminal training
├── utils_confusion.py        ← Standalone CM plots
├── config.py
├── feature_extraction.py
├── data_loader.py
├── ml_models.py
├── dl_models.py
├── evaluate.py
├── utils.py
└── Requirments.txt
```

## Setup
```bash
pip install -r Requirments.txt
```

## Training (Terminal Only)
```bash
# Train everything (ML + DL + gallery)
python train.py --mode all

# Train only ML (SVM, Random Forest, Gradient Boosting)
python train.py --mode ml

# Train only DL (X-Vector, ECAPA-TDNN)
python train.py --mode dl

# Build speaker gallery for verification (after DL training)
python train.py --mode gallery

# Evaluate all models and print accuracy
python train.py --mode eval

# Custom epochs
python train.py --mode dl --epochs 50
```

## Run Streamlit App
```bash
streamlit run App.py
```

## Generate Confusion Matrix Plots
```bash
python utils_confusion.py
# Saves PNGs to results/
```
### Generates excel file of Train & Test
 python export_splits.py
 
## What the App Does
- **Select Model Type**: Machine Learning or Deep Learning
- **Audio Input**: Upload WAV or record directly in browser
- **Identification**: Predict which of 20 speakers the audio belongs to
- **Verification**: Claim a speaker → get ACCEPT/REJECT decision
  - ML: uses predicted probability vs threshold
  - DL: uses cosine similarity of embeddings vs threshold
The Dataset is uploaded in kaggle(ping me to grt it)
