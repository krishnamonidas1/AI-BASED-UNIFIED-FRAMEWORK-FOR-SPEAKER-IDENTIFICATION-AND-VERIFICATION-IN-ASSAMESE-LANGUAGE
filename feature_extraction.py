"""
feature_extraction.py  –  MFCC vector for ML, log-mel spectrogram for DL
"""
import numpy as np
import librosa
import config


def extract_ml_features(wav_path: str) -> np.ndarray:
    """Flat feature vector: MFCC + delta + delta2 + spectral statistics."""
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC,
                                   n_fft=config.N_FFT,
                                   hop_length=config.HOP_LENGTH,
                                   win_length=config.WIN_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr           = librosa.feature.zero_crossing_rate(y)
    rms           = librosa.feature.rms(y=y)

    def stats(x):
        return np.concatenate([x.mean(axis=1), x.std(axis=1)])

    feat = np.concatenate([
        stats(mfcc), stats(delta), stats(delta2),
        spec_centroid.mean(axis=1), spec_centroid.std(axis=1),
        spec_rolloff.mean(axis=1),  spec_rolloff.std(axis=1),
        stats(spec_contrast),
        zcr.mean(axis=1), zcr.std(axis=1),
        rms.mean(axis=1), rms.std(axis=1),
    ])
    return feat.astype(np.float32)


def extract_dl_features(wav_path: str) -> np.ndarray:
    """Normalised log-mel spectrogram padded/truncated to FIXED_FRAMES → [n_mels, T]."""
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

    T, F = log_mel.shape[1], config.FIXED_FRAMES
    if T >= F:
        log_mel = log_mel[:, :F]
    else:
        pad = np.zeros((config.N_MELS, F - T), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)

    return log_mel.astype(np.float32)
