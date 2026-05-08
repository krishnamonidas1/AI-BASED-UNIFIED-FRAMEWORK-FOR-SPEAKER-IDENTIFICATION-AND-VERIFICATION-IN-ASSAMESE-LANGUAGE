
import os, tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import librosa
import librosa.display
import torch

import config
from utils import (load_all_ml, load_all_dl, load_gallery,
                   identify_ml, identify_dl,
                   verify_ml, verify_dl, majority_vote)
from evaluate import aggregate_cm

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assamese Speaker Identification & Verification",
    page_icon="",
    layout="centered",
)

# ── Clean CSS matching reference UI style ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1a2e;
}

/* Main title */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.15rem;
    font-weight: 600;
    color: #111827;
    margin: 1.4rem 0 0.6rem 0;
}
.section-header .diamond {
    color: #3b82f6;
    font-size: 0.9rem;
}

/* Result boxes */
.result-accept {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 12px 16px;
    color: #166534;
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 8px;
}
.result-reject {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 8px;
    padding: 12px 16px;
    color: #991b1b;
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 8px;
}
.result-identify {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 12px 16px;
    color: #166534;
    font-weight: 500;
    font-size: 0.95rem;
    margin-top: 8px;
}
.result-identify .speaker-name {
    font-weight: 700;
    font-size: 1.05rem;
}
.result-label {
    font-weight: 600;
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

/* Score pill */
.score-pill {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 8px;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: #f3f4f6;
    color: #374151;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* Status */
.status-ok   { color: #16a34a; font-weight: 600; }
.status-miss { color: #dc2626; font-weight: 600; }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load():
    ml_pipes, ml_le     = load_all_ml()
    xvec, ecapa, dl_le  = load_all_dl()
    gal_xvec  = load_gallery("X-Vector")
    gal_ecapa = load_gallery("ECAPA-TDNN")
    return ml_pipes, ml_le, xvec, ecapa, dl_le, gal_xvec, gal_ecapa

ml_pipes, ml_le, xvec, ecapa, dl_le, gal_xvec, gal_ecapa = _load()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – model status + results summary
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_saved_accuracies():
    """
    Load accuracy scores saved by train.py into results/accuracies.json.
    Falls back to re-evaluating from test data if the JSON doesn't exist.
    Returns dict: {model_name: accuracy_float} or {}
    """
    import json
    from evaluate import evaluate_ml, evaluate_dl
    from dl_models import XVectorNet, ECAPATDNNNet, load_dl_model

    acc_path = os.path.join(config.RESULTS_DIR, "accuracies.json")
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            return json.load(f)

    # Re-compute on the fly if JSON missing but models exist
    scores = {}
    try:
        from data_loader import load_ml_dataset
        pipes_tmp, le_tmp = load_all_ml()
        if pipes_tmp and le_tmp:
            _, X_te, _, y_te, _ = load_ml_dataset()
            res = evaluate_ml(pipes_tmp, X_te, y_te, le_tmp)
            for name, r in res.items():
                scores[name] = round(r["accuracy"] * 100, 2)
    except Exception:
        pass
    try:
        from data_loader import load_dl_dataset
        xv_tmp, ec_tmp, le_dl = load_all_dl()
        if xv_tmp and ec_tmp and le_dl:
            _, X_te, _, y_te, _ = load_dl_dataset()
            for mname, mdl in [("X-Vector", xv_tmp), ("ECAPA-TDNN", ec_tmp)]:
                r = evaluate_dl(mdl, X_te, y_te, le_dl, mname)
                scores[mname] = round(r[mname]["accuracy"] * 100, 2)
    except Exception:
        pass
    return scores


with st.sidebar:
    st.markdown("### 🎙 Model Status")

    def ok(cond, label):
        icon = "" if cond else ""
        st.markdown(f"{icon} {label}")

    ok(ml_pipes is not None and "SVM" in (ml_pipes or {}),               "SVM")
    ok(ml_pipes is not None and "Random Forest" in (ml_pipes or {}),     "Random Forest")
    ok(ml_pipes is not None and "Gradient Boosting" in (ml_pipes or {}), "Gradient Boosting")
    ok(xvec  is not None, "X-Vector")
    ok(ecapa is not None, "ECAPA-TDNN")

    st.divider()

    if ml_le is not None:
        st.markdown(f"**Speakers trained:** {len(ml_le.classes_)}")
        with st.expander("Speaker list"):
            for s in ml_le.classes_:
                st.caption(s)
    device_str = "GPU " if torch.cuda.is_available() else "CPU"
    st.markdown(f"**Device:** {device_str}")

    st.divider()

    # ── Results Summary ───────────────────────────────────────────────────────
    st.markdown("###  Results Summary")

    accs = _load_saved_accuracies()

    if not accs:
        st.caption("No results yet. Train models first.")
    else:
        # Color-code by accuracy band
        def acc_color(v):
            if v >= 90: return "#16a34a"    # green
            if v >= 75: return "#d97706"    # amber
            return "#dc2626"                # red

        def acc_bar(v):
            """Mini HTML progress bar."""
            color = acc_color(v)
            return (
                f'<div style="margin:6px 0 10px 0">'
                f'  <div style="display:flex;justify-content:space-between;'
                f'              font-size:0.78rem;font-weight:600;margin-bottom:3px">'
                f'    <span style="color:#374151">{{}}</span>'
                f'    <span style="color:{color}">{v:.1f}%</span>'
                f'  </div>'
                f'  <div style="background:#e5e7eb;border-radius:999px;height:7px">'
                f'    <div style="background:{color};width:{v}%;'
                f'                height:7px;border-radius:999px"></div>'
                f'  </div>'
                f'</div>'
            )

        # ML section
        ml_names = ["SVM", "Random Forest", "Gradient Boosting"]
        dl_names = ["X-Vector", "ECAPA-TDNN"]

        ml_avail = {n: accs[n] for n in ml_names if n in accs}
        dl_avail = {n: accs[n] for n in dl_names if n in accs}

        if ml_avail:
            st.markdown("**Machine Learning**")
            for name, val in ml_avail.items():
                st.markdown(acc_bar(val).format(name), unsafe_allow_html=True)

        if dl_avail:
            st.markdown("**Deep Learning**")
            for name, val in dl_avail.items():
                st.markdown(acc_bar(val).format(name), unsafe_allow_html=True)

        # Best model highlight
        if accs:
            best_name = max(accs, key=accs.get)
            best_val  = accs[best_name]
            st.markdown(
                f'<div style="background:#eff6ff;border:1px solid #bfdbfe;'
                f'border-radius:8px;padding:8px 12px;margin-top:6px;'
                f'font-size:0.82rem;color:#1e40af">'
                f' Best: <b>{best_name}</b> ({best_val:.1f}%)'
                f'</div>',
                unsafe_allow_html=True
            )

        if st.button("Refresh Results", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Speaker Identification &amp; Verification System</div>',
            unsafe_allow_html=True)
st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# Top controls
# ─────────────────────────────────────────────────────────────────────────────
model_type = st.selectbox("Select Model Type",
                          ["Machine Learning", "Deep Learning"])

st.divider()

# ── Audio input ───────────────────────────────────────────────────────────────
st.markdown("**Audio Input Method**")
input_method = st.radio("", ["Upload WAV", "Record Audio"],
                        label_visibility="collapsed")

tmp_path = None

if input_method == "Upload WAV":
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        st.audio(uploaded)

else:  # Record Audio
    audio_bytes = st.audio_input("Record your voice")
    if audio_bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes.read())
            tmp_path = tmp.name

# Waveform preview
if tmp_path:
    try:
        y_wav, sr_wav = librosa.load(tmp_path, sr=config.SAMPLE_RATE)
        fig_w, ax_w = plt.subplots(figsize=(7, 1.5))
        fig_w.patch.set_facecolor("white")
        ax_w.set_facecolor("#f9fafb")
        librosa.display.waveshow(y_wav, sr=sr_wav, ax=ax_w,
                                  color="#3b82f6", alpha=0.8)
        ax_w.set_xlabel("Time (s)", fontsize=8, color="#6b7280")
        ax_w.tick_params(colors="#9ca3af", labelsize=7)
        for spine in ax_w.spines.values():
            spine.set_edgecolor("#e5e7eb")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig_w); plt.close()
    except Exception:
        pass

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: waveform & probability bar chart
# ─────────────────────────────────────────────────────────────────────────────
def show_prob_bar(probas, classes, predicted_idx):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9fafb")
    colors = ["#3b82f6" if i == predicted_idx else "#d1d5db"
              for i in range(len(classes))]
    ax.bar(classes, probas * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Confidence (%)", fontsize=8, color="#6b7280")
    ax.tick_params(axis='x', rotation=45, labelsize=7, colors="#374151")
    ax.tick_params(axis='y', labelsize=7, colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e7eb")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ════════════  MACHINE LEARNING  ════════════
# ─────────────────────────────────────────────────────────────────────────────
if model_type == "Machine Learning":

    if ml_pipes is None or ml_le is None:
        st.warning("ML models not found. Run `python train.py --mode ml` first.")
    else:
        # ── IDENTIFICATION ───────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">'
            '<span class="diamond">◆</span>'
            'Machine Learning Speaker Identification'
            '</div>', unsafe_allow_html=True)

        ml_model_id = st.selectbox("Select ML Model",
                                    ["SVM", "Random Forest", "Gradient Boosting"],
                                    key="ml_id_sel")

        if tmp_path:
            with st.spinner("Identifying …"):
                spk, conf, probas, classes = identify_ml(
                    ml_pipes, ml_le, tmp_path, ml_model_id)
            pred_idx = list(classes).index(spk)
            st.markdown(
                f'<div class="result-identify">'
                f'Identified Speaker: <span class="speaker-name">{spk}</span>'
                f'<span class="score-pill">{conf:.1f}%</span>'
                f'</div>', unsafe_allow_html=True)
            show_prob_bar(probas, classes, pred_idx)
        else:
            st.info("Upload or record a WAV file to identify the speaker.")

        st.divider()

        # ── VERIFICATION ─────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">'
            '<span class="diamond">◆</span>'
            'Machine Learning Speaker Verification'
            '</div>', unsafe_allow_html=True)

        claimed_ml = st.selectbox("Claimed Speaker", list(ml_le.classes_),
                                   key="ml_verify_spk")
        ml_thresh  = st.slider("Acceptance Threshold (probability)",
                                0.0, 1.0, config.ML_VERIFY_THRESH, 0.01,
                                key="ml_thresh")

        if tmp_path:
            with st.spinner("Verifying …"):
                ml_model_ver = st.selectbox(
                    "Verify using model",
                    ["SVM", "Random Forest", "Gradient Boosting"],
                    key="ml_ver_sel")
                verdict = verify_ml(
                    {ml_model_ver: ml_pipes[ml_model_ver]},
                    ml_le, tmp_path, claimed_ml, ml_thresh)

            v = verdict[ml_model_ver]
            box_cls = "result-accept" if v["decision"] == "ACCEPT" else "result-reject"
            icon    = "" if v["decision"] == "ACCEPT" else ""
            st.markdown(
                f'<div class="{box_cls}">'
                f'{icon} <b>{v["decision"]}</b> — {claimed_ml}'
                f'<span class="score-pill">score {v["score"]:.3f}</span>'
                f'</div>', unsafe_allow_html=True)

            # All-model comparison table
            with st.expander("All ML models verdict"):
                all_verdicts = verify_ml(ml_pipes, ml_le, tmp_path,
                                          claimed_ml, ml_thresh)
                for mname, mv in all_verdicts.items():
                    badge = "result-accept" if mv["decision"]=="ACCEPT" \
                             else "result-reject"
                    st.markdown(
                        f'<span class="model-badge">{mname}</span>'
                        f'<b>{mv["decision"]}</b>  score={mv["score"]:.3f}',
                        unsafe_allow_html=True)
        else:
            st.info("Upload or record a WAV file to verify.")

# ─────────────────────────────────────────────────────────────────────────────
# ════════════  DEEP LEARNING  ════════════
# ─────────────────────────────────────────────────────────────────────────────
else:
    dl_models_map = {}
    if xvec  is not None: dl_models_map["X-Vector"]    = (xvec,  gal_xvec,  dl_le)
    if ecapa is not None: dl_models_map["ECAPA-TDNN"]  = (ecapa, gal_ecapa, dl_le)

    if not dl_models_map:
        st.warning("  DL models not found. Run `python train.py --mode dl` first.")
    else:
        # ── IDENTIFICATION ───────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">'
            '<span class="diamond">◆</span>'
            'Deep Learning Speaker Identification'
            '</div>', unsafe_allow_html=True)

        dl_model_id = st.selectbox("Select DL Model",
                                    list(dl_models_map.keys()),
                                    key="dl_id_sel")
        model_id, _, le_id = dl_models_map[dl_model_id]

        if tmp_path and le_id is not None:
            with st.spinner("Identifying …"):
                spk, conf, probas, classes = identify_dl(model_id, le_id, tmp_path)
            pred_idx = list(classes).index(spk)
            st.markdown(
                f'<div class="result-identify">'
                f'🔑 Identified Speaker: <span class="speaker-name">{spk}</span>'
                f'<span class="score-pill">{conf:.1f}%</span>'
                f'</div>', unsafe_allow_html=True)
            show_prob_bar(probas, classes, pred_idx)
        else:
            st.info("Upload or record a WAV file to identify the speaker.")

        st.divider()

        # ── VERIFICATION ─────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">'
            '<span class="diamond">◆</span>'
            'Deep Learning Speaker Verification'
            '</div>', unsafe_allow_html=True)

        le_ver = dl_le
        if le_ver is None:
            st.warning("DL label encoder not found.")
        else:
            claimed_dl = st.selectbox("Claimed Speaker",
                                       list(le_ver.classes_), key="dl_verify_spk")
            dl_thresh  = st.slider("Cosine Similarity Threshold",
                                    0.0, 1.0, config.COSINE_THRESHOLD, 0.01,
                                    key="dl_thresh")

            if tmp_path:
                dl_model_ver = st.selectbox("Verify using model",
                                             list(dl_models_map.keys()),
                                             key="dl_ver_sel")
                model_ver, gallery_ver, _ = dl_models_map[dl_model_ver]

                if gallery_ver is None:
                    st.warning(f"Gallery for {dl_model_ver} not built. "
                                "Run `python train.py --mode gallery`.")
                else:
                    with st.spinner("Verifying …"):
                        verdict = verify_dl(model_ver, gallery_ver,
                                             tmp_path, claimed_dl, dl_thresh)
                    box_cls = "result-accept" if verdict["decision"]=="ACCEPT" \
                               else "result-reject"
                    icon = "✅" if verdict["decision"] == "ACCEPT" else "❌"
                    st.markdown(
                        f'<div class="{box_cls}">'
                        f'{icon} <b>{verdict["decision"]}</b> — {claimed_dl}'
                        f'<span class="score-pill">cosine {verdict["score"]:.3f}</span>'
                        f'</div>', unsafe_allow_html=True)

                    # Cross-model comparison
                    with st.expander("All DL models verdict"):
                        all_dl_verdicts = {}
                        for mname, (mv_model, mv_gal, _) in dl_models_map.items():
                            if mv_gal is not None:
                                all_dl_verdicts[mname] = verify_dl(
                                    mv_model, mv_gal,
                                    tmp_path, claimed_dl, dl_thresh)
                        for mname, mv in all_dl_verdicts.items():
                            st.markdown(
                                f'<span class="model-badge">{mname}</span>'
                                f'<b>{mv["decision"]}</b>  cosine={mv["score"]:.3f}',
                                unsafe_allow_html=True)
                        if len(all_dl_verdicts) > 1:
                            mv_final = majority_vote(all_dl_verdicts)
                            st.markdown(f"**Majority vote:** `{mv_final}`")
            else:
                st.info("Upload or record a WAV file to verify.")

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup temp file
# ─────────────────────────────────────────────────────────────────────────────
if tmp_path and os.path.exists(tmp_path):
    try:
        os.unlink(tmp_path)
    except Exception:
        pass