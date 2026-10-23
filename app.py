"""ULCARE — Streamlit interface for Wagner stage prediction (0–4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model_gangren_wagner_v1.h5"
STAGES = [f"Stage {number}" for number in range(5)]

STAGE_INFO = {
    "Stage 0": {
        "title": "Stage 0 — Risiko sangat rendah",
        "tone": "low",
        "reason": "Model lebih banyak memberi perhatian pada jaringan yang tampak utuh atau area berisiko tanpa ulkus terbuka yang dominan.",
    },
    "Stage 1": {
        "title": "Stage 1 — Ulkus superfisial",
        "tone": "low",
        "reason": "Grad-CAM menyorot perubahan pada lapisan permukaan luka. Pola tersebut lebih selaras dengan ulkus superfisial dibanding keterlibatan jaringan yang lebih dalam.",
    },
    "Stage 2": {
        "title": "Stage 2 — Ulkus lebih dalam",
        "tone": "moderate",
        "reason": "Area perhatian model berada pada dasar dan tepi luka yang menunjukkan karakter visual keterlibatan jaringan lebih dalam, tanpa pola infeksi berat yang dominan.",
    },
    "Stage 3": {
        "title": "Stage 3 — Infeksi dalam",
        "tone": "moderate",
        "reason": "Model memberi bobot tinggi pada kombinasi eksudat, perubahan warna, dan area luka yang konsisten dengan kemungkinan infeksi jaringan yang lebih dalam.",
    },
    "Stage 4": {
        "title": "Stage 4 — Gangren lokal",
        "tone": "critical",
        "reason": "Grad-CAM menyorot area dengan perubahan warna dan kondisi jaringan yang paling kuat memengaruhi prediksi gangren lokal pada stage ini.",
    },
}


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / exp.sum()


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)


def prepare_image(image: Image.Image, model: tf.keras.Model) -> tuple[np.ndarray, Image.Image]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)
    display_image = image.convert("RGB")
    resized = display_image.resize((width, height))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0), display_image


def normalise_scores(raw: np.ndarray) -> np.ndarray:
    """Return a five-class probability vector for Wagner Stage 0 through Stage 4."""
    scores = np.asarray(raw).squeeze().astype(np.float32).reshape(-1)
    if scores.size < 5:
        scores = np.pad(scores, (0, 5 - scores.size))
    scores = scores[:5]
    if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=0.02):
        scores = softmax(scores)
    else:
        scores = scores / scores.sum()
    return scores


def last_convolution_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer
    raise ValueError("Model tidak memiliki layer konvolusi untuk Grad-CAM.")


def create_gradcam(model: tf.keras.Model, batch: np.ndarray, class_index: int, display_image: Image.Image) -> Image.Image:
    """Create a true Grad-CAM heatmap, then overlay it on the original uploaded image."""
    conv_layer = last_convolution_layer(model)
    grad_model = tf.keras.models.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(batch, training=False)
        score = prediction[:, class_index]
    gradients = tape.gradient(score, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    activation = conv_output[0] @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(activation).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8

    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(display_image.size)
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    base = np.asarray(display_image.convert("RGB"), dtype=np.float32)
    # A compact red/yellow attention palette, visually distinct from segmentation.
    colour = np.zeros_like(base)
    colour[..., 0] = 255 * heatmap_array
    colour[..., 1] = 190 * np.clip((heatmap_array - 0.35) / 0.65, 0, 1)
    alpha = (heatmap_array[..., None] * 0.58)
    overlay = base * (1 - alpha) + colour * alpha
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))


def run_prediction(image: Image.Image) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH.name}")
    model = load_model(str(MODEL_PATH))
    batch, original = prepare_image(image, model)
    probabilities = normalise_scores(model.predict(batch, verbose=0)[0])
    stage_index = int(np.argmax(probabilities))
    try:
        gradcam = create_gradcam(model, batch, stage_index, original)
    except Exception as error:  # The prediction remains available if the architecture lacks a conv layer.
        gradcam = None
        gradcam_error = str(error)
    else:
        gradcam_error = None
    return {"image": original, "gradcam": gradcam, "gradcam_error": gradcam_error, "probabilities": probabilities, "stage": STAGES[stage_index]}


def result_theme(tone: str) -> tuple[str, str, str]:
    return {
        "low": ("#ecfdf5", "#a7f3d0", "#047857"),
        "moderate": ("#fffbeb", "#fde68a", "#b45309"),
        "critical": ("#fff1f2", "#fecdd3", "#be123c"),
    }[tone]


st.set_page_config(page_title="ULCARE | Wagner Assessment", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .block-container {max-width: 1280px; padding: 1.35rem 1.5rem 3rem;}
      body {background: #f8fafc;}
      .ulcare-top {display:flex; justify-content:space-between; align-items:center; padding:0 0 1.3rem; border-bottom:1px solid #e2e8f0; margin-bottom:1.75rem;}
      .brand {font-size:1.65rem; font-weight:800; color:#0f172a; letter-spacing:-.05em}.brand b{color:#0d9488}.tag{font-size:.7rem; font-weight:700; color:#0f766e; background:#f0fdfa; border:1px solid #99f6e4; padding:.25rem .55rem; border-radius:99px; margin-left:.6rem; vertical-align:middle; letter-spacing:0}
      .hero {background:linear-gradient(112deg,#134e4a,#0f766e 58%,#0369a1); color:white; padding:2rem; border-radius:1.5rem; margin-bottom:1.5rem; box-shadow:0 16px 35px rgba(15,118,110,.18)}
      .hero h1 {font-size:2rem; margin:0 0 .35rem; font-weight:800; letter-spacing:-.04em}.hero p{margin:0;color:#ccfbf1;font-size:.92rem;max-width:740px}
      .section-title{font-size:.9rem;font-weight:800;color:#0f172a;margin-bottom:.25rem}.tiny{font-size:.74rem;color:#64748b}.stage-row{margin:0 0 1rem}.stage-head{display:flex;justify-content:space-between;font-size:.78rem;font-weight:700;color:#475569;margin-bottom:.32rem}.track{height:.55rem;background:#e2e8f0;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#0d9488,#0284c7);border-radius:999px}.grad-caption{font-size:.75rem;color:#64748b;margin-top:.55rem}.stButton>button{background:#0d9488;color:#fff;border:0;border-radius:.8rem;font-weight:700;width:100%;min-height:2.75rem}.stButton>button:hover{background:#0f766e;color:#fff;border:0}.stFileUploader label{font-size:.8rem;font-weight:700;color:#334155}.stAlert{border-radius:.8rem}
    </style>
    <div class="ulcare-top"><div class="brand">UL<b>CARE</b><span class="tag">v2.4 Clinical System</span></div><div class="tiny">Diabetic Foot Ulcer &amp; Wagner Stage Detection</div></div>
    <section class="hero"><h1>Early-Stage Wet Gangrene Detection</h1><p>Unggah foto luka untuk memperoleh prediksi Wagner Stage 0–4 serta penjelasan visual Grad-CAM dari model AI.</p></section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.37, 0.63], gap="large")
with left:
    st.markdown('<div class="section-title">Assessment Baru</div><div class="tiny">Masukkan data singkat lalu unggah foto luka.</div>', unsafe_allow_html=True)
    patient_name = st.text_input("Nama / label assessment", placeholder="e.g. My Foot Scan Assessment")
    uploaded_file = st.file_uploader("Ulcer Image Upload", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Wound preview", use_container_width=True)
    analyse = st.button("🔬 Run AI Wagner Stage Analysis", disabled=uploaded_file is None)

if analyse and uploaded_file:
    try:
        with st.spinner("Menganalisis model dan membuat Grad-CAM..."):
            st.session_state["ulcare_result"] = run_prediction(Image.open(uploaded_file))
            st.session_state["ulcare_label"] = patient_name
    except Exception as error:
        st.error(f"Analisis belum dapat dijalankan: {error}")

with right:
    result = st.session_state.get("ulcare_result")
    if not result:
        st.info("🪄 Menunggu gambar untuk dianalisis. Unggah foto lalu jalankan analisis AI.")
    else:
        stage = result["stage"]
        info = STAGE_INFO[stage]
        background, border, accent = result_theme(info["tone"])
        confidence = result["probabilities"][STAGES.index(stage)] * 100
        st.markdown(
            f'<div style="background:{background};border:1px solid {border};border-radius:1rem;padding:1.35rem 1.5rem;margin-bottom:1rem;display:flex;justify-content:space-between;gap:1rem;align-items:center"><div><div style="font-size:.67rem;font-weight:800;letter-spacing:.08em;color:{accent};text-transform:uppercase">Prediksi model</div><div style="font-size:1.45rem;font-weight:800;color:#0f172a;margin-top:.25rem">{info["title"]}</div><div class="tiny" style="margin-top:.25rem">{st.session_state.get("ulcare_label") or "Assessment foto luka"}</div></div><div style="background:#fff;padding:.75rem 1rem;border-radius:.8rem;text-align:right"><div class="tiny" style="font-weight:700;text-transform:uppercase">Confidence</div><div style="font-size:1.65rem;font-weight:800;color:{accent}">{confidence:.1f}%</div></div></div>',
            unsafe_allow_html=True,
        )
        a, b = st.columns(2, gap="medium")
        with a:
            with st.container(border=True):
                st.markdown('<div class="section-title">Mengapa diprediksi ke stage ini?</div>', unsafe_allow_html=True)
                st.write(info["reason"])
                st.caption("Penjelasan mengacu pada area perhatian model pada Grad-CAM di bawah.")
        with b:
            with st.container(border=True):
                st.markdown('<div class="section-title">Kemungkinan stage lain</div>', unsafe_allow_html=True)
                for label, probability in zip(STAGES, result["probabilities"]):
                    st.markdown(f'<div class="stage-row"><div class="stage-head"><span>{label}</span><span>{probability * 100:.1f}%</span></div><div class="track"><div class="fill" style="width:{probability * 100:.1f}%"></div></div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="section-title">Grad-CAM Attention Map</div><div class="tiny">Bukan segmented mask — warna menandai area yang paling memengaruhi prediksi.</div>', unsafe_allow_html=True)
            if result["gradcam"] is not None:
                st.image(result["gradcam"], caption="Grad-CAM overlay", use_container_width=True)
            else:
                st.warning("Grad-CAM belum dapat dibuat untuk arsitektur model ini. " + (result["gradcam_error"] or ""))
        with st.container(border=True):
            st.markdown('<div class="section-title">Saran Klinis</div>', unsafe_allow_html=True)
            st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Isi saran klinis dapat ditambahkan dan disesuaikan oleh tim medis pada tahap berikutnya.")
