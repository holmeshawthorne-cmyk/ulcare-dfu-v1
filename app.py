"""ULCARE Streamlit backend: predicts Wagner Stage 0–4 and creates Grad-CAM."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_gangren_wagner_v1.h5"
COMPONENT = components.declare_component("ulcare_interface", path=str(ROOT / "frontend"))
STAGE_REASONS = [
    "Model terutama memperhatikan area tanpa ulkus terbuka yang dominan, sehingga pola visualnya paling sesuai dengan Stage 0.",
    "Model memberi perhatian terbesar pada perubahan permukaan luka yang konsisten dengan ulkus superfisial Stage 1.",
    "Area Grad-CAM terkonsentrasi pada dasar dan tepi luka yang mendukung keterlibatan jaringan lebih dalam, sesuai Stage 2.",
    "Pola perhatian pada eksudat, perubahan warna, dan jaringan sekitar luka paling konsisten dengan infeksi dalam pada Stage 3.",
    "Area dengan perubahan warna dan kondisi jaringan paling kuat memengaruhi prediksi gangren lokal pada Stage 4.",
]


@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def decode_data_uri(data_uri: str) -> Image.Image:
    encoded = data_uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def prepare_image(image: Image.Image, model: tf.keras.Model) -> np.ndarray:
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    height, width = int(shape[1] or 224), int(shape[2] or 224)
    resized = image.resize((width, height))
    return np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, axis=0)


def probabilities(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32).squeeze().reshape(-1)
    if values.size < 5:
        raise ValueError("Model harus mengeluarkan lima kelas: Wagner Stage 0–4.")
    values = values[:5]
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=0.02):
        values = np.exp(values - values.max())
        values /= values.sum()
    return values / values.sum()


def last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("Tidak ditemukan layer Conv2D untuk membuat Grad-CAM.")


def gradcam(model: tf.keras.Model, batch: np.ndarray, target: int, original: Image.Image) -> Image.Image:
    conv = last_conv_layer(model)
    grad_model = tf.keras.Model(model.inputs, [conv.output, model.output])
    with tf.GradientTape() as tape:
        feature_maps, output = grad_model(batch, training=False)
        target_score = output[:, target]
    gradients = tape.gradient(target_score, feature_maps)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.squeeze(feature_maps[0] @ weights[..., tf.newaxis]).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    heatmap = np.asarray(Image.fromarray(np.uint8(heatmap * 255)).resize(original.size), dtype=np.float32) / 255.0

    base = np.asarray(original, dtype=np.float32)
    colour = np.zeros_like(base)
    colour[..., 0] = 255 * heatmap
    colour[..., 1] = 210 * np.clip((heatmap - .25) / .75, 0, 1)
    alpha = heatmap[..., None] * .58
    return Image.fromarray(np.uint8(np.clip(base * (1 - alpha) + colour * alpha, 0, 255)))


def image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def predict(data_uri: str) -> dict:
    model = load_model()
    original = decode_data_uri(data_uri)
    batch = prepare_image(original, model)
    scores = probabilities(model.predict(batch, verbose=0)[0])
    stage = int(np.argmax(scores))
    return {
        "stage": stage,
        "confidence": round(float(scores[stage] * 100), 1),
        "probabilities": [round(float(score * 100), 1) for score in scores],
        "reason": STAGE_REASONS[stage],
        "image": image_data_uri(original),
        "gradcam": image_data_uri(gradcam(model, batch, stage, original)),
    }


st.set_page_config(page_title="ULCARE", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>#MainMenu,header,footer,[data-testid='stToolbar'],[data-testid='stStatusWidget']{display:none!important}.block-container{padding:0!important;max-width:none!important}[data-testid='stAppViewContainer']{background:#f8fafc}</style>", unsafe_allow_html=True)

incoming = COMPONENT(result=st.session_state.get("model_result"), key="ulcare")
request_id = hash(incoming.get("image", "")) if isinstance(incoming, dict) else None
if (
    isinstance(incoming, dict)
    and incoming.get("action") == "predict"
    and request_id != st.session_state.get("processed_request")
):
    try:
        with st.spinner("Menganalisis model dan membuat Grad-CAM..."):
            st.session_state["processed_request"] = request_id
            st.session_state["model_result"] = predict(incoming["image"])
        st.rerun()
    except Exception as error:
        st.session_state["model_result"] = {"error": str(error)}
        st.rerun()
