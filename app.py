"""ULCARE Streamlit backend: predicts Wagner Stage 0–4 and builds a real Grad-CAM from the trained model."""

from __future__ import annotations

import base64
import contextlib
import io
import os
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_gangren_wagner_v1.h5"
FRONTEND = ROOT / "frontend"          # must contain index.html
MAX_SIDE = 768                         # longest side of the image sent back to the browser

# Must match how the model was TRAINED (the .h5 has no Rescaling layer inside):
#   "rescale"      -> pixel / 255            (ImageDataGenerator(rescale=1/255))
#   "mobilenet_v2" -> pixel / 127.5 - 1      (mobilenet_v2.preprocess_input)
PREPROCESS = os.getenv("ULCARE_PREPROCESS", "rescale").lower()


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def decode_data_uri(data_uri: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def prepare_image(image: Image.Image, model: tf.keras.Model) -> np.ndarray:
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    height, width = int(shape[1] or 224), int(shape[2] or 224)
    pixels = np.asarray(image.resize((width, height)), dtype=np.float32)
    pixels = pixels / 127.5 - 1.0 if PREPROCESS == "mobilenet_v2" else pixels / 255.0
    return pixels[None, ...]


def split_model(model: tf.keras.Model):
    """Return (feature_extractor, head_layers).

    This model is `Input -> MobileNetV2 (nested Functional) -> GAP -> Dropout -> Dense`, so the
    last Conv2D lives INSIDE the nested model and cannot be reached via `model.layers`.
    We therefore run the backbone to get its last feature map (out_relu, 7x7x1280) and replay the
    head layers on top of it, which keeps the gradient path fully connected.
    """
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            head = [l for l in model.layers[i + 1:] if not isinstance(l, tf.keras.layers.InputLayer)]
            return layer, head
    for i in range(len(model.layers) - 1, -1, -1):          # flat model fallback
        if len(model.layers[i].output.shape) == 4:
            extractor = tf.keras.Model(model.inputs, model.layers[i].output)
            return extractor, list(model.layers[i + 1:])
    raise ValueError("Tidak ditemukan feature map konvolusi untuk membuat Grad-CAM.")


def forward_with_cam(model: tf.keras.Model, batch: np.ndarray):
    """One forward pass -> (probabilities[5], predicted stage, raw Grad-CAM 2D array)."""
    backbone, head = split_model(model)
    with tf.GradientTape() as tape:
        fmap = backbone(tf.convert_to_tensor(batch), training=False)
        tape.watch(fmap)
        x = fmap
        for layer in head[:-1]:
            x = layer(x)
        last = head[-1]
        if isinstance(last, tf.keras.layers.Dense) and getattr(last.activation, "__name__", "") == "softmax":
            logits = tf.matmul(x, tf.convert_to_tensor(last.kernel))
            if last.bias is not None:
                logits = logits + tf.convert_to_tensor(last.bias)
            probs = tf.nn.softmax(logits)          # prediction
        else:
            logits = probs = last(x)
        stage = int(tf.argmax(probs[0]))
        score = logits[0, stage]                   # pre-softmax score => cleaner, class-specific map
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(fmap[0] * weights, axis=-1).numpy()
    return probs[0].numpy().astype(np.float64), stage, cam


def jet(values: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255.0


def overlay(original: Image.Image, cam: np.ndarray) -> Image.Image:
    cam = np.maximum(cam, 0)
    if cam.max() <= 1e-8:                          # all-negative map: fall back to magnitude
        cam = np.abs(cam)
    cam = cam / (cam.max() + 1e-8)
    heat = Image.fromarray(cam.astype(np.float32)).resize(original.size, Image.BICUBIC)
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0, 1)
    base = np.asarray(original, dtype=np.float32)
    alpha = (0.6 * heat ** 0.7)[..., None]         # low attention stays transparent
    return Image.fromarray(np.uint8(np.clip(base * (1 - alpha) + jet(heat) * alpha, 0, 255)))


def jpeg_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def predict(data_uri: str) -> dict:
    model = load_model()
    original = decode_data_uri(data_uri)
    probs, stage, cam = forward_with_cam(model, prepare_image(original, model))
    return {
        "stage": stage,
        "probabilities": [round(float(p) * 100, 2) for p in probs],
        "gradcam": jpeg_uri(overlay(original, cam)),
    }


# --------------------------------------------------------------------------- page
st.set_page_config(page_title="ULCARE", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """<style>
    #MainMenu,header,footer,
    [data-testid='stToolbar'],
    [data-testid='stStatusWidget'],
    [data-testid='stDecoration'] {
        display: none !important;
    }

    /* Jangan mengunci scrolling vertikal Streamlit. */
    html, body,
    [data-testid='stApp'],
    [data-testid='stAppViewContainer'],
    [data-testid='stMain'],
    section.main {
        overflow-x: hidden !important;
        background: #f8fafc;
    }

    .block-container {
        padding: 0 !important;
        max-width: none !important;
    }

    [data-testid='stVerticalBlock'] {
        gap: 0 !important;
    }

    iframe {
        display: block;
        width: 100%;
        max-width: 100%;
        border: 0;
    }
    </style>""",
    unsafe_allow_html=True,
)


if not (FRONTEND / "index.html").exists():
    st.error("File `frontend/index.html` tidak ditemukan. Pindahkan `index.html` ke dalam folder `frontend/` di samping `app.py`.")
    st.stop()

component = components.declare_component("ulcare_interface", path=str(FRONTEND))
incoming = component(result=st.session_state.get("model_result"), key="ulcare", default=None)

if (
    isinstance(incoming, dict)
    and incoming.get("action") == "predict"
    and incoming.get("id") != st.session_state.get("processed_id")
):
    st.session_state["processed_id"] = incoming["id"]
    try:
        outcome = predict(incoming["image"])
    except Exception as error:  # reported to the UI so the button never hangs
        traceback.print_exc()
        outcome = {"error": f"{type(error).__name__}: {error}"}
    outcome["id"] = incoming["id"]
    st.session_state["model_result"] = outcome
    st.rerun()

# Warm the model right after the UI is on screen so the first prediction is fast.
with contextlib.suppress(Exception):
    load_model()
