import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =============================================================================
# Page & constants
# =============================================================================
st.set_page_config(page_title="MediSkin – Monkeypox Screening", page_icon="🩺", layout="centered")

# File locations (relative to repo root)
IDX_JSON = Path("disease_class_indices.json")

# Preferred export locations (choose one path to commit to your repo)
SAVEDMODEL_DIR = Path("export/mediskin_savedmodel")      # <---- SavedModel folder
ARCH_JSON      = Path("export/architecture.json")        # <---- Architecture JSON
WEIGHTS_H5     = Path("export/mediskin_full.weights.h5") # <---- Full weights
FULL_H5        = Path("disease_mnv2.h5")                 # <---- Full model .h5 (NOT head-only)

IMG_SIZE = (224, 224)

# =============================================================================
# Styling
# =============================================================================
st.markdown(
    """
    <style>
      .stApp { background-color: #d1dff6; }
      .main {
        background-color: rgba(255,255,255,0.95);
        padding: 2rem; border-radius: 16px; max-width: 860px; margin: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,.25);
      }
      .meter { height: 10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
      .meter>span { display:block; height:100%; background:#2563eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Text resources
# =============================================================================
TEXT = {
    "en": {
        "title": "🧬 MediSkin – Monkeypox Screening",
        "subtitle": "Upload a skin image for rapid AI-based screening. This is not a substitute for medical advice.",
        "uploader": "📤 Upload a skin image (JPG/PNG)",
        "btn": "Analyze",
        "result": "Result",
        "advice": "Advice",
        "disclaimer": "Disclaimer: This tool provides screening support only. Please consult healthcare professionals for diagnosis.",
        "noimg": "Upload an image to start.",
        "running": "Running inference...",
    },
    "bm": {
        "title": "🧬 MediSkin – Saringan Monkeypox",
        "subtitle": "Muat naik imej kulit untuk saringan AI pantas. Ini bukan pengganti nasihat perubatan.",
        "uploader": "📤 Muat naik imej kulit (JPG/PNG)",
        "btn": "Analisis",
        "result": "Keputusan",
        "advice": "Nasihat",
        "disclaimer": "Penafian: Alat ini hanya sokongan saringan. Sila rujuk profesional kesihatan untuk diagnosis.",
        "noimg": "Muat naik imej untuk bermula.",
        "running": "Model sedang dijalankan...",
    },
}

ADVICE = {
    "en": {
        "monkeypox": "🛑 Avoid close contact, cover lesions, and seek medical attention immediately.",
        "normal": "✅ No concerning lesion detected. Maintain hygiene and monitor regularly.",
        "other_disease": "ℹ️ This is not typical of monkeypox. Consider a clinic consultation if the condition persists."
    },
    "bm": {
        "monkeypox": "🛑 Elakkan sentuhan rapat, tutup luka, dan dapatkan rawatan perubatan dengan segera.",
        "normal": "✅ Tiada luka membimbangkan dikesan. Kekalkan kebersihan dan pantau keadaan kulit.",
        "other_disease": "ℹ️ Ini bukan tipikal monkeypox. Pertimbangkan pergi ke klinik jika keadaan berlarutan."
    },
}

DISPLAY_NAME = {"other_disease": "Others"}  # Pretty label in English UI

# =============================================================================
# Helpers
# =============================================================================
def _load_labels():
    if not IDX_JSON.exists():
        st.error(f"`{IDX_JSON}` not found. Make sure it is committed to the repo.")
        st.stop()
    with open(IDX_JSON) as f:
        idx = json.load(f)  # e.g., {"monkeypox":0, "normal":1, "other_disease":2}
    labels = [c for c, _ in sorted(idx.items(), key=lambda x: x[1])]
    return labels

def _load_model_for_inference():
    """
    Load the Keras model using one of the supported formats:
      1) SavedModel folder (export/mediskin_savedmodel/)
      2) architecture.json + full weights (.weights.h5)
      3) full single .h5 model (disease_mnv2.h5)

    IMPORTANT: A 'head-only' .h5 will NOT work. You must export either the full model
    or the architecture JSON + full weights, or a SavedModel folder.
    """
    # 1) SavedModel directory
    if SAVEDMODEL_DIR.exists() and SAVEDMODEL_DIR.is_dir():
        try:
            model = tf.keras.models.load_model(SAVEDMODEL_DIR, compile=False)
            return model
        except Exception as e:
            st.warning(f"Found SavedModel at {SAVEDMODEL_DIR} but failed to load: {e}")

    # 2) Architecture JSON + full weights
    if ARCH_JSON.exists() and WEIGHTS_H5.exists():
        try:
            from tensorflow.keras.models import model_from_json
            with open(ARCH_JSON) as f:
                model = model_from_json(f.read())
            model.load_weights(str(WEIGHTS_H5))
            return model
        except Exception as e:
            st.warning(f"Found JSON+weights but failed to load: {e}")

    # 3) Single full .h5 model
    if FULL_H5.exists():
        try:
            # Works only if FULL_H5 is a FULL model (not head-only)
            model = tf.keras.models.load_model(FULL_H5, compile=False)
            return model
        except Exception as e:
            st.warning(f"Found {FULL_H5} but failed to load as a full model: {e}")

    # If nothing worked:
    st.error(
        "No valid model found.\n\n"
        "Please commit ONE of the following:\n"
        "1) A SavedModel folder at `export/mediskin_savedmodel/`, OR\n"
        "2) `export/architecture.json` + `export/mediskin_full.weights.h5`, OR\n"
        "3) A FULL model H5 at `disease_mnv2.h5` (not just the classifier head)."
    )
    st.stop()

@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    labels = _load_labels()
    model = _load_model_for_inference()
    return model, labels

def prep(img_pil: Image.Image):
    """Resize to 224x224 and apply MobileNetV2 preprocessing."""
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")
    x = preprocess_input(x)           # MobileNetV2 preprocessing
    return np.expand_dims(x, 0)

# =============================================================================
# UI
# =============================================================================
lang = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Melayu"], index=0)
L = "en" if lang.startswith("English") else "bm"

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title(TEXT[L]["title"])
st.caption(TEXT[L]["subtitle"])

uploaded = st.file_uploader(TEXT[L]["uploader"], type=["jpg", "jpeg", "png"])

# Load model + labels once
model, DZ_LABELS = load_model_and_labels()

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button(TEXT[L]["btn"], type="primary"):
        with st.spinner(TEXT[L]["running"]):
            x = prep(img)
            probs = model.predict(x, verbose=0)[0]
            j = int(np.argmax(probs))
            label_raw = DZ_LABELS[j]
            label_show = DISPLAY_NAME.get(label_raw, label_raw)
            conf = float(probs[j])

        st.subheader(TEXT[L]["result"])
        st.markdown(f"**{label_show}** — confidence **{conf*100:.1f}%**")
        st.markdown(
            f"<div class='meter'><span style='width:{conf*100:.1f}%'></span></div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader(TEXT[L]["advice"])
        st.info(ADVICE[L].get(label_raw, "—"))
        st.caption(TEXT[L]["disclaimer"])
else:
    st.info(TEXT[L]["noimg"])

st.markdown("</div>", unsafe_allow_html=True)
