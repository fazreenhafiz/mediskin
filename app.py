# app.py — MediSkin (Streamlit) — aligned with training
import os, json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="MediSkin – Monkeypox Screening", page_icon="🩺", layout="centered")

# ---------- PATHS ----------
DZ_MODEL_PATH = Path(os.getenv("DZ_MODEL_PATH", "disease_mnv2.h5"))        # full model .h5
DZ_IDX_JSON   = Path(os.getenv("DZ_IDX_JSON",  "disease_class_indices.json"))

# ---------- THEME ----------
st.markdown("""
<style>
.stApp { background:#d1dff6; }
.main { background:rgba(255,255,255,.95); padding:2rem; border-radius:16px; max-width:860px; margin:auto; box-shadow:0 4px 20px rgba(0,0,0,.25); }
.meter { height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.meter>span { display:block; height:100%; background:#2563eb; }
</style>
""", unsafe_allow_html=True)

# ---------- TEXT ----------
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
    }
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
    }
}
DISPLAY_NAME = {"other_disease": "Others"}

# ---------- MODEL / PREPROCESS ----------
IMG_SIZE = (224, 224)  # matches your training

@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    # Load labels
    if not DZ_IDX_JSON.exists():
        st.error(f"Missing file: {DZ_IDX_JSON}"); st.stop()
    with open(DZ_IDX_JSON, "r", encoding="utf-8") as f:
        idx = json.load(f)  # {"monkeypox":0, "normal":1, "other_disease":2}
    labels = [c for c, _ in sorted(idx.items(), key=lambda x: x[1])]

    # Load FULL model (same as your eval script)
    if not DZ_MODEL_PATH.exists():
        st.error(f"Model file not found: {DZ_MODEL_PATH}"); st.stop()
    model = tf.keras.models.load_model(str(DZ_MODEL_PATH))

    # Optional sanity check
    out_units = model.outputs[0].shape[-1]
    if out_units is not None and int(out_units) != len(labels):
        st.warning(f"Model output units ({int(out_units)}) != labels count ({len(labels)}).")
    return model, labels

def preprocess(img_pil: Image.Image):
    rgb = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(rgb, dtype="float32") / 255.0   # same as ImageDataGenerator(rescale=1./255)
    return np.expand_dims(x, 0)

# ---------- APP ----------
lang = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Melayu"], index=0)
L = "en" if lang.startswith("English") else "bm"

model, DZ_LABELS = load_model_and_labels()

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title(TEXT[L]["title"])
st.caption(TEXT[L]["subtitle"])

uploaded = st.file_uploader(TEXT[L]["uploader"], type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button(TEXT[L]["btn"], type="primary"):
        with st.spinner(TEXT[L]["running"]):
            x = preprocess(img)
            probs = model.predict(x, verbose=0)[0]
            j = int(np.argmax(probs))
            label_raw = DZ_LABELS[j]
            label = DISPLAY_NAME.get(label_raw, label_raw)
            conf = float(probs[j])

        st.subheader(TEXT[L]["result"])
        st.markdown(f"**{label}** — confidence **{conf*100:.1f}%**")
        st.markdown(f"<div class='meter'><span style='width:{conf*100:.1f}%'></span></div>", unsafe_allow_html=True)

        # Debug: show all class probabilities (remove later)
        st.write({lbl: float(p) for lbl, p in zip(DZ_LABELS, probs)})

        st.markdown("---")
        st.subheader(TEXT[L]["advice"])
        st.info(ADVICE[L].get(label_raw.lower(), "—"))

        st.caption(TEXT[L]["disclaimer"])
else:
    st.info(TEXT[L]["noimg"])

st.markdown("</div>", unsafe_allow_html=True)
