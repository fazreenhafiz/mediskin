import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# ---------- Paths (repo-relative) ----------
ROOT = Path(__file__).parent
DZ_MODEL_PATH = ROOT / "disease_mnv2_full.h5"           # FULL model
DZ_IDX_JSON   = ROOT / "disease_class_indices.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="MediSkin – Monkeypox Screening", page_icon="🩺", layout="centered")

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

@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    if not DZ_MODEL_PATH.exists():
        st.error(f"Model file not found: {DZ_MODEL_PATH}")
        st.stop()
    if not DZ_IDX_JSON.exists():
        st.error(f"Class index file not found: {DZ_IDX_JSON}")
        st.stop()

    model = load_model(str(DZ_MODEL_PATH))  # FULL model
    with open(DZ_IDX_JSON) as f:
        idx = json.load(f)  # {"monkeypox":0, "normal":1, "other_disease":2}
    labels = [c for c, _ in sorted(idx.items(), key=lambda x: x[1])]
    return model, labels

def prep(img: Image.Image):
    x = np.array(img.convert("RGB").resize(IMG_SIZE)).astype("float32") / 255.0
    return np.expand_dims(x, 0)

# -------- UI --------
lang = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Melayu"], index=0)
L = "en" if lang.startswith("English") else "bm"

st.title(TEXT[L]["title"])
st.caption(TEXT[L]["subtitle"])

uploaded = st.file_uploader(TEXT[L]["uploader"], type=["jpg", "jpeg", "png"])
model, DZ_LABELS = load_model_and_labels()

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    if st.button(TEXT[L]["btn"], type="primary"):
        with st.spinner(TEXT[L]["running"]):
            probs = model.predict(prep(img), verbose=0)[0]
            j = int(np.argmax(probs))
            label_raw = DZ_LABELS[j]
            label = DISPLAY_NAME.get(label_raw, label_raw)
            conf = float(probs[j])

        st.subheader(TEXT[L]["result"])
        st.write(f"**{label}** — confidence **{conf*100:.1f}%**")
        st.progress(conf)
        st.subheader(TEXT[L]["advice"])
        st.info(ADVICE[L].get(label_raw.lower(), "—"))
        st.caption(TEXT[L]["disclaimer"])
else:
    st.info(TEXT[L]["noimg"])
