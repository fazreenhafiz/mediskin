import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# ---------- Paths (repo-relative) ----------
ROOT = Path(__file__).parent
DZ_MODEL_PATH = ROOT / "disease_mnv2.h5"           # FULL model
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
    # ----- read labels first -----
    if not DZ_IDX_JSON.exists():
        st.error(f"Missing file: {DZ_IDX_JSON}")
        st.stop()
    with open(DZ_IDX_JSON) as f:
        idx = json.load(f)  # {"monkeypox":0, "normal":1, "other_disease":2}
    labels = [c for c, _ in sorted(idx.items(), key=lambda x: x[1])]
    num_classes = len(labels)

    # ----- 1) Try loading a FULL model (.h5 or SavedModel dir) -----
    if DZ_MODEL_PATH.exists() and DZ_MODEL_PATH.suffix == ".h5":
        try:
            model = load_model(str(DZ_MODEL_PATH))
            return model, labels
        except Exception as e:
            st.warning(f"Full-model load failed, will try rebuild+weights. Details: {e}")

    if DZ_MODEL_PATH.is_dir():
        try:
            model = load_model(str(DZ_MODEL_PATH))
            return model, labels
        except Exception as e:
            st.warning(f"SavedModel load failed, will try rebuild+weights. Details: {e}")

    # ----- 2) Rebuild EXACT inference graph and load weights only -----
    # IMPORTANT: do NOT pass input_tensor into the application model; call it explicitly.
    inp = tf.keras.Input(shape=(224, 224, 3), name="input")
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None,           # we will load weights from file
        input_shape=(224, 224, 3)
    )
    feats = backbone(inp, training=False)  # single call -> single input
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(feats)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)
    model = tf.keras.Model(inp, out, name="mnv2_classifier")

    if DZ_MODEL_PATH.exists():
        try:
            # If the file contains only some layers, skip mismatches to avoid graph errors.
            model.load_weights(str(DZ_MODEL_PATH), by_name=True, skip_mismatch=True)
            return model, labels
        except Exception as e:
            st.error(
                "Could not load weights into the rebuilt model.\n\n"
                f"Path: {DZ_MODEL_PATH}\nError: {e}\n\n"
                "Make sure the file is either a FULL model (.h5 or SavedModel) "
                "or the exact backbone+head weights."
            )
            st.stop()
    else:
        st.error(f"Model file/folder not found: {DZ_MODEL_PATH}")
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


