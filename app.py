import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import json
import tensorflow as tf

# -------------------------------- PAGE CONFIG
st.set_page_config(page_title="MediSkin – Monkeypox Screening", page_icon="🩺", layout="centered")

# -------------------------------- PATHS  (files must live in the repo root on Streamlit Cloud)
DZ_MODEL_PATH = Path("disease_mnv2.h5")              # weights file
DZ_IDX_JSON   = Path("disease_class_indices.json")   # {"monkeypox":0, "normal":1, "other_disease":2}
IMG_SIZE = (224, 224)

# -------------------------------- THEME
st.markdown("""
<style>
.stApp { background:#d1dff6; }
.main { background:rgba(255,255,255,.95); padding:2rem; border-radius:16px; max-width:860px; margin:auto; box-shadow:0 4px 20px rgba(0,0,0,.25); }
.meter { height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.meter>span { display:block; height:100%; background:#2563eb; }
</style>
""", unsafe_allow_html=True)

# -------------------------------- TEXT
TEXT = {
    "en": {"title":"🧬 MediSkin – Monkeypox Screening","subtitle":"Upload a skin image for rapid AI-based screening. This is not a substitute for medical advice.","uploader":"📤 Upload a skin image (JPG/PNG)","btn":"Analyze","result":"Result","advice":"Advice","disclaimer":"Disclaimer: This tool provides screening support only. Please consult healthcare professionals for diagnosis.","noimg":"Upload an image to start.","running":"Running inference...","chat_title":"💬 MediSkin Chatbot (FAQ)","chat_hint":"Ask about monkeypox "},
    "bm": {"title":"🧬 MediSkin – Saringan Monkeypox","subtitle":"Muat naik imej kulit untuk saringan AI pantas. Ini bukan pengganti nasihat perubatan.","uploader":"📤 Muat naik imej kulit (JPG/PNG)","btn":"Analisis","result":"Keputusan","advice":"Nasihat","disclaimer":"Penafian: Alat ini hanya sokongan saringan. Sila rujuk profesional kesihatan untuk diagnosis.","noimg":"Muat naik imej untuk bermula.","running":"Model sedang dijalankan...","chat_title":"💬 MediSkin Chatbot  (FAQ)","chat_hint":"Tanya tentang monkeypox"}
}
ADVICE = {
    "en": {"monkeypox":"🛑 Avoid close contact, cover lesions, and seek medical attention immediately.","normal":"✅ No concerning lesion detected. Maintain hygiene and monitor regularly.","other_disease":"ℹ️ This is not typical of monkeypox. Consider a clinic consultation if the condition persists."},
    "bm": {"monkeypox":"🛑 Elakkan sentuhan rapat, tutup luka, dan dapatkan rawatan perubatan dengan segera.","normal":"✅ Tiada luka membimbangkan dikesan. Kekalkan kebersihan dan pantau.","other_disease":"ℹ️ Ini bukan tipikal monkeypox. Pertimbangkan pergi ke klinik jika berlarutan."}
}
DISPLAY_NAME = {"other_disease": "Others"}

# -------------------------------- MODEL LOADER (force rebuild + weights)
@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    # labels
    if not DZ_IDX_JSON.exists():
        st.error(f"Missing file: {DZ_IDX_JSON}")
        st.stop()
    with open(DZ_IDX_JSON) as f:
        idx = json.load(f)
    labels = [c for c, _ in sorted(idx.items(), key=lambda x: x[1])]
    num_classes = len(labels)

    # build a SINGLE-input MobileNetV2 graph
    inp = tf.keras.Input(shape=(224, 224, 3), name="input")
    backbone = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(224,224,3))
    feats = backbone(inp, training=False)              # <-- single call => single input
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(feats)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)
    model = tf.keras.Model(inp, out, name="mnv2_classifier")

    # load weights by name ONLY (skip mismatches), never load full model
    if not DZ_MODEL_PATH.exists():
        st.error(f"Model weights not found: {DZ_MODEL_PATH}")
        st.stop()
    try:
        model.load_weights(str(DZ_MODEL_PATH), by_name=True, skip_mismatch=True)
    except Exception as e:
        st.error(f"Failed to load weights from '{DZ_MODEL_PATH}': {e}")
        st.stop()

    return model, labels

def prep(img_pil: Image.Image):
    rgb = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(rgb).astype("float32") / 255.0
    return np.expand_dims(x, 0)

# -------------------------------- APP
lang = st.sidebar.selectbox("Language / Bahasa", ["English","Bahasa Melayu"], index=0)
L = "en" if lang.startswith("English") else "bm"

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title(TEXT[L]["title"]); st.caption(TEXT[L]["subtitle"])

# load model
model, DZ_LABELS = load_model_and_labels()
DZ_DISPLAY = [DISPLAY_NAME.get(lbl, lbl) for lbl in DZ_LABELS]

uploaded = st.file_uploader(TEXT[L]["uploader"], type=["jpg","jpeg","png"])
if uploaded:
    from PIL import Image
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button(TEXT[L]["btn"], type="primary"):
        with st.spinner(TEXT[L]["running"]):
            probs = model.predict(prep(img), verbose=0)[0]
            j = int(np.argmax(probs)); label_raw = DZ_LABELS[j]
            label = DISPLAY_NAME.get(label_raw, label_raw); conf = float(probs[j])

        st.subheader(TEXT[L]["result"])
        st.markdown(f"**{label}** — confidence **{conf*100:.1f}%**")
        st.markdown(f"<div class='meter'><span style='width:{conf*100:.1f}%'></span></div>", unsafe_allow_html=True)
        st.markdown("---"); st.subheader(TEXT[L]["advice"]); st.info(ADVICE[L].get(label_raw, "—"))
        st.caption(TEXT[L]["disclaimer"])
else:
    st.info(TEXT[L]["noimg"])
st.markdown("</div>", unsafe_allow_html=True)
