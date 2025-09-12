# app.py — MediSkin (Streamlit)
import os, json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# -------------------------------- PAGE CONFIG
st.set_page_config(page_title="MediSkin – Monkeypox Screening", page_icon="🩺", layout="centered")

# -------------------------------- PATHS (repo-root defaults; allow env override)
DZ_MODEL_PATH = Path(os.getenv("DZ_MODEL_PATH", "disease_mnv2.h5"))        # weights only
DZ_IDX_JSON   = Path(os.getenv("DZ_IDX_JSON",  "disease_class_indices.json"))

# -------------------------------- THEME
st.markdown("""
<style>
.stApp { background:#d1dff6; }
.main { background:rgba(255,255,255,.95); padding:2rem; border-radius:16px; max-width:860px; margin:auto; box-shadow:0 4px 20px rgba(0,0,0,.25); }
.meter { height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.meter>span { display:block; height:100%; background:#2563eb; }
</style>
""", unsafe_allow_html=True)

# -------------------- TEXT (EN / BM) -----------------
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
        "chat_title": "💬 MediSkin Chatbot (FAQ)",
        "chat_hint": "Ask about monkeypox"
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
        "chat_title": "💬 MediSkin Chatbot  (FAQ)",
        "chat_hint": "Tanya tentang monkeypox"
    }
}

ADVICE = {
    "en": {
        "monkeypox": "🛑 Avoid close contact, cover lesions, and seek medical attention immediately.",
        "normal": "✅ No concerning lesion detected. Maintain hygiene and monitor regularly.",
        "others": "ℹ️ This is not typical of monkeypox. Consider a clinic consultation if the condition persists."
    },
    "bm": {
        "monkeypox": "🛑 Elakkan sentuhan rapat, tutup luka, dan dapatkan rawatan perubatan dengan segera.",
        "normal": "✅ Tiada luka membimbangkan dikesan. Kekalkan kebersihan dan pantau keadaan kulit.",
        "others": "ℹ️ Ini bukan tipikal monkeypox. Pertimbangkan pergi ke klinik jika keadaan berlarutan."
    }
}

DISPLAY_NAME = {"other_disease": "Others"}

# -------------------- MODEL (build clean graph + load weights) ----------
# Set to the size you trained with; if unsure and you used MobileNetV2 defaults, 224 is typical.
IMG_SIZE = (224, 224)

@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    # ---- Load labels
    if not DZ_IDX_JSON.exists():
        st.error(f"Missing file: {DZ_IDX_JSON}")
        st.stop()
    with open(DZ_IDX_JSON, "r", encoding="utf-8") as f:
        idx = json.load(f)  # e.g. {"monkeypox":0,"normal":1,"other_disease":2}
    labels = [c for c, _ in sorted(idx.items(), key=lambda x: x[1])]
    num_classes = len(labels)

    # ---- Build a SINGLE-input MobileNetV2 classifier
    inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input")
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights=None, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="mnv2"
    )
    x = base(inp, training=False)                 # IMPORTANT: call once → one tensor forward
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="cls")(x)
    model = tf.keras.Model(inp, out, name="mnv2_classifier")

    # ---- Load weights ONLY (avoid loading a broken graph)
    if not DZ_MODEL_PATH.exists():
        st.error(f"Model weights not found: {DZ_MODEL_PATH}")
        st.stop()
    try:
        model.load_weights(str(DZ_MODEL_PATH), by_name=True, skip_mismatch=True)
    except Exception as e:
        st.error(f"Failed to load weights from '{DZ_MODEL_PATH}': {e}")
        st.stop()

    return model, labels

def preprocess(img_pil: Image.Image):
    rgb = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(rgb, dtype="float32")
    # If you trained with /255, keep this; if you used tf.keras.applications preprocessing, swap accordingly.
    x = x / 255.0
    return np.expand_dims(x, 0)

# -------------------- Optional chatbot (unchanged) ----------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    FAQ_PAIRS = [
        ("what is monkeypox",
         "Monkeypox is a viral zoonotic disease caused by the monkeypox virus. It can spread from animals to humans and between people."),
        ("what are symptoms of monkeypox",
         "Common symptoms include fever, headache, swollen lymph nodes, muscle aches, followed by a skin rash or lesions that may resemble smallpox or chickenpox."),
        ("how does monkeypox spread",
         "Monkeypox spreads through close contact with lesions, body fluids, respiratory droplets, or contaminated objects like bedding."),
        ("is monkeypox dangerous",
         "Monkeypox is usually self-limiting, lasting 2–4 weeks. Severe cases can occur, especially in children or immunocompromised people."),
        ("is there a treatment for monkeypox",
         "There is no specific treatment. Supportive care is given, and smallpox vaccines or antivirals may help in some cases."),
        ("how to prevent monkeypox",
         "Prevention includes avoiding close contact with infected individuals, practicing good hand hygiene, and using protective equipment when caring for patients.")
    ]
    _VEC = TfidfVectorizer().fit([q for q, _ in FAQ_PAIRS])
    _CORP = _VEC.transform([q for q, _ in FAQ_PAIRS])

    def bot_reply(user_text: str) -> str:
        if not user_text.strip():
            return "Ask me about monkeypox"
        q = _VEC.transform([user_text.lower()])
        sims = cosine_similarity(q, _CORP)[0]
        i = int(np.argmax(sims))
        return FAQ_PAIRS[i][1] if sims[i] >= 0.2 else "I’m not sure."

    CHATBOT_AVAILABLE = True
except Exception:
    CHATBOT_AVAILABLE = False

# -------------------- APP ----------------------------
lang = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Melayu"], index=0)
L = "en" if lang.startswith("English") else "bm"

if CHATBOT_AVAILABLE:
    st.sidebar.markdown(f"### {TEXT[L]['chat_title']}")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "Hi! " + TEXT[L]["chat_hint"]}]
    for m in st.session_state.chat:
        st.sidebar.chat_message(m["role"]).write(m["content"])
    user_msg = st.sidebar.chat_input(TEXT[L]["chat_hint"])
    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        st.session_state.chat.append({"role": "assistant", "content": bot_reply(user_msg)})
        st.rerun()
else:
    st.sidebar.markdown("_FAQ chatbot requires `scikit-learn` (run `pip install scikit-learn`)._")

model, DZ_LABELS = load_model_and_labels()
DZ_DISPLAY = [DISPLAY_NAME.get(lbl, lbl) for lbl in DZ_LABELS]

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

        st.markdown("---")
        st.subheader(TEXT[L]["advice"])
        st.info(ADVICE[L].get(label_raw.lower(), "—"))

        st.caption(TEXT[L]["disclaimer"])
else:
    st.info(TEXT[L]["noimg"])

st.markdown("</div>", unsafe_allow_html=True)

