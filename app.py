# app.py
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="MediSkin – Monkeypox Screening",
    page_icon="🩺",
    layout="centered"
)

# -------------------- PATH HELPERS -------------------
# We support either:
#   ./disease_mnv2.h5  and ./disease_class_indices.json
# OR
#   ./models/disease_mnv2.h5 and ./models/disease_class_indices.json
def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

DZ_MODEL_PATH = find_first_existing([
    Path("disease_mnv2.h5"),
    Path("models") / "disease_mnv2.h5"
])

DZ_IDX_JSON = find_first_existing([
    Path("disease_class_indices.json"),
    Path("models") / "disease_class_indices.json"
])

IMG_SIZE = (224, 224)

# -------------------- THEME / STYLES -----------------
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
    unsafe_allow_html=True
)

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
        "chat_hint": "Ask about monkeypox "
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
        "chat_title": "💬 Chatbot MediSkin (FAQ)",
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

# Nicely display names (optional)
DISPLAY_NAME = {"other_disease": "Others"}

# -------------------- MODEL LOADING ------------------
@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    # Validate files exist
    if DZ_MODEL_PATH is None or DZ_IDX_JSON is None:
        missing = []
        if DZ_MODEL_PATH is None:
            missing.append("disease_mnv2.h5 (root or models/)")
        if DZ_IDX_JSON is None:
            missing.append("disease_class_indices.json (root or models/)")
        st.error("Missing model files: " + ", ".join(missing))
        st.stop()

    # Read class indices first
    with open(DZ_IDX_JSON, "r", encoding="utf-8") as f:
        idx = json.load(f)  # e.g. {"monkeypox":0, "normal":1, "other_disease":2}
    labels = [c for c, _ in sorted(idx.items(), key=lambda x: x[1])]
    num_classes = len(labels)

    # 1) Try full-model load (works if file contains architecture+weights
    #    saved with the older Keras/TF format)
    try:
        mdl = load_model(str(DZ_MODEL_PATH))
        # quick forward build to ensure compatibility
        _ = mdl.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), verbose=0)
        return mdl, labels
    except Exception as e:
        st.info(f"Full-model load failed, will try rebuild+weights. Details: {e}")

    # 2) Rebuild the exact inference graph and load weights only.
    #    Architecture: MobileNetV2 (no top) -> GAP -> Dense softmax
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)
    mdl = tf.keras.Model(inputs, outputs, name="mnv2_classifier")

    try:
        mdl.load_weights(str(DZ_MODEL_PATH))
    except Exception as e:
        st.error(f"Failed to load weights from {DZ_MODEL_PATH}: {e}")
        st.stop()

    return mdl, labels

def prep(img_pil: Image.Image):
    rgb = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(rgb).astype("float32") / 255.0
    return np.expand_dims(x, 0)

# -------------------- SIMPLE FAQ CHATBOT -------------
# Pure local TF-IDF (no API). Shows in sidebar.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    FAQ_PAIRS = [
        ("what is monkeypox",
         "Monkeypox is a viral zoonotic disease caused by the monkeypox virus. It can spread from animals to humans and between people."),
        ("what are symptoms of monkeypox",
         "Common symptoms include fever, headache, swollen lymph nodes, muscle aches, followed by a skin rash or lesions."),
        ("how does monkeypox spread",
         "It spreads via close contact with lesions, body fluids, respiratory droplets, or contaminated objects."),
        ("is monkeypox dangerous",
         "Usually self-limiting (2–4 weeks), but severe cases can occur in children or immunocompromised people."),
        ("is there a treatment for monkeypox",
         "No specific treatment; supportive care. Smallpox vaccines/antivirals may help in some cases."),
        ("how to prevent monkeypox",
         "Avoid close contact with infected individuals, practice hand hygiene, and use protection when caring for patients.")
    ]
    _VEC = TfidfVectorizer().fit([q for q, _ in FAQ_PAIRS])
    _CORP = _VEC.transform([q for q, _ in FAQ_PAIRS])

    def bot_reply(user_text: str) -> str:
        if not user_text.strip():
            return "Ask me about monkeypox."
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

# Chatbot UI
if CHATBOT_AVAILABLE:
    st.sidebar.markdown(f"### {TEXT[L]['chat_title']}")
    if "chat" not in st.session_state:
        greet = "Hi! Ask about monkeypox." if L == "en" else "Hai! Tanya tentang monkeypox."
        st.session_state.chat = [{"role": "assistant", "content": greet}]
    for m in st.session_state.chat:
        st.sidebar.chat_message(m["role"]).write(m["content"])
    user_msg = st.sidebar.chat_input(TEXT[L]["chat_hint"])
    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        st.session_state.chat.append({"role": "assistant", "content": bot_reply(user_msg)})
        st.rerun()
else:
    st.sidebar.info("FAQ chatbot needs `scikit-learn`.")

# Load model + labels
model, DZ_LABELS = load_model_and_labels()
DZ_DISPLAY = [DISPLAY_NAME.get(lbl, lbl) for lbl in DZ_LABELS]

# -------------------- MAIN UI ------------------------
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title(TEXT[L]["title"])
st.caption(TEXT[L]["subtitle"])

uploaded = st.file_uploader(TEXT[L]["uploader"], type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button(TEXT[L]["btn"], type="primary"):
        with st.spinner(TEXT[L]["running"]):
            x = prep(img)
            probs = model.predict(x, verbose=0)[0]
            j = int(np.argmax(probs))
            label_raw = DZ_LABELS[j]
            label = DISPLAY_NAME.get(label_raw, label_raw)
            conf = float(probs[j])

        st.subheader(TEXT[L]["result"])
        st.markdown(f"**{label}** — confidence **{conf*100:.1f}%**")
        st.markdown(
            f"<div class='meter'><span style='width:{conf*100:.1f}%'></span></div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader(TEXT[L]["advice"])
        st.info(ADVICE[L].get(label_raw.lower(), "—"))
        st.caption(TEXT[L]["disclaimer"])
else:
    st.info(TEXT[L]["noimg"])

st.markdown("</div>", unsafe_allow_html=True)
