"""
demo.py — Final ABSA Demo (Best UI + Franco + Confidence)
Run: streamlit run demo.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# ── Your modules (UNCHANGED) ─────────────────────────────
from train import ABSAModel, DEVICE, build_input_text, CFG
from utils import (
    decode_predictions_with_confidence,
    ASPECTS,
    SENTIMENTS,
    JOINT_LABELS,
    NUM_LABELS,
    is_franco_arabic,
    transliterate_franco,
)
from transformers import AutoTokenizer

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Arabic ABSA — DeepX",
    page_icon="🌟",
    layout="wide"
)

# ── Load Model (KEEP YOUR VERSION) ───────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
    model = ABSAModel(CFG["model_name"], NUM_LABELS, CFG["dropout"]).to(DEVICE)

    model.load_state_dict(
        torch.load(
            Path(CFG["output_dir"]) / "final_model_seed42.pt",
            map_location=DEVICE,
        )
    )

    with open(Path(CFG["output_dir"]) / "threshold_seed42.json") as f:
        threshold = json.load(f)["threshold"]

    return model.eval(), tokenizer, threshold


# ── Load model ───────────────────────────────────────────
with st.spinner("Loading Arabic NLP model..."):
    model, tokenizer, threshold = load_model()

st.success(f"✅ Model loaded | Threshold: {threshold:.3f}")

# ── UI Styling ───────────────────────────────────────────
SENT_STYLE = {
    "positive": ("🟢", "#28a745"),
    "negative": ("🔴", "#dc3545"),
    "neutral": ("🟡", "#ffc107"),
}

ASPECT_AR = {
    "food": "🍽️ الطعام",
    "service": "🤝 الخدمة",
    "price": "💰 السعر",
    "cleanliness": "🧹 النظافة",
    "delivery": "🚚 التوصيل",
    "ambiance": "🏠 الأجواء",
    "app_experience": "📱 التطبيق",
    "general": "⭐ عام",
    "none": "❓ لا يوجد",
}

# ── Title ────────────────────────────────────────────────
st.title("🌟 Arabic Aspect-Based Sentiment Analysis")
st.markdown("**DeepX Team — Hackathon 2026**")
st.divider()

# ── Layout ───────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

# ───────────────────────────────
# LEFT SIDE (INPUT)
# ───────────────────────────────
with col1:
    st.subheader("📝 Enter a Review")

    # Examples
    examples = {
        "مطعم إيجابي": ("الأكل كان لذيذ جداً والخدمة ممتازة بس السعر غالي شوية", 4, "restaurant"),
        "تطبيق سلبي": ("التطبيق بطيء جداً وفيه مشاكل كثيرة في الدفع", 2, "ecommerce"),
        "توصيل متأخر": ("الطلب وصل متأخر جداً والأكل كان بارد", 1, "restaurant"),
        "كافيه ممتاز": ("المكان جميل والقهوة رائعة والخدمة ممتازة", 5, "cafe"),
        "Franco Example": ("el akl kan helw bs el service we7sha", 3, "restaurant"),
    }

    selected = st.selectbox("Choose example:", ["(write your own)"] + list(examples.keys()))

    if selected != "(write your own)":
        default_text, default_stars, default_cat = examples[selected]
    else:
        default_text, default_stars, default_cat = "", 3, ""

    review = st.text_area(
        "Review:",
        value=default_text,
        height=130,
        placeholder="اكتب تقييمك هنا..."
    )

    star_rating = st.slider("⭐ Star Rating", 1, 5, default_stars)
    business_cat = st.text_input("Business Category", value=default_cat)

    predict_btn = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)

# ───────────────────────────────
# RIGHT SIDE (OUTPUT)
# ───────────────────────────────
with col2:
    st.subheader("📊 Results")

    if predict_btn and review.strip():

        # ── Franco Handling ──
        text = review
        if is_franco_arabic(text):
            text = transliterate_franco(text)
            st.info("🇪🇬 Franco detected → converted to Arabic")

        # ── Build input ──
        input_text = build_input_text(
            pd.Series({
                "review_text": text,
                "star_rating": star_rating,
                "business_category": business_cat
            })
        )

        enc = tokenizer(
            input_text,
            max_length=CFG["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Inference ──
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        pred = decode_predictions_with_confidence(probs, threshold)

        aspects = pred["aspects"]
        sentiments = pred["aspect_sentiments"]
        confidences = pred["confidence"]

        avg_conf = np.mean(list(confidences.values())) if confidences else 0

        # ── Aspect Cards ──
        for asp in aspects:
            sent = sentiments[asp]
            conf = confidences.get(asp, 0)

            emoji, color = SENT_STYLE.get(sent, ("⚪", "#6c757d"))
            asp_ar = ASPECT_AR.get(asp, asp)

            st.markdown(f"""
            <div style="
                background:{color}22;
                border-left:4px solid {color};
                padding:10px 14px;
                margin:6px 0;
                border-radius:6px;">
                <b>{asp_ar}</b>
                &nbsp;&nbsp;{emoji}
                <span style="color:{color};font-weight:bold">
                    {sent.upper()}
                </span>
                &nbsp;&nbsp;(conf: {conf:.2f})
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(conf, 1.0))

        # ── Average confidence ──
        st.metric("Average Confidence", f"{avg_conf:.2f}")

        if avg_conf < 0.4:
            st.warning("⚠️ Low confidence prediction")

        # ── Top Probabilities Chart ──
        st.markdown("**Top Label Probabilities:**")
        top_idx = np.argsort(probs)[::-1][:10]
        top_data = {JOINT_LABELS[i]: float(probs[i]) for i in top_idx}
        st.bar_chart(top_data)

        # ── JSON ──
        with st.expander("🔍 Raw JSON"):
            st.json(pred)

    elif predict_btn:
        st.warning("Please enter a review first.")

# ── Footer ──────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:gray;font-size:0.9em'>
DeepX Team | Arabic ABSA | MARBERT | Hackathon 2026<br>
food · service · price · cleanliness · delivery · ambiance · app_experience · general
</div>
""", unsafe_allow_html=True)
