"""
demo.py – Streamlit demo with confidence bars and Franco‑Arabic support.
Uses the single seed‑42 MARBERT model.  Run: streamlit run demo.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

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


# ── Load the single model (cached) ─────────────────────────────────
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


model, tokenizer, threshold = load_model()

# ── UI constants ───────────────────────────────────────────────────
SENT_STYLE = {
    "positive": ("🟢", "#28a745"),
    "negative": ("🔴", "#dc3545"),
    "neutral": ("🟡", "#ffc107"),
}

ASPECT_ICONS = {
    "food": "🍽️", "service": "🤝", "price": "💰",
    "cleanliness": "🧹", "delivery": "🚚", "ambiance": "🏠",
    "app_experience": "📱", "general": "⭐", "none": "❓",
}

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(page_title="Arabic ABSA", layout="wide")
st.title("🌟 Arabic Aspect‑Based Sentiment Analysis")
st.markdown("Single MARBERT model · Confidence scores · Franco‑Arabic ready")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Enter a Review")
    review = st.text_area(
        "Review text (Arabic, English, or Franco):",
        height=150,
        placeholder="اكتب تقييمك هنا...",
    )

    if st.button("🔮 Analyze Sentiment", type="primary"):
        if review.strip():
            text = review

            # ── Franco‑Arabic detection & transliteration ─────────
            if is_franco_arabic(text):
                text = transliterate_franco(text)
                st.info("🇪🇬 Franco‑Arabic detected → transliterated to Arabic script for analysis.")

            # ── Build model input ─────────────────────────────────
            input_text = build_input_text(
                pd.Series({"review_text": text, "star_rating": 3, "business_category": ""})
            )
            enc = tokenizer(
                input_text,
                max_length=CFG["max_len"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # ── Infer ─────────────────────────────────────────────
            with torch.no_grad():
                logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            pred = decode_predictions_with_confidence(probs, threshold)
            avg_conf = np.mean(list(pred["confidence"].values())) if pred["confidence"] else 0.0

            with col2:
                st.subheader("📊 Results")
                for asp in pred["aspects"]:
                    sent = pred["aspect_sentiments"][asp]
                    conf = pred["confidence"].get(asp, 0)
                    emoji, color = SENT_STYLE[sent]
                    icon = ASPECT_ICONS.get(asp, "")
                    st.markdown(
                        f"""
                        <div style="background:{color}22; border-left:4px solid {color};
                                    padding:10px 14px; margin:6px 0; border-radius:6px;">
                            <span style="font-size:1.1em;font-weight:bold">{icon} {asp}</span>
                            &nbsp;&nbsp;{emoji} <span style="color:{color};font-weight:bold">{sent.upper()}</span>
                            &nbsp;&nbsp;(confidence: {conf:.2f})
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(min(conf, 1.0))

                st.metric("Average Confidence", f"{avg_conf:.2f}")
                if avg_conf < 0.4:
                    st.warning("⚠️ Low confidence – prediction may be unreliable.")

                with st.expander("🔍 Raw JSON"):
                    st.json(pred)
        else:
            st.warning("Please enter a review text.")