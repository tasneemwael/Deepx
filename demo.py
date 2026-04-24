"""
demo.py — Interactive ABSA demo for judges
Run: streamlit run demo.py
"""

import streamlit as st
import torch
import json
import numpy as np
from pathlib import Path

# ── Import our solution modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    ABSAModel, clean_arabic, build_input_text,
    decode_predictions, JOINT_LABELS, NUM_LABELS, CFG, DEVICE
)
from transformers import AutoTokenizer

# ── Page config
st.set_page_config(
    page_title="Arabic ABSA — DeepX",
    page_icon="🌟",
    layout="wide",
)

# ── Load model (cached)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
    model     = ABSAModel(CFG["model_name"], NUM_LABELS, CFG["dropout"])
    ckpt_path = Path(CFG["output_dir"]) / "final_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    # Load tuned threshold
    thr_path = Path(CFG["output_dir"]) / "threshold.json"
    threshold = 0.35
    if thr_path.exists():
        with open(thr_path) as f:
            threshold = json.load(f)["threshold"]
    return model, tokenizer, threshold


def predict_single(text, star_rating, business_cat, model, tokenizer, threshold):
    import pandas as pd
    row = pd.Series({
        "review_text": text,
        "star_rating": star_rating,
        "business_category": business_cat,
    })
    input_text = build_input_text(row)
    enc = tokenizer(
        input_text,
        max_length=CFG["max_len"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    result = decode_predictions(probs, threshold)
    return result, probs


# ── Sentiment emoji / color mapping
SENT_STYLE = {
    "positive": ("🟢", "#28a745", "Positive"),
    "negative": ("🔴", "#dc3545", "Negative"),
    "neutral":  ("🟡", "#ffc107", "Neutral"),
}

ASPECT_ARABIC = {
    "food":           "🍽️ الطعام",
    "service":        "🤝 الخدمة",
    "price":          "💰 السعر",
    "cleanliness":    "🧹 النظافة",
    "delivery":       "🚚 التوصيل",
    "ambiance":       "🏠 الأجواء",
    "app_experience": "📱 تطبيق",
    "general":        "⭐ عام",
    "none":           "❓ لا يوجد",
}

# ────────────────────────────────────────────
#  MAIN UI
# ────────────────────────────────────────────

st.title("🌟 Arabic Aspect-Based Sentiment Analysis")
st.markdown("**DeepX Team** — Hackathon 2026 | تحليل المشاعر بالعربية")
st.divider()

# ── Load model
with st.spinner("Loading Arabic NLP model..."):
    try:
        model, tokenizer, threshold = load_model()
        st.success(f"✅ Model loaded | Threshold: {threshold:.3f}")
    except Exception as e:
        st.error(f"❌ Model not found. Run `python train.py` first.\n\n{e}")
        st.stop()

# ── Two columns layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📝 Enter a Review")

    # Preset examples
    examples = {
        "مطعم إيجابي":     ("الأكل كان لذيذ جداً والخدمة ممتازة بس السعر غالي شوية", 4, "مطعم"),
        "تطبيق سلبي":      ("التطبيق بطيء جداً وفيه مشاكل كثيرة في الدفع", 2, "ecommerce"),
        "توصيل متأخر":     ("الطلب وصل متأخر جداً والأكل كان بارد، الخدمة سيئة", 1, "مطعم"),
        "كافيه ممتاز":     ("المكان نظيف وجميل جداً والجو رائع والقهوة تحفة", 5, "كافيه"),
        "تقييم محايد":     ("التجربة عادية مش مميزة مش سيئة", 3, "مطعم"),
    }

    selected = st.selectbox("Or pick an example:", ["(type your own)"] + list(examples.keys()))
    if selected != "(type your own)":
        default_text, default_stars, default_cat = examples[selected]
    else:
        default_text, default_stars, default_cat = "", 3, "مطعم"

    review_text = st.text_area(
        "Review text (Arabic):", value=default_text, height=120,
        placeholder="اكتب تقييمك هنا بالعربي..."
    )
    star_rating  = st.slider("⭐ Star Rating:", 1, 5, default_stars)
    business_cat = st.text_input("Business Category:", value=default_cat)

    predict_btn = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Results")

    if predict_btn and review_text.strip():
        with st.spinner("Analyzing..."):
            result, probs = predict_single(
                review_text, star_rating, business_cat, model, tokenizer, threshold
            )

        aspects      = result["aspects"]
        aspect_sents = result["aspect_sentiments"]

        # ── Show detected aspects
        for asp in aspects:
            sent  = aspect_sents[asp]
            emoji, color, label = SENT_STYLE.get(sent, ("⚪", "#6c757d", sent))
            asp_arabic = ASPECT_ARABIC.get(asp, asp)
            st.markdown(
                f"""<div style="
                    background:{color}22; border-left:4px solid {color};
                    padding:10px 14px; margin:6px 0; border-radius:6px;">
                    <span style="font-size:1.1em;font-weight:bold">{asp_arabic}</span>
                    &nbsp;&nbsp;{emoji} <span style="color:{color};font-weight:bold">{label}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── JSON output
        st.markdown("**JSON Output:**")
        st.json({
            "aspects":           aspects,
            "aspect_sentiments": aspect_sents,
        })

        # ── Confidence chart
        st.markdown("**Label Probabilities (top 10):**")
        top_idx  = np.argsort(probs)[::-1][:10]
        top_data = {JOINT_LABELS[i]: float(probs[i]) for i in top_idx}
        st.bar_chart(top_data)

    elif predict_btn:
        st.warning("Please enter a review text first.")

# ── Footer
st.divider()
st.markdown("""
<div style='text-align:center;color:gray;font-size:0.9em'>
DeepX Team | Arabic ABSA | AraBERT v2 | Hackathon 2026<br>
Aspects: food · service · price · cleanliness · delivery · ambiance · app_experience · general
</div>
""", unsafe_allow_html=True)
