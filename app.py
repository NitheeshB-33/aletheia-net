import streamlit as st
from transformers import pipeline
import re
from datetime import datetime

# ------------------ UI ------------------
st.set_page_config(
    page_title="Aletheia-Net",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Aletheia-Net")
st.subheader("Automated Credibility Assessment System")

with st.sidebar:
    st.markdown("### System Architecture")
    st.info("""
    **Layer 1 — Semantic Classifier**
    Zero-Shot Transformer
    
    **Layer 2 — Logic Guardrails**
    Date & anomaly detection
    
    **Layer 3 — Summarization**
    BART Neural Summarizer
    """)
    st.warning("⚠️ Hybrid AI + Rule Engine")

# ------------------ MODELS ------------------
@st.cache_resource
def load_models():
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1"
    )

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    return classifier, summarizer

with st.spinner("Booting neural engines..."):
    classifier, summarizer = load_models()

# ------------------ LOGIC GUARDRAILS ------------------
def check_future_dates(text):
    current_year = datetime.now().year
    years = re.findall(r'\b(20[2-9][0-9])\b', text)
    for year in years:
        if int(year) > current_year:
            return True, year
    return False, None

# ------------------ ZERO-SHOT VOTING ENGINE ------------------
def classify_long_text_zero_shot(text, labels, classifier):
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    score_map = {label: 0 for label in labels}

    for chunk in chunks:
        r = classifier(chunk, labels)
        for label, score in zip(r["labels"], r["scores"]):
            score_map[label] += score

    final_label = max(score_map, key=score_map.get)
    final_score = score_map[final_label] / len(chunks)

    return final_label, final_score

# ------------------ UI INPUT ------------------
input_text = st.text_area(
    "Input Source Text:",
    height=200,
    placeholder="Paste article content here..."
)

# ------------------ ANALYSIS ------------------
if st.button("Analyze Text", type="primary"):

    if len(input_text) < 50:
        st.warning("Text too short for credibility analysis.")
    else:
        try:
            # Rule-based check
            has_future, future_year = check_future_dates(input_text)

            # AI semantic classification
            candidate_labels = [
                "reliable factual news",
                "fake news",
                "clickbait",
                "opinion piece"
            ]

            top_label, confidence = classify_long_text_zero_shot(
                input_text,
                candidate_labels,
                classifier
            )

            confidence_pct = round(confidence * 100, 2)

            st.divider()
            col1, col2 = st.columns(2)

            # ------------------ DECISION ENGINE ------------------
            if has_future:
                status = "UNVERIFIED (Speculative)"
                msg = f"Mentions a future year (**{future_year}**). Predictions are not verifiable facts."

                with col1:
                    st.error(f"STATUS: {status}")
                    st.metric("Risk Score", "100%", delta="Speculative")

                safe = False

            elif top_label in ["fake news", "clickbait"]:
                status = "UNVERIFIED"
                msg = f"Classified as **{top_label}** by semantic AI (Confidence {confidence_pct}%)."

                with col1:
                    st.error(f"STATUS: {status}")
                    st.metric("Risk Score", f"{confidence_pct}%", delta="High Risk", delta_color="inverse")

                safe = False

            else:
                status = "VERIFIED"
                msg = f"Classified as **{top_label}** with {confidence_pct}% confidence."

                with col1:
                    st.success(f"STATUS: {status}")
                    st.metric("Credibility Score", f"{confidence_pct}%", delta="High Confidence")

                safe = True

            with col2:
                st.markdown("### Analysis Report")
                st.write(msg)
                if not safe:
                    st.caption("Recommendation: Cross-check with trusted news sources.")

            # ------------------ SUMMARY ------------------
            st.divider()
            st.subheader("Executive Summary")

            with st.spinner("Generating neural abstract..."):
                summary = summarizer(
                    input_text,
                    max_length=60,
                    min_length=30,
                    do_sample=False
                )[0]["summary_text"]

                st.info(summary)

        except Exception as e:
            st.error(f"System Error: {e}")
