import streamlit as st
from transformers import pipeline

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
    **Layer 1: Classification**
    *DistilBERT Model*
    
    **Layer 2: Summarization**
    *BART Transformer*
    
    **Layer 3: Analysis**
    *Probability Scoring*
    """)

@st.cache_resource
def load_pipeline():
    # Initialize the classification pipeline
    classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
    
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    return classifier, summarizer

with st.spinner("Initializing System Modules..."):
    classifier, summarizer = load_pipeline()

input_text = st.text_area("Input Source Text:", height=200, placeholder="Paste article content here...")

if st.button("Analyze Text", type="primary"):
    if len(input_text) < 50:
        st.warning("Input text is too short for accurate analysis.")
    else:
        try:
            # Classification
            result = classifier(input_text[:512])[0]
            label = result['label']
            score = result['score']
            
            # --- FINAL LOGIC FIX ---
            # LABEL_0 is usually the "Fake" class in this specific dataset
            is_fake = label == "LABEL_0" 
            confidence_pct = round(score * 100, 2)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if is_fake:
                    st.error(f"STATUS: UNVERIFIED")
                    st.metric("Risk Score", f"{confidence_pct}%", delta="High Risk", delta_color="inverse")
                else:
                    st.success(f"STATUS: VERIFIED")
                    st.metric("Credibility Score", f"{confidence_pct}%", delta="High Confidence")

            with col2:
                st.markdown("#### Analysis Report")
                if is_fake:
                    st.write(f"The system detected linguistic patterns consistent with misinformation (Confidence: {confidence_pct}%).")
                else:
                    st.write(f"The text structure is consistent with verified news sources (Confidence: {confidence_pct}%).")

            st.divider()
            st.subheader("Executive Summary")
            with st.spinner("Generating abstract..."):
                summary_result = summarizer(input_text, max_length=60, min_length=30, do_sample=False)
                summary_text = summary_result[0]['summary_text']
                st.info(summary_text)
                
        except Exception as e:
            st.error(f"Processing Error: {e}")
