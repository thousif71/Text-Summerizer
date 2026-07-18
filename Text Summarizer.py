import streamlit as st
import nltk
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# --- Setup & Caching ---
# Download required NLTK data quietly, including the new punkt_tab requirement
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Cache the heavy ML model so it only loads once
@st.cache_resource
def load_ml_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# --- Summarization Functions ---
def run_abstractive_summary(text):
    summarizer = load_ml_summarizer()
    # Calculate dynamic lengths based on input size
    input_length = len(text.split())
    max_len = min(130, max(30, int(input_length * 0.3)))
    min_len = min(30, int(input_length * 0.1))
    
    result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return result[0]['summary_text']

def run_extractive_summary(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary_sentences])

# --- UI & Decision Engine ---
st.set_page_config(page_title="Summarizer Advisor", layout="wide")
st.title("Text Summarizer Decision Advisor")
st.markdown("Adjust your constraints below to determine the best algorithm, then test it on your own text.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Your Constraints")
    
    hardware = st.radio(
        "Execution Environment",
        ["Cheap CPU Server / Raspberry Pi", "Standard Laptop", "Cloud GPU / High-End PC"]
    )
    
    goal = st.radio(
        "Primary Priority",
        ["Execution Speed & Zero Cost", "Absolute Factual Accuracy", "Human-like Readability & Flow"]
    )

    # --- Scoring Logic ---
    extractive_score = 0
    abstractive_score = 0

    # Hardware evaluation
    if hardware == "Cheap CPU Server / Raspberry Pi":
        extractive_score += 50
        abstractive_score -= 20
    elif hardware == "Standard Laptop":
        extractive_score += 20
        abstractive_score += 10
    else:
        extractive_score += 10
        abstractive_score += 40

    # Priority evaluation
    if goal == "Execution Speed & Zero Cost":
        extractive_score += 40
        abstractive_score -= 10
    elif goal == "Absolute Factual Accuracy":
        extractive_score += 30
        abstractive_score -= 10
    else: # Human-like Readability
        extractive_score -= 10
        abstractive_score += 50

    # Determine Winner
    winner = "Extractive (Sumy/LSA)" if extractive_score > abstractive_score else "Abstractive (Transformers/BART)"

    st.divider()
    st.subheader("Recommendation Score")
    st.progress(max(0, min(100, extractive_score)), text=f"Extractive Fit: {extractive_score}%")
    st.progress(max(0, min(100, abstractive_score)), text=f"Abstractive Fit: {abstractive_score}%")
    
    st.success(f"**Recommended Path:** {winner}")

with col2:
    st.header("2. Test the Winner")
    
    user_text = st.text_area("Paste your long text here:", height=250, placeholder="Enter at least 3-4 sentences of text to summarize...")
    
    if st.button("Generate Summary"):
        if len(user_text.strip()) < 50:
            st.warning("Please enter a bit more text to get a meaningful summary.")
        else:
            with st.spinner(f"Running {winner} summary..."):
                if "Extractive" in winner:
                    final_summary = run_extractive_summary(user_text, sentence_count=3)
                else:
                    final_summary = run_abstractive_summary(user_text)
            
            st.subheader("Result")
            st.info(final_summary)