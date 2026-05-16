# Streamlit UI for Arabic Text Summarization
import os
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "1"

from pathlib import Path
import streamlit as st
import numpy as np
import re
from keras.saving import load_model
from transformers import AutoTokenizer

# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Arabic Text Summarizer",
    layout="wide"
)

st.title("Arabic Text Summarization")
st.markdown("Generate summaries for Arabic articles using your trained Seq2Seq model.")

# LOAD MODEL & TOKENIZER
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "summarizer_model.keras"

st.write("BASE_DIR:", BASE_DIR)
st.write("Model exists:", MODEL_PATH.exists())

if MODEL_PATH.exists():
    st.write("Model size:", MODEL_PATH.stat().st_size)

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    return model, tokenizer

model, tokenizer = load_resources()

# CLEANING FUNCTION
# ------------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# TOKENIZATION
# ------------------------------------------------------------
MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 80

def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        return_tensors="tf"
    )
    return tokens["input_ids"]

# SUMMARY GENERATION
# ------------------------------------------------------------
def summarize(text):
    input_seq = preprocess_text(text)
    
    # NOTE: Ensure your Keras model supports single-input inference.
    # If it's a standard Seq2Seq, you will need an autoregressive loop here.
    prediction = model.predict(input_seq, verbose=0)
    
    predicted_ids = np.argmax(prediction, axis=-1)[0]
    summary = tokenizer.decode(predicted_ids, skip_special_tokens=True)
    return summary

# SIDEBAR & SESSION STATE
# ------------------------------------------------------------
st.sidebar.header("Settings")

# Initialize session state for the text area
if "article_text" not in st.session_state:
    st.session_state.article_text = ""

if st.sidebar.button("Load Example"):
    st.session_state.article_text = "الذكاء الاصطناعي أصبح من أهم التقنيات الحديثة التي تؤثر على مختلف المجالات مثل التعليم والصحة والصناعة، حيث يساعد في تحليل البيانات واتخاذ القرارات بسرعة ودقة عالية."

# MAIN INPUT
# ------------------------------------------------------------
article = st.text_area(
    "Enter Arabic Article",
    value=st.session_state.article_text,
    height=300,
    placeholder="Paste Arabic text here...",
    key="article_input" 
)

# GENERATE BUTTON
# ------------------------------------------------------------
if st.button("Generate Summary"):
    # Read from the current text area value
    current_text = st.session_state.article_input 
    
    if not current_text.strip():
        st.warning("Please enter some Arabic text.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize(current_text)
            
        st.success("Summary Generated Successfully")
        st.subheader("Generated Summary")
        st.write(summary)
        
        st.subheader("Original Text")
        st.write(current_text)

# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit + TensorFlow")