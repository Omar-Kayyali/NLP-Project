# Streamlit UI for Arabic Text Summarization
import os
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "1"



import streamlit as st
import tensorflow as tf
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
@st.cache_resource
# this loads the model and tokenizer once and caches them for future use, improving performance

def load_resources():
    model = load_model("model.keras", compile=False)

    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

    return model, tokenizer


model, tokenizer = load_resources()

# CLEANING FUNCTION
# ------------------------------------------------------------
def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove extra spaces
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

    # Prediction
    prediction = model.predict(input_seq, verbose=0)

    # Greedy decoding
    predicted_ids = np.argmax(prediction, axis=-1)[0]

    summary = tokenizer.decode(predicted_ids, skip_special_tokens=True)

    return summary


# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Settings")

example_text = st.sidebar.button("Load Example")



# MAIN INPUT
# ------------------------------------------------------------
def_text = ""

if example_text:
    def_text = "الذكاء الاصطناعي أصبح من أهم التقنيات الحديثة التي تؤثر على مختلف المجالات مثل التعليم والصحة والصناعة، حيث يساعد في تحليل البيانات واتخاذ القرارات بسرعة ودقة عالية."

article = st.text_area(
    "Enter Arabic Article",
    value=def_text,
    height=300,
    placeholder="Paste Arabic text here..."
)


# GENERATE BUTTON
# ------------------------------------------------------------
if st.button("Generate Summary"):

    if not article.strip():
        st.warning("Please enter some Arabic text.")

    else:
        with st.spinner("Generating summary..."):
            summary = summarize(article)

        st.success("Summary Generated Successfully")

        st.subheader("Generated Summary")
        st.write(summary)

        st.subheader("Original Text")
        st.write(article)



# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit + TensorFlow")


## Install Requirements
"""
Run:

```bash
pip install streamlit tensorflow transformers sentencepiece
```

## Run the App

```bash
streamlit run app.py
```

## Expected Project Structure

```text
project/
│
├── app.py
├── summarizer_model.keras
├── data.csv
└── other training files...
```

## Optional Improvements
"""