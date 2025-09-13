import streamlit as st
import pickle
import lime
import lime.lime_text
import numpy as np
import requests
from io import BytesIO

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Fake News Detector 📰",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Function to load pickle from Google Drive
# ------------------------------
def load_pickle_from_drive(drive_url):
    """
    Load a pickle file from a Google Drive shareable link.
    """
    file_id = drive_url.split("/d/")[1].split("/")[0]
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(BytesIO(response.content))

# ------------------------------
# Load Model, Vectorizer, Threshold from Google Drive
# ------------------------------
model = load_pickle_from_drive("https://drive.google.com/file/d/16E2-kJiQUSvVwFPk4lXnZoYlWc9anSF7/view?usp=sharing")
vectorizer = load_pickle_from_drive("https://drive.google.com/file/d/1Lvrtj2M_cVmcjMVF4_xmiD80nN5Epeep/view?usp=sharing")
best_threshold = load_pickle_from_drive("https://drive.google.com/file/d/1Otfi_VsZG1_C3jguU0fzyncGFk4pnWDO/view?usp=sharing")

# Wrap model + vectorizer for LIME
pipeline = lambda texts: model.predict_proba(vectorizer.transform(texts))

# ------------------------------
# Fake keyword override list
# ------------------------------
fake_keywords = [
    "alien", "ufo", "parallel universe", "telepathy", "psychic", "astrology", "dragons",
    "miracle cure", "cure for cancer", "aids cure", "herbal remedy", "immortality",
    "anti-aging pill", "immortal", "miracle pill", "secret government", "mind control",
    "flat earth", "chemtrails", "illuminati", "fake pandemic", "microchip vaccine",
    "hidden cure", "miracle drug", "ancient aliens", "time travel", "teleportation"
]

# ------------------------------
# Sidebar Dashboard
# ------------------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.markdown("Quick overview of the app and model performance.")

# Model Info
st.sidebar.subheader("ℹ️ About the Model")
st.sidebar.info(
    "This app uses a **Logistic Regression Classifier** trained on TF-IDF (word + char n-grams). "
    "Balanced training and oversampling ensure fair predictions for both FAKE and REAL news. "
    "A custom probability threshold improves fake news detection."
)

# Accuracy
st.sidebar.subheader("✅ Model Accuracy")
st.sidebar.success("Accuracy: 0.8906 (~89.1%)")

# Class Metrics
metrics_table = """
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| FAKE (0) | 0.91 | 0.86 | 0.88 | 7644 |
| REAL (1) | 0.87 | 0.92 | 0.90 | 8048 |
"""
st.sidebar.subheader("📊 Class-wise Metrics")
st.sidebar.markdown(metrics_table)

# Macro & Weighted averages
averages_table = """
| Metric | Macro Avg | Weighted Avg |
|--------|-----------|--------------|
| Precision | 0.89 | 0.89 |
| Recall    | 0.89 | 0.89 |
| F1-score  | 0.89 | 0.89 |
"""
st.sidebar.subheader("📊 Averages")
st.sidebar.markdown(averages_table)

# Threshold
st.sidebar.subheader("⚖️ Custom Threshold")
st.sidebar.write(f"Best threshold for **FAKE detection** = `{best_threshold:.4f}`")

# Dataset Info
st.sidebar.subheader("📂 Dataset Info")
st.sidebar.write(
    """
    - Combined datasets: **news.csv + news_extra1.csv**  
    - Labels: **0 = FAKE**, **1 = REAL**  
    - Balanced with RandomOverSampler  
    - TF-IDF: word (1–2) + char (3–6), max 75k features  
    """
)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit, scikit-learn & LIME")

# ------------------------------
# Main App
# ------------------------------
st.title("📰 Fake News Detector")
st.write("Enter a news snippet below, and the model will classify it as **FAKE** or **REAL**.")

# User input
user_input = st.text_area("✍️ Paste news text here:", height=200)

if st.button("🔍 Analyze"):
    if user_input.strip():
        # Predict probabilities
        prob = pipeline([user_input])[0]

        # Apply hybrid rule: keyword override > threshold
        text_lower = user_input.lower()
        triggered_terms = [kw for kw in fake_keywords if kw in text_lower]

        if triggered_terms:
            pred_label = 0
            override = True
        else:
            pred_label = 0 if prob[0] >= best_threshold else 1
            override = False

        pred_class = "🟥 FAKE News" if pred_label == 0 else "🟩 REAL News"

        # Display result
        st.markdown(f"### Prediction: {pred_class}")
        st.write(f"**FAKE Probability (0):** {prob[0]:.4f}")
        st.write(f"**REAL Probability (1):** {prob[1]:.4f}")
        st.write(f"**Custom Threshold for FAKE:** {best_threshold:.4f}")

        if override:
            st.warning(
                f"⚠️ Keyword override applied "
                f"(flagged as FAKE due to suspicious term: **{', '.join(triggered_terms)}**)"
            )

        # --------------------------
        # LIME Explanation (fast mode)
        # --------------------------
        explainer = lime.lime_text.LimeTextExplainer(class_names=["FAKE", "REAL"])
        with st.spinner("Generating LIME explanation..."):
            exp = explainer.explain_instance(
                user_input,
                pipeline,
                num_features=10,
                num_samples=300   # reduced for speed
            )

        st.subheader("🔎 Explanation (LIME)")
        st.components.v1.html(exp.as_html(), height=600, scrolling=True)

    else:
        st.warning("⚠️ Please enter some text to analyze.")
