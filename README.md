# 📰 Fake News Detector with LIME (Streamlit App)

## 📌 Overview
This project is a **Fake News Detector** built with **Streamlit** that classifies a given news snippet as **FAKE** or **REAL**.  
It uses:

- **Logistic Regression Classifier** trained on TF-IDF features (word + character n-grams)
- A **custom probability threshold** to improve FAKE news detection
- **LIME (Local Interpretable Model-Agnostic Explanations)** to highlight which words influenced the prediction

The app is deployed on **Streamlit Cloud** and fetches trained models (`.pkl` files) directly from **Google Drive**, ensuring smooth deployment without large file issues on GitHub.

---

## ✨ Features
- 📰 **Detects Fake vs Real news** in real-time
- 📊 **Sidebar Dashboard** showing model metrics and dataset information
- ⚖️ **Custom Threshold** for balanced FAKE detection
- 🔎 **LIME explanations** to make predictions interpretable
- ☁️ **Google Drive integration** → avoids large file size limits on GitHub
- 🚀 Deployable instantly on **Streamlit Cloud**

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) → Interactive web UI
- [scikit-learn](https://scikit-learn.org/) → Logistic Regression + TF-IDF vectorizer
- [LIME](https://github.com/marcotcr/lime) → Model interpretability
- [NumPy](https://numpy.org/) → Numerical computations
- [Requests](https://docs.python-requests.org/) → Fetch models from Google Drive

---

## 📂 Project Structure
- `app.py` → Main Streamlit app
- train_model.py → Model Training
- `requirements.txt` → Dependencies for Streamlit Cloud  
- `README.md` → Project documentation  

> ⚠️ Note: `.pkl` files (model, vectorizer, threshold) are **not stored locally**. They are loaded from Google Drive at runtime.

---

## 📊 Model Details

**Algorithm:** Logistic Regression  
**Features:** TF-IDF (word-level 1–2 grams, character-level 3–6 grams, max 75k features)  
**Dataset:**
- Combined `news.csv` + `news_extra1.csv`
- Labels: 0 = FAKE, 1 = REAL
- Additional synthetic extreme-fake samples (aliens, time travel, etc.)
- Balanced using RandomOverSampler

### Performance

| Metric    | FAKE (0) | REAL (1) |
|-----------|----------|----------|
| Precision | 0.91     | 0.87     |
| Recall    | 0.86     | 0.91     |
| F1-score  | 0.88     | 0.89     |
| Support   | 7674     | 8048     |

**Accuracy:** 88.8%  
**Macro Avg F1:** 0.89  
**Custom Threshold:** tuned for FAKE detection

---

## 🔎 Example Usage

- Paste a snippet of news into the text area  
- Click **🔍 Analyze**  
- The app outputs:  
  - Prediction (FAKE/REAL)  
  - Class probabilities  
  - Threshold used  
  - LIME explanation (highlighting words driving the decision)

---

## 🚀 Future Improvements

- Add support for deep learning models (e.g., BERT, RoBERTa)  
- Enable multi-language news detection  
- Allow users to upload CSVs for batch classification  
- Improve UI with charts for LIME visualization

---

## Acknowledgements

- Streamlit for the deployment platform  
- scikit-learn for ML tools  
- [LIME](https://github.com/marcotcr/lime) for explainable AI  
- Datasets 
  -- https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
  -- https://www.kaggle.com/datasets/hassanamin/textdb3

---

## 🗂️ Google Drive Links for Models

| File             | Google Drive Link |
|-----------------|-----------------|
| model.pkl       | [Link](https://drive.google.com/file/d/16E2-kJiQUSvVwFPk4lXnZoYlWc9anSF7/view?usp=sharing) |
| vectorizer.pkl  | [Link](https://drive.google.com/file/d/1Lvrtj2M_cVmcjMVF4_xmiD80nN5Epeep/view?usp=sharing) |
| threshold.pkl   | [Link](https://drive.google.com/file/d/1Otfi_VsZG1_C3jguU0fzyncGFk4pnWDO/view?usp=sharing) |

