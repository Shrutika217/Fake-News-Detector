import pandas as pd
import numpy as np
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from sklearn.pipeline import FeatureUnion
from imblearn.over_sampling import RandomOverSampler

# -----------------------------
# Load datasets
# -----------------------------
df1 = pd.read_csv('news.csv')
df2 = pd.read_csv('news_extra1.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Normalize labels
df['label'] = df['label'].replace({'FAKE': 0, 'REAL': 1})

# Drop missing text rows
df = df.dropna(subset=['text']).reset_index(drop=True)
print("Dataset shape after cleaning:", df.shape)
print("Label counts:\n", df['label'].value_counts())

# -----------------------------
# Extreme keywords fake news samples
# -----------------------------
extreme_fake_keywords = [
    "alien", "parallel universe", "telepathy", "psychic", "astrology", "dragons",
    "miracle cure", "cure for cancer", "aids cure", "herbal remedy",
    "anti-aging pill", "immortality", "cancer cure", "miracle pill",
    "secret government", "mind control", "flat earth", "chemtrails",
    "illuminati", "fake pandemic", "microchip vaccine", "immortal",
    "hidden cure", "miracle drug", "ancient aliens", "time travel",
    "teleportation", "ufo", "cancer reversed"
]

# Create one fake sentence per keyword
extreme_fake_sentences = [f"This news talks about {kw}." for kw in extreme_fake_keywords]
extreme_labels = [0] * len(extreme_fake_sentences)

df_extreme = pd.DataFrame({'text': extreme_fake_sentences, 'label': extreme_labels})

# Add to main dataframe
df = pd.concat([df, df_extreme], ignore_index=True)
print("Dataset shape after adding extreme fake news:", df.shape)
print("Label counts:\n", df['label'].value_counts())

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -----------------------------
# Hybrid TF-IDF Vectorizer
# -----------------------------
vectorizer = FeatureUnion([
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=10000)),
    ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', max_features=20000))
])

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Handle imbalance with RandomOverSampler
# -----------------------------
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_tfidf, y_train)
print("Resampled label counts:\n", pd.Series(y_train_bal).value_counts())

# -----------------------------
# Logistic Regression
# -----------------------------
model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga')
model.fit(X_train_bal, y_train_bal)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# Best probability threshold
# -----------------------------
probs = model.predict_proba(X_test_tfidf)[:, 0]  # FAKE probability
precision, recall, thresholds = precision_recall_curve(y_test, probs, pos_label=0)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print("🔎 Best threshold for FAKE detection:", best_threshold)

# -----------------------------
# Save model, vectorizer, threshold
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)
print("✅ Model, Vectorizer, and Threshold saved successfully!")

# -----------------------------
# Test extreme keyword examples
# -----------------------------
print("\n--- Extreme keyword test ---")
for sentence in extreme_fake_sentences[:10]:
    proba = model.predict_proba(vectorizer.transform([sentence]))[0]
    pred_label = 0 if proba[0] > best_threshold else 1
    print("\nText:", sentence)
    print("Predicted label:", pred_label, "(0=FAKE, 1=REAL)")
    print("Probabilities:", proba)
