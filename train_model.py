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

# -----------------------------
# Normalize labels
# -----------------------------
df['label'] = df['label'].replace({'FAKE': 0, 'REAL': 1})

# Drop missing text rows
df = df.dropna(subset=['text']).reset_index(drop=True)

print("Dataset shape after cleaning:", df.shape)
print("Label counts:\n", df['label'].value_counts())

# -----------------------------
# Extreme fake news base samples
# -----------------------------
extreme_fake = [
    "Scientists have invented a teleportation device that works instantly.",
    "Miracle cure for cancer discovered in remote jungle.",
    "Aliens secretly controlling world governments, claims new report.",
    "Time machine prototype successfully tested in private lab.",
    "Cure for aging found, will be released next year.",
    "Vaccine proven to give superhuman strength, experts claim.",
    "Invisibility cloak now available for public, scientists announce.",
    "Moon landing was a hoax, new evidence revealed.",
    "Secret portal to parallel universe discovered in Antarctica.",
    "Mind reading device invented to spy on citizens globally.",
    "Atlantis found under the ocean by deep-sea expedition.",
    "Magic pill guarantees weight loss overnight.",
    "Telepathy machine tested on students with amazing results.",
    "Secret government lab creates immortality serum.",
    "Dragons exist, captured on hidden camera in Himalayas.",
    "Flying cars finally available for public purchase next month.",
    "Scientists reverse time in controlled experiments.",
    "Extraterrestrial life found living among humans.",
    "Mind control device sold illegally on dark web.",
    "Energy drink grants infinite energy, tests show.",
    "Psychic predicts all stock market movements correctly.",
    "Supercomputer predicts apocalypse within 24 hours.",
    "New tech allows humans to breathe underwater.",
    "Robot achieves consciousness, plans revolution.",
    "Ancient civilization found on Mars by private telescope.",
    "Teleportation trials begin on humans with 100% success.",
    "Secret city discovered under Sahara desert.",
    "Scientists develop serum to live for 500 years.",
    "Magnetically powered cars replace gasoline vehicles overnight.",
    "Weather manipulation machine causes snow in July."
]

# -----------------------------
# Fake news augmentation
# -----------------------------
def augment_fake(samples, n_aug=3):
    augmented = []
    for text in samples:
        for _ in range(n_aug):
            aug_text = text
            # randomly apply small variations
            if random.random() > 0.5:
                aug_text = aug_text.replace("scientists", "researchers")
            if random.random() > 0.5:
                aug_text = aug_text.replace("device", "technology")
            if random.random() > 0.5:
                aug_text = aug_text.replace("discovered", "found")
            augmented.append(aug_text)
    return augmented

augmented_fake = augment_fake(extreme_fake, n_aug=5)
all_fake_texts = extreme_fake + augmented_fake
extreme_labels = [0] * len(all_fake_texts)

df_extreme = pd.DataFrame({'text': all_fake_texts, 'label': extreme_labels})
df = pd.concat([df, df_extreme], ignore_index=True)

print("Dataset shape after adding extreme + augmented fake news:", df.shape)
print("Label counts:\n", df['label'].value_counts())

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -----------------------------
# Hybrid TF-IDF Vectorizer (smaller size for speed)
# -----------------------------
vectorizer = FeatureUnion([
    ('char', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=3,
        max_features=10000
    )),
    ('word', TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        stop_words='english',
        max_features=20000
    ))
])

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Handle imbalance (fast oversampling)
# -----------------------------
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_tfidf, y_train)
print("Resampled label counts:\n", pd.Series(y_train_bal).value_counts())

# -----------------------------
# Logistic Regression (fast + balanced)
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
# Find best probability threshold
# -----------------------------
probs = model.predict_proba(X_test_tfidf)[:, 0]  # FAKE prob
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
# Extreme fake news test
# -----------------------------
print("\n--- Extreme fake news test ---")
for sample in extreme_fake[:10]:
    proba = model.predict_proba(vectorizer.transform([sample]))[0]
    pred_label = 0 if proba[0] > best_threshold else 1
    print("\nText:", sample)
    print("Predicted label:", pred_label, "(0=FAKE, 1=REAL)")
    print("Probabilities:", proba)
