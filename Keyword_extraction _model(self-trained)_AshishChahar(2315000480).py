"""
All-in-One Keyword Classifier
-----------------------------
Author: Ashish Kumar Chahar
Description:
A complete pipeline for keyword-based topic classification using Logistic Regression 
and Random Forest. Includes data balancing, TF-IDF vectorization, model training,
evaluation, and sample predictions.

Usage:
    1. Place your CSV file in the same directory.
    2. Run this script: python keyword_classifier.py
"""

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------
# Configuration
# ---------------------------
CSV_FILE = "keywords_for_labeling.csv"   # <-- Replace with your file name
TOPICS_FILTER = [
    "Artificial intelligence", "Climate Change", "Quantum computing",
    "History of the Internet", "Machine learning", "Cybersecurity",
    "Data Science", "Human Brain"
]
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------
# Step 1: Load Data
# ---------------------------
print("📂 Loading CSV file...")
df = pd.read_csv(CSV_FILE)
print("\nOriginal label distribution:\n", df['label'].value_counts())

# Optional topic filter
df = df[df["topic"].isin(TOPICS_FILTER)]

# ---------------------------
# Step 2: Balance Dataset
# ---------------------------
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=RANDOM_STATE
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
print("\nBalanced label distribution:\n", df_balanced['label'].value_counts())

# ---------------------------
# Step 3: Vectorization
# ---------------------------
X = df_balanced["keyword"]
y = df_balanced["label"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X_vec = vectorizer.fit_transform(X)

# ---------------------------
# Step 4: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ---------------------------
# Step 5: Logistic Regression
# ---------------------------
log_model = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n🔹 Logistic Regression Results:")
print(classification_report(y_test, y_pred_log))

# ---------------------------
# Step 6: Random Forest
# ---------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n🔹 Random Forest Results:")
print(classification_report(y_test, y_pred_rf))

# ---------------------------
# Step 7: Test on Custom Keywords
# ---------------------------
test_phrases = [
    "the result",
    "an unprecedented rise",
    "the most important effect",
    "capital",
    "mechanized textiles spinning",
    "advanced machinery",
    "the 1830s",
    "continental europe",
    "value",
    "the western world",
    "property rights",
    "the emergence",
    "the adoption"
]

X_test_phrases = vectorizer.transform(test_phrases)
log_preds = log_model.predict(X_test_phrases)
rf_preds = rf_model.predict(X_test_phrases)

print("\n🔹 Logistic Regression Predictions:")
for phrase, pred in zip(test_phrases, log_preds):
    print(f"{phrase} → {int(pred)}")

print("\n🔹 Random Forest Predictions:")
for phrase, pred in zip(test_phrases, rf_preds):
    print(f"{phrase} → {int(pred)}")
