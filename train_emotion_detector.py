import os
os.chdir(r"C:\Users\tellm\Desktop\Mental Health Screening & Well-Being Assessment .csv")

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD MASTER DATASET
# ─────────────────────────────────────────────
print("⏳ Loading master dataset...")
df = pd.read_csv("data/master_dataset.csv")
print(f"✅ Loaded: {len(df)} rows")
print(f"   Labels: {df['label'].nunique()} unique")
print()

# ─────────────────────────────────────────────
# STEP 2: CLEAN THE TEXT
# ─────────────────────────────────────────────
print("⏳ Cleaning text...")

def clean_text(text):
    text = str(text).lower()               # lowercase
    text = re.sub(r'http\S+', '', text)    # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # keep only letters
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text

df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 10]  # remove very short texts
df = df.dropna()

print(f"✅ After cleaning: {len(df)} rows")
print()

# ─────────────────────────────────────────────
# STEP 3: KEEP ONLY TOP LABELS
# (removes rare labels with very few examples)
# ─────────────────────────────────────────────
print("⏳ Filtering labels...")
label_counts = df['label'].value_counts()
top_labels = label_counts[label_counts >= 100].index
df = df[df['label'].isin(top_labels)]

print(f"✅ Labels kept (100+ examples): {len(top_labels)}")
print(df['label'].value_counts().to_string())
print()

# ─────────────────────────────────────────────
# STEP 4: SPLIT INTO TRAIN AND TEST
# ─────────────────────────────────────────────
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {len(X_train)} rows")
print(f"   Test set:  {len(X_test)} rows")
print()

# ─────────────────────────────────────────────
# STEP 5: TF-IDF — CONVERT TEXT TO NUMBERS
# ─────────────────────────────────────────────
print("⏳ Converting text to numbers with TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=10000,   # top 10,000 most useful words
    ngram_range=(1, 2),   # single words AND pairs like "feeling hopeless"
    min_df=2              # ignore words that appear only once
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"✅ TF-IDF matrix: {X_train_tfidf.shape}")
print()

# ─────────────────────────────────────────────
# STEP 6: TRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────────
print("⏳ Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # handles imbalanced labels fairly
    random_state=42
)
model.fit(X_train_tfidf, y_train)
print("✅ Model trained!")
print()

# ─────────────────────────────────────────────
# STEP 7: EVALUATE ACCURACY
# ─────────────────────────────────────────────
print("📊 Evaluating on test set...")
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred) * 100
print(f"✅ Overall Accuracy: {acc:.1f}%")
print()
print("📋 Per-label breakdown:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# STEP 8: SAVE THE MODEL
# ─────────────────────────────────────────────
print("⏳ Saving model...")
with open("data/emotion_detector_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model saved!")
print("   data/emotion_detector_model.pkl")
print("   data/tfidf_vectorizer.pkl")
print()

# ─────────────────────────────────────────────
# STEP 9: TEST WITH REAL SENTENCES
# ─────────────────────────────────────────────
print("🔮 Testing with real sentences:")
print("─" * 45)

test_sentences = [
    "I feel so hopeless and empty, nothing makes me happy anymore",
    "I cant stop worrying about everything, my heart is always racing",
    "I keep having nightmares about what happened to me",
    "I lost my home in the flood, I have nothing left",
    "I just need someone to talk to, I feel so alone",
    "mai bahut udaas hoon, kuch achha nahi lagta",  # Urdu mixed
]

for sentence in test_sentences:
    cleaned = clean_text(sentence)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    conf    = round(max(proba) * 100, 1)
    print(f"  Input : {sentence[:55]}...")
    print(f"  Detected : {pred} ({conf}% confident)")
    print()
