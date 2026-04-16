import os
os.chdir(r"C:\Users\tellm\Desktop\Mental Health Screening & Well-Being Assessment .csv")

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD BOTH FILES
# ─────────────────────────────────────────────

# --- English CSV (18 rows) ---
df_en = pd.read_csv("survey.csv")
df_en = df_en.drop(columns=["Timestamp"])
df_en.columns = [
    "age", "profession",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9",
    "gad1","gad2","gad3","gad4","gad5","gad6","gad7",
    "who1","who2","who3","who4","who5"
]

# Extract number from "2: More than half the days" → 2
score_cols = [c for c in df_en.columns if c not in ["age","profession"]]
for col in score_cols:
    df_en[col] = df_en[col].astype(str).str.split(":").str[0].str.strip().astype(int)

print(f"✅ English CSV loaded:  {len(df_en)} rows")

# --- Urdu XLSX (4 rows) ---
df_ur = pd.read_excel("english.xlsx")
df_ur.columns = [
    "age", "profession",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9",
    "gad1","gad2","gad3","gad4","gad5","gad6","gad7",
    "who1","who2","who3","who4","who5"
]

# Extract number from "several days (1)" → 1
def extract_number(val):
    match = re.search(r'\((\d)\)', str(val))
    return int(match.group(1)) if match else 0

for col in score_cols:
    df_ur[col] = df_ur[col].apply(extract_number)

print(f"✅ Urdu XLSX loaded:    {len(df_ur)} rows")

# ─────────────────────────────────────────────
# STEP 2: COMBINE BOTH DATASETS
# ─────────────────────────────────────────────
df = pd.concat([df_en, df_ur], ignore_index=True)
print(f"✅ Combined dataset:    {len(df)} rows total")
print()

# ─────────────────────────────────────────────
# STEP 3: COMPUTE CLINICAL SCORES
# ─────────────────────────────────────────────
phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]
gad_cols = ["gad1","gad2","gad3","gad4","gad5","gad6","gad7"]
who_cols = ["who1","who2","who3","who4","who5"]

df["phq_score"] = df[phq_cols].sum(axis=1)
df["gad_score"] = df[gad_cols].sum(axis=1)
df["who_score"] = df[who_cols].sum(axis=1)

print("📊 Score Summary (all 22 people):")
print(df[["phq_score","gad_score","who_score"]].describe().round(1))
print()

# ─────────────────────────────────────────────
# STEP 4: CREATE LABELS
# ─────────────────────────────────────────────
def phq_label(s):
    if s <= 4:    return "Minimal"
    elif s <= 9:  return "Mild"
    elif s <= 14: return "Moderate"
    else:         return "Severe"

def gad_label(s):
    if s <= 4:    return "Minimal"
    elif s <= 9:  return "Mild"
    elif s <= 14: return "Moderate"
    else:         return "Severe"

def who_label(s):
    return "Good" if s >= 13 else "Low"

df["depression_label"] = df["phq_score"].apply(phq_label)
df["anxiety_label"]    = df["gad_score"].apply(gad_label)
df["wellbeing_label"]  = df["who_score"].apply(who_label)

print("🏷️  Label Distribution (22 people):")
print("  Depression:", df["depression_label"].value_counts().to_dict())
print("  Anxiety:   ", df["anxiety_label"].value_counts().to_dict())
print("  Wellbeing: ", df["wellbeing_label"].value_counts().to_dict())
print()

# ─────────────────────────────────────────────
# STEP 5: PREPARE FEATURES
# ─────────────────────────────────────────────
le_age  = LabelEncoder()
le_prof = LabelEncoder()
df["age_enc"]  = le_age.fit_transform(df["age"])
df["prof_enc"] = le_prof.fit_transform(df["profession"])

feature_cols = phq_cols + gad_cols + who_cols
X = pd.concat([df[feature_cols], df[["age_enc","prof_enc"]]], axis=1)

print(f"✅ Features ready: {X.shape[1]} columns, {X.shape[0]} rows")
print()

# ─────────────────────────────────────────────
# STEP 6: TRAIN 3 MODELS
# ─────────────────────────────────────────────
models         = {}
label_encoders = {}

for target_name, label_col in [
    ("Depression", "depression_label"),
    ("Anxiety",    "anxiety_label"),
    ("Wellbeing",  "wellbeing_label"),
]:
    le  = LabelEncoder()
    y   = le.fit_transform(df[label_col])
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X, y)
    models[target_name]         = clf
    label_encoders[target_name] = le
    acc = clf.score(X, y) * 100
    print(f"✅ {target_name} model trained | Training accuracy: {acc:.0f}%")

print()
print("⚠️  Training accuracy on 22 rows is optimistic.")
print("   Collect more responses for reliable real-world performance.")
print()

# ─────────────────────────────────────────────
# STEP 7: PREDICT FOR A NEW PERSON
# ─────────────────────────────────────────────
# PHQ/GAD: 0=Not at all  1=Several days  2=More than half  3=Nearly every day
# WHO:     0=At no time  1=Some time  2=Less than half  3=More than half  4=Most time  5=All time

new_person = {
    "phq1": 2, "phq2": 2, "phq3": 1, "phq4": 2,
    "phq5": 1, "phq6": 1, "phq7": 1, "phq8": 0, "phq9": 0,
    "gad1": 2, "gad2": 1, "gad3": 2, "gad4": 1,
    "gad5": 1, "gad6": 1, "gad7": 1,
    "who1": 2, "who2": 2, "who3": 2, "who4": 1, "who5": 2,
    "age":        "18 to 25 years",
    "profession": "Student"
}

try:
    age_enc  = le_age.transform([new_person["age"]])[0]
    prof_enc = le_prof.transform([new_person["profession"]])[0]
except:
    age_enc, prof_enc = 0, 0

new_features = [new_person[c] for c in feature_cols] + [age_enc, prof_enc]
new_X = pd.DataFrame([new_features], columns=X.columns)

print("🔮 Prediction for new person:")
print("─" * 40)
for target_name, clf in models.items():
    le      = label_encoders[target_name]
    pred    = clf.predict(new_X)[0]
    label   = le.inverse_transform([pred])[0]
    proba   = clf.predict_proba(new_X)[0]
    classes = le.inverse_transform(range(len(proba)))
    conf    = {c: round(p*100, 1) for c, p in zip(classes, proba)}
    print(f"  {target_name:12}: {label}")
    print(f"  Confidence  : {conf}")
    print()

# ─────────────────────────────────────────────
# STEP 8: SAVE COMBINED DATASET
# ─────────────────────────────────────────────
df.to_csv("combined_dataset.csv", index=False)
print("💾 Combined dataset saved as: combined_dataset.csv")
print(f"   Total rows: {len(df)} (18 English + 4 Urdu)")
