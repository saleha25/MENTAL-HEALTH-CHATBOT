# Empath AI — Mental Health Screening & Well-Being Assessment
**Target Users:** Internally Displaced Persons (IDPs) in Pakistan  
**Stack:** Flask · OpenAI GPT-4.1-mini · scikit-learn · HTML/CSS/JS

---

## Overview
A web-based mental health support chatbot that conducts empathetic conversations, runs clinical screening tools (PHQ-9, GAD-7, WHO-5), detects crisis signals, and escalates to professional helplines when needed. Culturally adapted for Pakistani users with Quranic verses and Urdu-aware text normalization.

---

## Architecture

```
User (Browser)
    │  HTTP (JSON)
    ▼
app.py  ──── Flask backend (routes: /, /init, /chat, /reset)
    │
    ├── Safety Layer
    │     ├── guardrail.pkl       — jailbreak / blacklist / manipulation patterns
    │     ├── crisis_detector.pkl — weighted regex → emergency / crisis / borderline / distress / safe
    │     └── psychosis_detector.pkl — hallucination & command-voice patterns
    │
    ├── Emotion Layer
    │     ├── emotion_detector_model.pkl  — LogisticRegression (TF-IDF)
    │     └── tfidf_vectorizer.pkl
    │
    ├── Response Layer
    │     ├── OpenAI GPT-4.1-mini  — dynamic empathetic replies
    │     └── response_templates.pkl  — offline fallback templates
    │
    └── Assessment Layer
          └── PHQ-9 / GAD-7 / WHO-5  (scored in-session, saved to assessments/)
```

---

## Conversation Flow

```
[Start] → STAGE_NAME → STAGE_ROLE → STAGE_LISTEN ⇄ STAGE_ASSESS → STAGE_RESPOND
                                          │
                              (crisis detected at any point)
                                          │
                                   Crisis Response + Helpline Referral
```

| Stage | What Happens |
|-------|-------------|
| **Name** | Collects user's name; skips to Listen if distress detected |
| **Role** | Classifies user (student / homemaker / daily worker / elderly / government) |
| **Listen** | Empathetic conversation via GPT-4.1-mini; safety checks on every message |
| **Assess** | 9-question PHQ/GAD/WHO screening with numeric scoring |
| **Respond** | Post-assessment support; severity-based referral |

---

## ML Models

| File | Type | Purpose |
|------|------|---------|
| `emotion_detector_model.pkl` | Logistic Regression | Classifies text into 10+ emotions (depression, anxiety, PTSD, OCD, grief, stress…) |
| `tfidf_vectorizer.pkl` | TF-IDF (10k features, bigrams) | Text → numeric vectors |
| `crisis_detector.pkl` | Weighted regex rules | 5-level crisis scoring |
| `psychosis_detector.pkl` | Regex patterns | Hallucination / command-voice detection |
| `guardrail.pkl` | Pattern lists | Blocks jailbreaks, harmful framing, manipulation |
| `response_templates.pkl` | Dict | Fallback responses by emotion × severity × role |
| `ml_crisis_model.pkl` + `ml_crisis_tfidf.pkl` | ML crisis classifier | Secondary crisis detection |

---

## Training Pipeline

### Emotion Detector (`train_emotion_detector.py`)
1. Load `data/master_dataset.csv` (multi-source Reddit + counseling corpora)
2. Clean text (lowercase, strip URLs/punctuation)
3. Filter labels with < 100 examples
4. TF-IDF vectorize → Logistic Regression (`class_weight='balanced'`)
5. Save `emotion_detector_model.pkl` + `tfidf_vectorizer.pkl`

### Clinical Score Models (`mental_health_combined.py`)
1. Merge English survey CSV (18 rows) + Urdu XLSX (4 rows)
2. Compute PHQ-9, GAD-7, WHO-5 totals
3. Label: Minimal / Mild / Moderate / Severe (PHQ/GAD); Good / Low (WHO-5)
4. Train 3 Random Forest classifiers (one per scale)
5. Save `combined_dataset.csv`

---

## Data Sources

| File | Description |
|------|-------------|
| `survey.csv` | 18-row English PHQ/GAD/WHO survey responses |
| `english.xlsx` | 4-row Urdu-translated survey responses |
| `data/master_dataset.csv` | Merged training corpus for emotion classifier |
| `data/external/reddit_mental_health.csv` | Reddit posts with mental health labels |
| `data/external/counsel_chat.csv` | CounselChat Q&A pairs |
| `data/external/empathetic_counseling.csv` | Counselor dialogue dataset |
| `data/external/mental_chat_16k.csv` | 16k mental health conversations |
| `data/external/mental_health_counseling.csv` | Counseling transcripts |
| `data/500_anonymized_Reddit_users_posts_labels.csv` | Labeled Reddit posts |

---

## Safety Features

- **Crisis override** — hard-coded phrase list triggers immediate emergency response
- **Grief intent** — regex detects "joining the dead" framing
- **Psychosis detection** — awake hallucinations vs. dream grief (different responses)
- **Guardrail** — blocks jailbreaks, framing attacks, synonym obfuscation, manipulation
- **Text normalization** — handles typos (`sucide`, `kll myself`, leet-speak, repeated chars)
- **PHQ-9 Q9 gate** — score ≥ 2 on suicidal ideation question triggers crisis path
- **Helplines surfaced** — Umang `0317-4288665` and `1122` in all crisis responses

---

## Frontend (`templates/index.html`, `static/`)

- Single-page chat UI (no framework)
- Message types styled differently: normal / crisis (amber) / emergency (red)
- Typing indicator, auto-resize textarea, timestamp per message
- "New Chat" button calls `/reset`

---

## Assessment Output

Completed sessions saved as JSON to `assessments/`:
```json
{
  "resourceType": "MentalHealthAssessment",
  "patient_name": "...",
  "phq9_total": 12,
  "gad7_total": 8,
  "who5_total": 14,
  "severity": "Moderate",
  "flagged_for_followup": true
}
```

---

## Key Configuration

| Setting | Value |
|---------|-------|
| LLM | `gpt-4.1-mini` via OpenAI Responses API |
| Max output tokens | 180 |
| History window | Last 12 turns (8 sent to LLM) |
| Session secret | `mental_health_fyp_2026` |
| Port | `5000` |

---

## Running the App

```bash
export OPENAI_API_KEY=sk-...
python app.py
# → http://localhost:5000
```
