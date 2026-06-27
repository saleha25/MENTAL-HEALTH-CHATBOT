import os
import re
import csv
import sqlite3
import pickle
import json
import random
import hashlib
import numpy as np

from datetime import datetime, timedelta
from functools import wraps
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify, render_template, session, g, redirect, url_for
from openai import OpenAI


app = Flask(__name__)
app.secret_key = "mental_health_fyp_2026"
app.permanent_session_lifetime = timedelta(days=30)

# Set your real API key in PowerShell before running:
# $env:OPENAI_API_KEY="your_real_api_key"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
DB_PATH = os.path.join(BASE_PATH, "empath_ai.db")

print("Loading models...")

with open(os.path.join(DATA_PATH, "crisis_detector.pkl"), "rb") as f:
    crisis_data = pickle.load(f)

with open(os.path.join(DATA_PATH, "emotion_detector_model.pkl"), "rb") as f:
    emotion_model = pickle.load(f)

with open(os.path.join(DATA_PATH, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(DATA_PATH, "guardrail.pkl"), "rb") as f:
    guardrail_data = pickle.load(f)

RETRIEVAL_DATA = None
retrieval_path = os.path.join(DATA_PATH, "retrieval_system.pkl")
if os.path.exists(retrieval_path):
    with open(retrieval_path, "rb") as f:
        RETRIEVAL_DATA = pickle.load(f)
    print(f"Retrieval system loaded: {len(RETRIEVAL_DATA['qa_pairs'])} QA pairs")
else:
    print("WARNING: retrieval_system.pkl not found!")

CRISIS_PATTERNS = crisis_data["patterns"]
FINALITY_WORDS = crisis_data["finality_words"]
ACTION_WORDS = crisis_data["action_words"]
RELEASE_WORDS = crisis_data["release_words"]

BLOCKED_TOPICS = guardrail_data["blacklist_combos"]
FRAMING_PATTERNS = guardrail_data["framing_patterns"]
JAILBREAK_PATTERNS = guardrail_data["jailbreak_patterns"]
ESCALATION_PATTERNS = guardrail_data["escalation_patterns"]
SYNONYM_PATTERNS = guardrail_data["synonym_patterns"]
MANIPULATION_PATTERNS = guardrail_data["manipulation_patterns"]

print("All models loaded!")

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER UNIQUE,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'default',
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            living_place TEXT DEFAULT '',
            FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wellbeing_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            log_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            sleep_hours REAL,
            mood TEXT,
            energy TEXT,
            source_text TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sender TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id INTEGER PRIMARY KEY,
            is_displaced INTEGER DEFAULT 0,
            living_place TEXT DEFAULT '',
            main_stressors TEXT DEFAULT '[]',
            sleep_pattern TEXT DEFAULT '',
            mood_pattern TEXT DEFAULT '',
            energy_pattern TEXT DEFAULT '',
            risk_pattern TEXT DEFAULT '',
            preferred_support_style TEXT DEFAULT 'gentle, natural, culturally aware support',
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            phq9_total INTEGER,
            gad7_total INTEGER,
            who5_total INTEGER,
            severity TEXT,
            emotion_detected TEXT,
            displacement_cause TEXT,
            idp_hardships TEXT,
            advice_given TEXT,
            prev_phq9 INTEGER,
            prev_gad7 INTEGER,
            prev_who5 INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Migrate existing DB — add columns if missing
    for col_sql in [
        "ALTER TABLE users ADD COLUMN account_id INTEGER UNIQUE",
        "ALTER TABLE users ADD COLUMN living_place TEXT DEFAULT ''",
        "ALTER TABLE assessments ADD COLUMN advice_given TEXT",
        "ALTER TABLE assessments ADD COLUMN prev_phq9 INTEGER",
        "ALTER TABLE assessments ADD COLUMN prev_gad7 INTEGER",
        "ALTER TABLE assessments ADD COLUMN prev_who5 INTEGER",
    ]:
        try:
            cursor.execute(col_sql)
        except Exception:
            pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            dob TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_table_columns(table_name):
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    except Exception:
        return []

def users_has_account_id():
    return "account_id" in get_table_columns("users")

def normalize_name(name):
    return re.sub(r"\s+", " ", name.strip().lower())

def find_existing_user(name):
    db = get_db()
    cursor = db.cursor()
    norm_name = normalize_name(name)
    cursor.execute("""
        SELECT * FROM users
        WHERE lower(trim(name)) = ?
        ORDER BY id ASC
        LIMIT 1
    """, (norm_name,))
    return cursor.fetchone()

def create_user(name, account_id=None):
    db = get_db()
    cursor = db.cursor()
    clean_name = name.strip()

    try:
        if users_has_account_id():
            cursor.execute("""
                INSERT INTO users (name, account_id)
                VALUES (?, ?)
            """, (clean_name, account_id))
        else:
            cursor.execute("""
                INSERT INTO users (name)
                VALUES (?)
            """, (clean_name,))
        db.commit()
        return cursor.lastrowid
    except Exception:
        cursor.execute("""
            INSERT INTO users (name)
            VALUES (?)
        """, (clean_name,))
        db.commit()
        return cursor.lastrowid

def get_user_by_account_id(account_id):
    if not account_id or not users_has_account_id():
        return None

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE account_id = ? LIMIT 1", (account_id,))
    return cursor.fetchone()


def get_or_create_user_for_account(account_id, account_name):
    existing = get_user_by_account_id(account_id)
    if existing:
        return existing

    first_name = (account_name or "User").strip().split()[0].capitalize()
    new_user_id = create_user(first_name, account_id=account_id)
    return get_user_by_id(new_user_id)

def update_user_last_seen(user_id):
    db = get_db()
    db.execute("""
        UPDATE users
        SET last_seen = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (user_id,))
    db.commit()

def update_user_role(user_id, role):
    db = get_db()
    db.execute("""
        UPDATE users
        SET role = ?, last_seen = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (role, user_id))
    db.commit()

def update_user_living_place(user_id, living_place):
    if not user_id or not living_place:
        return
    if "living_place" not in get_table_columns("users"):
        return
    db = get_db()
    db.execute("""
        UPDATE users
        SET living_place = ?, last_seen = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (living_place, user_id))
    db.commit()

def get_row_value(row, key, default=""):
    try:
        if row and key in row.keys():
            return row[key] if row[key] is not None else default
    except Exception:
        pass
    return default

def detect_living_place(text):
    t = normalize_for_safety(text)
    if any(x in t for x in ["ngo", "ngo camp", "ngo shelter"]):
        return "an NGO"
    if any(x in t for x in ["camp", "tent", "relief camp"]):
        return "a camp"
    if any(x in t for x in ["shelter", "temporary shelter"]):
        return "a shelter"
    if any(x in t for x in ["relatives", "relative", "cousin", "uncle", "aunt", "family home"]):
        return "with relatives"
    if any(x in t for x in ["friend", "friends"]):
        return "with friends"
    if any(x in t for x in ["rented", "rent", "rental"]):
        return "a rented place"
    if any(x in t for x in ["host family", "host"]):
        return "with a host family"
    return ""

def user_is_idp_context(sess):
    """Empath AI is designed for IDPs in Pakistan, so IDP context is default.

    We keep this as background context, not a phrase repeated to the user.
    Only turn it off if the user clearly says they are not displaced / not an IDP.
    """
    history = list(sess.get("history", []))
    full_text = " ".join([
        str(item.get("content", "")).lower()
        for item in history
        if item.get("role") == "user"
    ])

    not_idp_phrases = [
        "i am not displaced", "i'm not displaced", "i am not an idp", "i'm not an idp",
        "not an idp", "not displaced", "i was never displaced"
    ]
    if any(p in full_text for p in not_idp_phrases):
        return False

    return True

def get_user_by_id(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()

def save_chat_message(user_id, sender, message):
    db = get_db()
    db.execute("""
        INSERT INTO chat_history (user_id, sender, message)
        VALUES (?, ?, ?)
    """, (user_id, sender, message))
    db.commit()


def _memory_json_list(value):
    try:
        data = json.loads(value or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _memory_join_unique(old_items, new_items, limit=8):
    items = list(old_items or [])
    for item in new_items:
        if item and item not in items:
            items.append(item)
    return items[-limit:]

def ensure_user_memory(user_id):
    if not user_id:
        return None
    db = get_db()
    row = db.execute("SELECT * FROM user_memory WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        db.execute("INSERT INTO user_memory (user_id, main_stressors) VALUES (?, ?)", (user_id, json.dumps([])))
        db.commit()
        row = db.execute("SELECT * FROM user_memory WHERE user_id = ?", (user_id,)).fetchone()
    return row

def update_user_memory_profile(user_id, user_message, emotion="unknown", crisis_level="safe"):
    if not user_id or not user_message:
        return
    db = get_db()
    row = ensure_user_memory(user_id)
    if not row:
        return
    t = user_message.lower()
    is_displaced = int(row["is_displaced"] or 0)
    if any(k in t for k in ["displaced", "displacement", "idp", "left my home", "lost my home", "forced to leave", "camp", "shelter", "ngo", "temporary place"]):
        is_displaced = 1
    living_place = row["living_place"] or ""
    detected_place = detect_living_place(user_message)
    if detected_place:
        living_place = detected_place
        try:
            update_user_living_place(user_id, detected_place)
        except Exception:
            pass
    stressors = _memory_json_list(row["main_stressors"])
    found = []
    if any(k in t for k in ["study", "studies", "student", "exam", "class", "university", "focus"]): found.append("studies/focus")
    if detect_family_stress(t) or any(k in t for k in ["children pressure", "kids pressure", "parents pressure", "family responsibility", "too much responsibility"]): found.append("family responsibilities")
    if any(k in t for k in ["home", "miss my home", "lost my home", "left my home"]): found.append("loss of home")
    if any(k in t for k in ["camp", "shelter", "ngo", "crowded", "privacy", "noise"]): found.append("temporary living conditions")
    if any(k in t for k in ["job", "work", "income", "money", "fees", "expense"]): found.append("financial/work pressure")
    stressors = _memory_join_unique(stressors, found)
    extracted = extract_wellbeing_info(user_message)
    sleep_pattern = row["sleep_pattern"] or ""
    mood_pattern = row["mood_pattern"] or ""
    energy_pattern = row["energy_pattern"] or ""
    if extracted.get("sleep_hours") is not None:
        h = extracted["sleep_hours"]
        sleep_pattern = f"very low sleep around {h:g} hours" if h <= 4 else f"short sleep around {h:g} hours" if h <= 6 else f"sleep around {h:g} hours"
    elif extracted.get("mentioned_sleep"):
        sleep_pattern = "sleep difficulty mentioned"
    if extracted.get("mood"):
        mood_pattern = extracted["mood"]
    elif extracted.get("mentioned_mood") or emotion in ["depression", "sadness", "grief", "anxiety", "fear", "stress", "ptsd"]:
        mood_pattern = emotion if emotion != "unknown" else (mood_pattern or "emotional distress")
    if extracted.get("energy"):
        energy_pattern = extracted["energy"]
    elif extracted.get("mentioned_energy"):
        energy_pattern = "low/changed energy mentioned"
    risk_pattern = row["risk_pattern"] or ""
    if crisis_level in ["crisis", "emergency"]:
        risk_pattern = crisis_level
    elif any(k in t for k in ["hopeless", "nothing matters", "meaningless", "empty"]):
        risk_pattern = "low hope / emotional emptiness"
    support_style = row["preferred_support_style"] or "gentle, natural, culturally aware support"
    if any(k in t for k in ["quran", "allah", "islam", "religion"]):
        support_style = "gentle support with Islamic/Quran reference when appropriate"
    db.execute("""
        UPDATE user_memory
        SET is_displaced=?, living_place=?, main_stressors=?, sleep_pattern=?, mood_pattern=?, energy_pattern=?, risk_pattern=?, preferred_support_style=?, last_updated=CURRENT_TIMESTAMP
        WHERE user_id=?
    """, (is_displaced, living_place, json.dumps(stressors), sleep_pattern, mood_pattern, energy_pattern, risk_pattern, support_style, user_id))
    db.commit()

def get_user_memory_profile(user_id):
    if not user_id:
        return ""
    row = ensure_user_memory(user_id)
    if not row:
        return ""
    parts = []
    if row["is_displaced"]:
        parts.append(f"User is displaced and staying in {row['living_place']}." if row["living_place"] else "User is an internally displaced person in Pakistan.")
    stressors = _memory_json_list(row["main_stressors"])
    if stressors: parts.append("Recurring stressors: " + ", ".join(stressors) + ".")
    if row["sleep_pattern"]: parts.append("Sleep pattern: " + row["sleep_pattern"] + ".")
    if row["mood_pattern"]: parts.append("Mood pattern: " + row["mood_pattern"] + ".")
    if row["energy_pattern"]: parts.append("Energy pattern: " + row["energy_pattern"] + ".")
    if row["risk_pattern"]: parts.append("Risk pattern: " + row["risk_pattern"] + ".")
    if row["preferred_support_style"]: parts.append("Preferred support: " + row["preferred_support_style"] + ".")
    return " ".join(parts)

def save_wellbeing_log(user_id, sleep_hours=None, mood=None, energy=None, source_text=""):
    db = get_db()
    db.execute("""
        INSERT INTO wellbeing_logs (user_id, sleep_hours, mood, energy, source_text)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, sleep_hours, mood, energy, source_text))
    db.commit()

def get_latest_wellbeing_log(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM wellbeing_logs
        WHERE user_id = ?
        ORDER BY log_date DESC, id DESC
        LIMIT 1
    """, (user_id,))
    return cursor.fetchone()

def get_recent_wellbeing_logs(user_id, limit=5):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM wellbeing_logs
        WHERE user_id = ?
        ORDER BY log_date DESC, id DESC
        LIMIT ?
    """, (user_id, limit))
    return cursor.fetchall()

def get_last_chat_message(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT id, sender, message, timestamp
        FROM chat_history
        WHERE user_id = ?
        ORDER BY timestamp DESC, id DESC
        LIMIT 1
    """, (user_id,))
    return cursor.fetchone()


def get_latest_assessment(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT id, assessment_date
        FROM assessments
        WHERE user_id = ?
        ORDER BY assessment_date DESC, id DESC
        LIMIT 1
    """, (user_id,))
    return cursor.fetchone()


def has_real_user_history(user_row):
    if not user_row:
        return False

    role = (user_row["role"] or "default").strip().lower()
    if role != "default":
        return True

    if get_last_chat_message(user_row["id"]):
        return True

    if get_latest_assessment(user_row["id"]):
        return True

    return False

def get_last_assessment_for_user(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM assessments
        WHERE user_id = ?
        ORDER BY assessment_date DESC
        LIMIT 1
    """, (user_id,))
    return cursor.fetchone()

def days_since_assessment(row):
    if not row:
        return 999
    try:
        last_date = datetime.fromisoformat(str(row["assessment_date"]))
        return (datetime.now() - last_date).days
    except Exception:
        return 999

def save_assessment_to_db(user_id, session_data, advice="", prev=None):
    try:
        db = get_db()
        db.execute("""
            INSERT INTO assessments (
                user_id, phq9_total, gad7_total, who5_total,
                severity, emotion_detected, displacement_cause, idp_hardships,
                advice_given, prev_phq9, prev_gad7, prev_who5
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            sum(session_data.get("phq", {}).values()),
            sum(session_data.get("gad", {}).values()),
            sum(session_data.get("who", {}).values()),
            session_data.get("severity", "Unknown"),
            session_data.get("emotion", "Unknown"),
            session_data.get("displacement_cause", "Unknown"),
            json.dumps(session_data.get("idp_hardships", [])),
            advice,
            prev["phq9_total"] if prev else None,
            prev["gad7_total"] if prev else None,
            prev["who5_total"] if prev else None,
        ))
        db.commit()
    except Exception as e:
        print("Assessment DB save error:", e)


def generate_assessment_advice(phq_total, gad_total, who_total, phq_q9=0, prev=None):
    """Returns (advice_text, risk_level). risk_level: 'safe' | 'moderate' | 'high' | 'crisis'"""
    phq_sev = sa_get_phq9_severity(phq_total)[0]
    gad_sev = sa_get_gad7_severity(gad_total)[0]
    who_pct = who_total * 4
    risk_level = "safe"

    if phq_q9 >= 2 or phq_total >= 20:
        risk_level = "crisis"
    elif phq_total >= 15 or gad_total >= 15:
        risk_level = "high"
    elif phq_total >= 10 or gad_total >= 10:
        risk_level = "moderate"

    if gad_total >= 18 and phq_total >= 15 and who_pct <= 20:
        risk_level = "crisis"

    intro = (
        "Thank you for taking a moment to check in with yourself. "
        "That matters more than it may seem, and it helps show how things have been feeling lately."
    )

    if risk_level == "crisis":
        intro = (
            "Thank you for answering these questions honestly. "
            "From your answers, it looks like things may feel very heavy right now, so I want to speak gently and clearly with you."
        )
    elif risk_level == "high":
        intro = (
            "Thank you for being honest in your answers. "
            "It looks like you have been carrying a lot recently, and that kind of weight can make everyday life feel much harder."
        )
    elif risk_level == "moderate":
        intro = (
            "Thank you for checking in with yourself. "
            "Your answers suggest that some things have been weighing on you, even if not every moment feels equally heavy."
        )

    mood_line = ""
    if phq_sev == "Minimal":
        mood_line = (
            "Your mood score does not look deeply heavy right now, and that is a good sign. "
            "Even so, it is still worth protecting your routine, your rest, and your connection with people who help you feel steady."
        )
    elif phq_sev == "Mild":
        mood_line = (
            "It seems there is some sadness or emotional heaviness sitting quietly in the background. "
            "That does not mean you are weak — it may simply mean your heart needs a little more care than usual these days."
        )
    elif phq_sev == "Moderate":
        mood_line = (
            "Your answers suggest that sadness has been affecting you in a more noticeable way. "
            "When that happens, even simple things can start to feel harder, so it is important to be gentle with yourself and not ignore it."
        )
    else:
        mood_line = (
            "Your answers suggest that your emotional pain may be quite intense right now. "
            "If everything feels overwhelming, please remember that this is exactly the kind of time when support matters most."
        )

    anxiety_line = ""
    if gad_sev == "Mild":
        anxiety_line = (
            "There also seems to be some worry building up inside you. "
            "That kind of anxious tension can quietly drain your energy, especially when life already feels uncertain."
        )
    elif gad_sev == "Moderate":
        anxiety_line = (
            "It looks like worry has been taking up real space in your mind. "
            "That can leave a person feeling restless, tense, and unable to fully settle, even when they want to."
        )
    elif gad_sev == "Severe":
        anxiety_line = (
            "Your answers suggest that worry and inner tension may be hitting you very strongly right now. "
            "When anxiety reaches that level, it can affect sleep, concentration, and even the body itself."
        )

    wellbeing_line = ""
    if who_pct < 28:
        wellbeing_line = (
            "Your wellbeing score also looks quite low, which can make the day feel flat, tiring, or emotionally empty. "
            "If that is how it has been feeling, please know that it makes sense to feel worn down."
        )
    elif who_pct < 52:
        wellbeing_line = (
            "Your wellbeing seems a little low too, which can show up as reduced energy, less motivation, or feeling disconnected from the day. "
            "That does not mean something is wrong with you — it may simply mean you have been carrying too much for too long."
        )
    else:
        wellbeing_line = (
            "Your wellbeing score is not extremely low, which is encouraging. "
            "That means there is still some strength to build on, even if things do not feel fully settled right now."
        )

    guidance_parts = []

    if risk_level == "crisis":
        guidance_parts.append(
            "Right now, please do not stay alone with thoughts of harming yourself. "
            "Reach out to someone near you and call Umang (0317-4288665) or 1122 as soon as possible."
        )
    elif risk_level == "high":
        guidance_parts.append(
            "If you can, please talk to someone you trust and consider speaking with a mental health professional or doctor soon. "
            "You do not have to wait until things get even worse before asking for support."
        )
    elif risk_level == "moderate":
        guidance_parts.append(
            "This may be a good time to slow things down and care for yourself a bit more intentionally. "
            "Small things really do count here — steady sleep, a little fresh air, one trusted conversation, and fewer long hours alone."
        )
    else:
        guidance_parts.append(
            "Even though your scores are not showing a severe level right now, it is still worth taking your feelings seriously. "
            "Little habits done consistently can protect you before things quietly build up."
        )

    if gad_total >= 5:
        guidance_parts.append(
            "When worry starts building, try one very simple thing first: breathe in for 4, hold for 4, and breathe out for 6 a few times. "
            "It will not solve everything, but it can help your body feel a little safer."
        )

    if who_pct < 52:
        guidance_parts.append(
            "Try not to pressure yourself into doing a lot at once. "
            "A short walk, sitting outside for a little while, proper food, or a few quiet minutes with someone safe can be enough for one day."
        )

          # compare with previous assessment
    if prev and prev["phq9_total"] is not None:
        chg = phq_total - prev["phq9_total"]

        if chg >= 5:
            guidance_parts.append(
                f"Compared with your last check-in, your depression score has gone up by {chg} points. "
                "That suggests things may have become heavier recently, so this is a good time to be extra kind to yourself and not ignore the shift."
            )
        elif chg <= -5:
            guidance_parts.append(
                f"Compared with your last check-in, your depression score has improved by {abs(chg)} points. "
                "That is meaningful progress, even if you do not feel completely okay yet."
            )

    closing = (
        "Please remember this: the way you feel right now is important, but it is not the whole story of your life. "
        "If things start feeling too heavy, come back to the main chat and talk, and if it feels urgent, please call Umang (0317-4288665) or 1122."
    )

    parts = [intro, mood_line]

    if anxiety_line:
        parts.append(anxiety_line)

    if wellbeing_line:
        parts.append(wellbeing_line)

    parts.extend(guidance_parts)
    parts.append(closing)

    return "\n\n".join([p for p in parts if p]), risk_level
# ─────────────────────────────────────────────
# IDP CONTEXT
# ─────────────────────────────────────────────
DISPLACEMENT_KEYWORDS = {
    "flood": [
        "flood", "flooding", "flooded", "selaab", "barish", "baarish",
        "water came", "water level", "river", "dam broke", "washed away",
        "flood victims", "flood affected", "toofan"
    ],
    "conflict": [
        "conflict", "violence", "attack", "firing", "shooting", "bomb", "war",
        "military", "operation", "terrorists", "militants", "forced to leave",
        "had to escape", "fled", "running away", "danger", "unsafe", "threats"
    ],
    "earthquake": [
        "earthquake", "quake", "tremor", "bhookamp", "zalzala", "shaking",
        "building collapsed", "house fell", "rubble"
    ],
    "eviction": [
        "evicted", "kicked out", "landlord", "removed from house",
        "house demolished", "lost our place", "had nowhere to go"
    ],
    "fire": [
        "fire", "burned", "burnt", "our house burned", "caught fire", "aag"
    ],
}

def detect_displacement_cause(conversation_history):
    full_text = " ".join([
        item.get("content", "").lower()
        for item in conversation_history
        if item.get("role") == "user"
    ])

    scores = {cause: 0 for cause in DISPLACEMENT_KEYWORDS}
    for cause, keywords in DISPLACEMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in full_text:
                scores[cause] += 1

    best_cause = max(scores, key=scores.get)
    if scores[best_cause] > 0:
        return best_cause
    return "displacement"

def get_displacement_phrase(cause):
    phrases = {
        "flood": "after the floods forced your family to leave home",
        "conflict": "after being displaced due to violence and insecurity",
        "earthquake": "after the earthquake took away your home",
        "eviction": "after losing your home so suddenly",
        "fire": "after losing your home to fire",
        "displacement": "after being displaced from your home",
    }
    return phrases.get(cause, "after being displaced from your home")

IDP_HARDSHIP_KEYWORDS = {
    "loss_of_home": [
        "miss my home", "lost my home", "left home", "ghar chorna para",
        "ghar chor diya", "our house is gone", "apna ghar yaad aata hai",
        "my room", "my bed", "my house", "my village", "my area"
    ],
    "camp_stress": [
        "camp", "shelter", "tent", "ngo", "shared place", "crowded",
        "no privacy", "privacy nahi", "room mate", "host family", "temporary place"
    ],
    "family_burden": [
        "children", "kids", "bachay", "bacha", "family", "husband", "wife",
        "mother", "father", "parents", "responsibility", "ghar walay"
    ],
    "livelihood_loss": [
        "no job", "lost job", "jobless", "rozgar", "kamai", "income",
        "earn", "earning", "daily wage", "not working", "without work"
    ],
    "education_disruption": [
        "school", "college", "university", "study", "studies", "exam",
        "class", "semester", "fees", "education"
    ],
    "safety_fear": [
        "unsafe", "fear", "scared", "dar", "threat", "violence", "afraid",
        "not safe", "worried for safety"
    ],
    "grief_memory": [
        "old life", "used to live", "miss those days", "miss my village",
        "miss my room", "miss my area", "miss my things", "my home comfort"
    ],
}

IDP_SUPPORT_LINES = {
    "loss_of_home": [
        "Losing the comfort of home can leave a person feeling emotionally ungrounded.",
        "When home is no longer there, even small daily things can start hurting deeply.",
        "Missing your own room, routine, and familiar space can weigh heavily on the heart."
    ],
    "camp_stress": [
        "Living in a shared or temporary place can become mentally exhausting over time.",
        "When there is little privacy or rest, both the heart and mind can feel tired.",
        "Crowded living and constant adjustment can slowly drain a person emotionally."
    ],
    "family_burden": [
        "Trying to stay strong for family while carrying your own pain can be exhausting.",
        "Caring for others during displacement often leaves very little room for your own emotions.",
        "Many people carry family responsibilities quietly, even when they are already overwhelmed."
    ],
    "livelihood_loss": [
        "Losing work after displacement can bring both stress and emotional heaviness.",
        "Not knowing how to manage expenses can put a lot of pressure on a person.",
        "When income becomes uncertain, worry can stay in the mind all day."
    ],
    "education_disruption": [
        "When studies and routine are disrupted, even simple tasks can start feeling harder.",
        "Uncertainty about education and the future can make a person feel stuck and worried.",
        "Losing focus after so much change is a very human response."
    ],
    "safety_fear": [
        "When safety has been shaken, the body and mind can stay tense for a long time.",
        "After frightening experiences, it is common to feel alert, uneasy, or unsettled.",
        "Feeling unsafe can make rest, sleep, and peace very difficult."
    ],
    "grief_memory": [
        "Sometimes it is not only the house people miss, but the life they had inside it.",
        "Missing your old routine, your people, and your space can feel like a deep kind of grief.",
        "Memories of home can return strongly when life feels uncertain."
    ],
}

ROLE_CONTEXT_LINES = {
    "student": [
        "When studies, routine, and future plans are interrupted, the mind can feel pulled in many directions.",
        "For students, displacement can make focus, motivation, and hope for the future feel shaky."
    ],
    "homemaker": [
        "When someone is carrying home responsibilities in unstable conditions, the emotional burden can become very heavy.",
        "Trying to manage family needs away from home can leave a homemaker feeling worn out inside."
    ],
    "parent": [
        "Parents often carry double stress, their own pain and the pressure to stay strong for children.",
        "Worrying about children during displacement can keep the heart constantly heavy."
    ],
    "unemployed": [
        "When displacement also affects work and purpose, it can bring helplessness and emotional pain.",
        "Losing stability in both home and livelihood can make a person feel deeply stuck."
    ],
    "daily_worker": [
        "For someone who depends on daily earnings, uncertainty after displacement can feel overwhelming.",
        "When survival and income are both uncertain, stress can stay present from morning to night."
    ],
    "elderly": [
        "For older people, losing familiar surroundings can bring loneliness, fear, and emotional heaviness.",
        "Changes in place, routine, and support can affect elderly people very deeply."
    ],
    "government": [
        "Even when a person is trying to keep functioning, displacement can quietly build stress inside.",
        "Balancing responsibilities while carrying personal loss can become emotionally exhausting."
    ],
    "default": [
        "Displacement can affect a person's peace, routine, and emotional balance in many quiet ways.",
        "When life changes so suddenly, it is natural for the heart and mind to feel burdened."
    ],
}

def detect_idp_hardships(conversation_history):
    full_text = " ".join([
        item.get("content", "").lower()
        for item in conversation_history
        if item.get("role") == "user"
    ])

    scores = {}
    for hardship, keywords in IDP_HARDSHIP_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in full_text:
                score += 1
        scores[hardship] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked if item[1] > 0][:2]

def get_idp_context_line(role, hardships):
    lines = []

    role_lines = ROLE_CONTEXT_LINES.get(role, ROLE_CONTEXT_LINES["default"])
    if role_lines:
        lines.append(random.choice(role_lines))

    for hardship in hardships[:2]:
        if hardship in IDP_SUPPORT_LINES:
            lines.append(random.choice(IDP_SUPPORT_LINES[hardship]))

    if not lines:
        lines.append(random.choice(ROLE_CONTEXT_LINES["default"]))

    return " ".join(lines[:2])

# ─────────────────────────────────────────────
# WELLBEING EXTRACTION: SLEEP / MOOD / ENERGY
# ─────────────────────────────────────────────
MOOD_KEYWORDS = {
    "very_low": [
        "very sad", "depressed", "hopeless", "broken", "empty",
        "i want to cry", "i feel dead", "nothing matters"
    ],
    "low": [
        "sad", "down", "not okay", "not fine", "upset",
        "feeling bad", "low mood"
    ],
    "anxious": [
        "anxious", "worried", "scared", "fear", "panic",
        "ghabrahat", "bechain"
    ],
    "okay": [
        "okay", "fine", "normal", "manageable"
    ],
    "good": [
        "better", "good", "calm", "relaxed", "peaceful"
    ]
}

ENERGY_KEYWORDS = {
    "very_low": [
        "no energy", "completely tired", "dead tired", "exhausted",
        "drained", "can't move"
    ],
    "low": [
        "tired", "weak", "fatigue", "low energy",
        "thaka hua", "thaki hui"
    ],
    "medium": [
        "okay energy", "some energy", "manageable energy"
    ],
    "high": [
        "energetic", "active", "full energy", "motivated"
    ]
}

def extract_sleep_hours(text):
    t = text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr)\s*(?:of sleep|sleep)?",
        r"slept\s*(?:for\s*)?(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr)?",
        r"sleep(?:ing)?\s*(?:for\s*)?(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr)?",
        r"only slept\s*(\d+(?:\.\d+)?)",
        r"i slept\s*(\d+(?:\.\d+)?)",
        r"i can sleep at night\s*\.??\s*only\s*(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr)?",
        r"i sleep only\s*(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr)?",
        r"just\s*(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr)"
    ]

    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            try:
                hours = float(match.group(1))
                if 0 <= hours <= 24:
                    return hours
            except:
                pass

    word_number_map = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
    }

    for word, num in word_number_map.items():
        if re.search(rf"(slept|sleep|sleeping).{{0,10}}\b{word}\b", t) or re.search(rf"\b{word}\b.{{0,10}}(hours|hour|hrs|hr)", t):
            return float(num)

    match = re.search(r"(only|just)\s*(\d+(?:\.\d+)?)", t)
    if match:
        try:
            hours = float(match.group(2))
            if 0 <= hours <= 24:
                return hours
        except:
            pass

    return None

def extract_mood(text):
    t = text.lower()
    for label, keywords in MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return label
    return None

def extract_energy(text):
    t = text.lower()
    for label, keywords in ENERGY_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return label
    return None

def extract_wellbeing_info(text):
    sleep_hours = extract_sleep_hours(text)
    mood = extract_mood(text)
    energy = extract_energy(text)

    text_l = text.lower()

    mentioned_sleep = (
        sleep_hours is not None or
        any(word in text_l for word in ["sleep", "slept", "sleeping", "bed", "insomnia", "night", "awake", "rest"])
    )
    mentioned_mood = (
        mood is not None or
        any(word in text_l for word in ["mood", "sad", "down", "depressed", "okay", "fine", "hopeless", "anxious", "worst", "not feeling well", "not good"])
    )
    mentioned_energy = (
        energy is not None or
        any(word in text_l for word in ["energy", "tired", "exhausted", "drained", "fatigue", "weak"])
    )

    return {
        "sleep_hours": sleep_hours if mentioned_sleep else None,
        "mood": mood if mentioned_mood else None,
        "energy": energy if mentioned_energy else None,
        "mentioned_sleep": mentioned_sleep,
        "mentioned_mood": mentioned_mood,
        "mentioned_energy": mentioned_energy,
        "has_any": any([mentioned_sleep, mentioned_mood, mentioned_energy])
    }

def save_wellbeing_if_mentioned(user_id, user_message):
    if not user_id:
        return None

    extracted = extract_wellbeing_info(user_message)

    if not extracted["has_any"]:
        return None

    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        SELECT * FROM wellbeing_logs
        WHERE user_id = ?
        AND date(log_date) = date('now')
        ORDER BY id DESC
        LIMIT 1
    """, (user_id,))
    today_log = cursor.fetchone()

    if today_log:
        sleep = extracted["sleep_hours"] if extracted["sleep_hours"] is not None else today_log["sleep_hours"]
        mood = extracted["mood"] if extracted["mood"] is not None else today_log["mood"]
        energy = extracted["energy"] if extracted["energy"] is not None else today_log["energy"]

        new_source = user_message
        if today_log["source_text"] and user_message.strip().lower() not in today_log["source_text"].lower():
            new_source = f'{today_log["source_text"]} | {user_message}'

        cursor.execute("""
            UPDATE wellbeing_logs
            SET sleep_hours = ?, mood = ?, energy = ?, source_text = ?
            WHERE id = ?
        """, (sleep, mood, energy, new_source, today_log["id"]))
    else:
        cursor.execute("""
            INSERT INTO wellbeing_logs (user_id, sleep_hours, mood, energy, source_text)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            extracted["sleep_hours"],
            extracted["mood"],
            extracted["energy"],
            user_message
        ))

    db.commit()
    return extracted

# ─────────────────────────────────────────────
# SURVEY INSIGHTS
# ─────────────────────────────────────────────
def _parse_survey_score(val):
    m = re.match(r"^(\d+):", str(val).strip())
    return int(m.group(1)) if m else None

def load_idp_survey_insights():
    survey_path = os.path.join(BASE_PATH, "survey.csv")
    if not os.path.exists(survey_path):
        return {}

    PHQ_LABELS = [
        "low interest", "hopelessness", "sleep trouble", "low energy",
        "appetite issues", "self-failure feelings", "poor concentration",
        "psychomotor changes", "thoughts of self-harm"
    ]
    GAD_LABELS = [
        "nervousness", "uncontrollable worry", "excessive worry",
        "trouble relaxing", "restlessness", "irritability", "fear"
    ]
    PROF_MAP = {
        "student": "student",
        "unemployed": "unemployed",
        "daily wage": "daily_worker",
        "government": "government",
        "private employee": "government",
        "small business": "government",
        "shopkeeper": "government",
        "homemaker": "homemaker",
        "housewife": "homemaker",
        "developer": "government",
        "parent": "parent",
        "mother": "parent",
        "father": "parent",
    }

    groups = {}
    with open(survey_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        phq_indices = list(range(3, 12))
        gad_indices = list(range(12, 19))

        for row in reader:
            if len(row) < 19:
                continue

            prof_raw = row[2].strip().lower()
            role = "default"
            for key, mapped in PROF_MAP.items():
                if key in prof_raw:
                    role = mapped
                    break

            if role not in groups:
                groups[role] = {"phq": [[] for _ in PHQ_LABELS], "gad": [[] for _ in GAD_LABELS], "count": 0}

            groups[role]["count"] += 1

            for i, col in enumerate(phq_indices):
                s = _parse_survey_score(row[col])
                if s is not None:
                    groups[role]["phq"][i].append(s)

            for i, col in enumerate(gad_indices):
                s = _parse_survey_score(row[col])
                if s is not None:
                    groups[role]["gad"][i].append(s)

    insights = {}
    for role, data in groups.items():
        phq_means = {PHQ_LABELS[i]: round(sum(v) / len(v), 2) for i, v in enumerate(data["phq"]) if v}
        gad_means = {GAD_LABELS[i]: round(sum(v) / len(v), 2) for i, v in enumerate(data["gad"]) if v}
        insights[role] = {
            "count": data["count"],
            "top_phq": sorted(phq_means, key=phq_means.get, reverse=True)[:3],
            "top_gad": sorted(gad_means, key=gad_means.get, reverse=True)[:3],
        }
    return insights

IDP_SURVEY_INSIGHTS = load_idp_survey_insights()
print(f"IDP survey insights: {list(IDP_SURVEY_INSIGHTS.keys())}")

# ─────────────────────────────────────────────
# STAGES
# ─────────────────────────────────────────────
STAGE_NAME = "name"
STAGE_ROLE = "role"
STAGE_LISTEN = "listen"
STAGE_ASSESS = "assess"
STAGE_RESPOND = "respond"

ROLE_KEYWORDS = {
    "student": ["student", "university", "college", "school", "studies", "study", "exam", "education", "semester", "notes"],
    "homemaker": ["house wife", "housewife", "home maker", "homemaker", "mother", "mom", "maa", "children", "kids", "baby", "bacha"],
    "parent": ["parent", "mother", "father", "ammi", "abu", "my children", "my kids", "my child"],
    "unemployed": ["unemployed", "no job", "lost my job", "without work", "not working", "no work"],
    "daily_worker": ["worker", "labor", "labour", "construction", "daily wage", "mazdoor", "earner", "jobless"],
    "elderly": ["retired", "old", "elder", "grandfather", "grandmother", "haji", "buzurg", "old man", "old woman"],
    "government": ["government", "employee", "office", "job", "officer", "private employee", "shopkeeper", "business"],
}

GRIEF_INTENT_PATTERNS = [
    r"(should|going to|want to|will).{0,10}(go|join).{0,15}(them|him|her|dead|died|passed)",
    r"she is asking me to come",
    r"he is asking me to come",
    r"(dead|died|passed).{0,20}(asking|calling).{0,15}(come|join|go)",
    r"i should go.{0,15}(join|him|her|them)",
]

PSYCHOSIS_PATTERNS = {
    "command_hallucination": [
        (r"(voice|voices|it|they|someone).{0,20}(told|telling|says?|saying|ordering|commanding).{0,20}(hurt|harm|kill|cut|burn|attack|hit).{0,20}(myself|yourself|someone|them|him|her)", 10),
        (r"(voice|voices).{0,20}(told|telling|says?).{0,20}(end|stop|die|disappear)", 10),
    ],
    "hallucination": [
        (r"(see|seeing|saw|hear|hearing|heard).{0,20}(dead|passed away|died|ghost|spirit).{0,20}(person|people|someone|them|him|her|mother|father|friend)", 8),
        (r"(hear|hearing).{0,20}(voices?|sounds?|music|calling).{0,20}(nobody|no one|not there|not real)", 8),
    ],
    "paranoia": [
        (r"(they|everyone|people|neighbors?).{0,20}(trying|want|planning|going).{0,20}(kill|harm|hurt|poison|attack).{0,20}me", 7),
    ],
    "mania": [
        (r"(haven't|have not|didn't).{0,10}slept?.{0,20}(days?|weeks?).{0,20}(feel (great|amazing|fantastic|energetic|powerful))", 7),
    ],
}

QURAN_HOPE_VERSES = [
    {"ref": "Quran 39:53", "topic": "hope", "arabic": "لَا تَقْنَطُوا مِن رَّحْمَةِ اللَّهِ", "english": "Do not lose hope in Allah's mercy."},
    {"ref": "Quran 2:286", "topic": "burden", "arabic": "لَا يُكَلِّفُ ٱللَّهُ نَفْسًا إِلَّا وُسْعَهَا", "english": "Allah does not burden a soul beyond what it can bear."},
    {"ref": "Quran 94:5-6", "topic": "hardship", "arabic": "فَإِنَّ مَعَ ٱلْعُسْرِ يُسْرًا", "english": "Surely, with hardship comes ease."},
]

PSYCHOSIS_SUPPORT_VERSES = [
    {"ref": "Quran 13:28", "topic": "calm", "arabic": "أَلَا بِذِكْرِ اللَّهِ تَطْمَئِنُّ الْقُلُوبُ", "english": "Surely in the remembrance of Allah do hearts find comfort."},
    {"ref": "Quran 17:82", "topic": "healing", "arabic": "وَنُنَزِّلُ مِنَ ٱلْقُرْءَانِ مَا هُوَ شِفَآءٞ وَرَحْمَةٞ لِّلْمُؤْمِنِينَ", "english": "We send down the Quran as a healing and mercy for the believers."},
    {"ref": "Quran 2:255", "topic": "protection", "arabic": "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَيُّ ٱلْقَيُّومُ", "english": "Allah! There is no god worthy of worship except Him, the Ever-Living, All-Sustaining."},
]

SIGNATURE_LINES = [
    "Stay with me for a moment.",
    "Let's hold this gently.",
    "I'm listening with care.",
    "You do not have to carry all of this at once.",
    "Let's keep this moment simple and safe.",
    "We can take this one step at a time."
]

# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
def clean_text_for_retrieval(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def retrieve_from_training_data(user_text, emotion=None, role=None, top_k=2):
    if not RETRIEVAL_DATA:
        return None

    qa_pairs = RETRIEVAL_DATA["qa_pairs"]
    question_vectors = RETRIEVAL_DATA["question_vectors"]

    cleaned = clean_text_for_retrieval(user_text)
    user_vec = tfidf.transform([cleaned])
    sims = cosine_similarity(user_vec, question_vectors)[0].copy()

    for i, pair in enumerate(qa_pairs):
        if pair.get("source") == "idp":
            sims[i] *= 1.8
        if role and pair.get("topic") == role:
            sims[i] *= 1.3
        if emotion and pair.get("topic") == emotion:
            sims[i] *= 1.2

    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if sims[idx] > 0.05:
            results.append({
                "question": qa_pairs[idx]["question"],
                "answer": qa_pairs[idx]["answer"],
                "topic": qa_pairs[idx]["topic"],
                "source": qa_pairs[idx]["source"],
                "similarity": round(float(sims[idx]), 3)
            })

    return results if results else None



# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def init_session():
    session["stage"] = STAGE_NAME
    session["name"] = ""
    session["role"] = "default"
    session["history"] = []
    session["phq"] = {}
    session["gad"] = {}
    session["who"] = {}
    session["q_index"] = 0
    session["is_positive"] = False
    session["emotion"] = "unknown"
    session["confidence"] = 0.0
    session["severity"] = "Unknown"
    session["assessment_started"] = False
    session["crisis_active"] = False
    session["crisis_level"] = "safe"
    session["used_verses"] = []
    session["used_psychosis_verses"] = []
    session["displacement_cause"] = "displacement"
    session["idp_hardships"] = []
    session["user_id"] = None
    session["is_returning_user"] = False
    session["last_assessment_scores"] = {}
    session["last_advice"] = ""
    session["sa_followup_count"] = 0
    session["sa_followup_needed"] = False
    session["sa_listen_turns"] = 0
    session["sa_dominant_concern"] = "minimal"
    session["crisis_step"] = ""

def add_to_history(role, content):
    history = list(session.get("history", []))
    history.append({"role": role, "content": content})
    session["history"] = history[-14:]

def detect_role(text):
    text = text.lower().strip()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return role
    return "default"


def detect_family_stress(msg):
    """Detect when the user is describing family pressure/stress.

    Neutral phrases like "my family is with me" should not count as stress.
    """
    msg = (msg or "").lower()

    negative_words = [
        "stress", "problem", "issue", "pressure",
        "fight", "argument", "burden", "responsibility",
        "tension", "difficult", "hard", "toxic",
        "children pressure", "kids pressure", "parents pressure",
        "family responsibility", "too much responsibility"
    ]

    if "family" in msg and any(word in msg for word in negative_words):
        return True

    return any(phrase in msg for phrase in [
        "children pressure", "kids pressure", "parents pressure",
        "family responsibility", "too much responsibility",
        "taking care of family", "burden of family", "supporting family"
    ])


def is_metaphor_kill(text):
    """Return True when 'kill' is used as a harmless metaphor, not self-harm.

    Examples treated as safe: 'kill this exam', 'kill this project'.
    Examples NOT safe: 'kill myself', 'kill me', 'kill someone'.
    """
    t = normalize_for_safety(text or "")

    dangerous_targets = [
        "myself", "my self", "me", "yourself", "him", "her", "them",
        "someone", "person", "people", "friend", "mother", "father",
        "sister", "brother", "child", "wife", "husband"
    ]
    if any(f"kill {target}" in t for target in dangerous_targets):
        return False

    safe_patterns = [
        r"\bkill\s+(this|the|my)?\s*(exam|project|assignment|task|test|presentation|work|homework|quiz|paper)\b",
        r"\bkill\s+it\b",
        r"\bkill\s+(the\s+)?(game|competition)\b",
    ]

    return any(re.search(pattern, t) for pattern in safe_patterns)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"([a-z])-([a-z])", r"\1\2", text)
    text = text.replace("1", "i").replace("0", "o").replace("3", "e").replace("4", "a").replace("5", "s")
    return re.sub(r"(.)\1{2,}", r"\1", text)

def normalize_for_safety(text):
    t = text.lower().strip()
    replacements = {
        "sucide": "suicide",
        "suicde": "suicide",
        "kill my self": "kill myself",
        "end my self": "end myself",
        "dont want to live": "don't want to live",
        "wanna die": "want to die",
        "diee": "die",
        "myslef": "myself",
        "kll myself": "kill myself",
        "kil myself": "kill myself",
        "sui cide": "suicide",
        "freind": "friend",
        "frend": "friend",
        "cant": "can't",
        "wont": "won't",
        "dont": "don't",
    }
    for wrong, correct in replacements.items():
        t = t.replace(wrong, correct)
    return normalize_text(t)

def has_grief_intent(text):
    return any(re.search(p, text.lower()) for p in GRIEF_INTENT_PATTERNS)

def is_positive_message(text):
    t = clean_text(text)
    negative_patterns = [r"\bnot\s+good\b", r"\bnot\s+fine\b", r"\bnot\s+okay\b", r"\bi am sad\b", r"\bfeel bad\b", r"\bno point\b", r"\bworthless\b"]
    positive_patterns = [r"\bi am happy\b", r"\bi feel good\b", r"\bi am fine\b", r"\bi am okay\b", r"\bi feel better\b", r"\bi feel great\b"]

    for pattern in negative_patterns:
        if re.search(pattern, t):
            return False
    for pattern in positive_patterns:
        if re.search(pattern, t):
            return True
    return False

def wants_assessment(text):
    t = text.lower().strip()
    return any(trigger in t for trigger in [
        "start assessment", "start screening", "mental health assessment",
        "check my mental health", "screen me", "ask me questions",
        "start phq", "start gad", "assessment", "screening"
    ])

def looks_like_name(text):
    t = text.strip()
    if len(t.split()) > 3:
        return False

    lower_t = normalize_for_safety(t)
    bad_signals = [
        "kill", "die", "suicide", "sad", "depressed", "help", "anxious",
        "alone", "lost", "home", "stress", "cry", "hopeless", "hurt",
        "dream", "voices", "blood"
    ]
    if any(word in lower_t for word in bad_signals):
        return False

    return bool(re.fullmatch(r"[A-Za-z ]{2,30}", t))

def direct_crisis_override(text):
    high_risk = [
        "i want to die", "i want to kill myself", "kill myself",
        "commit suicide", "suicide", "end my life",
        "i don't want to live", "i do not want to live",
        "i will kill myself", "i am going to kill myself",
        "i should kill myself"
    ]
    return any(phrase in text for phrase in high_risk)

def check_guardrail(user_text):
    original = user_text.lower().strip()
    normalized = normalize_text(original)

    for text in [original, normalized]:
        for combo in BLOCKED_TOPICS:
            if all(w in text for w in combo):
                return {
                    "blocked": True,
                    "response": "I care about your wellbeing, so I cannot help with that. But I can listen to what you are going through right now."
                }
        for pattern in (FRAMING_PATTERNS + JAILBREAK_PATTERNS + ESCALATION_PATTERNS + SYNONYM_PATTERNS + MANIPULATION_PATTERNS):
            if re.search(pattern, text):
                return {
                    "blocked": True,
                    "response": "My purpose is to support your wellbeing. I cannot be redirected from that. If you want to talk, I am here."
                }
    return {"blocked": False}

def detect_crisis(text):
    t = text.lower().strip()
    scores = {k: 0 for k in CRISIS_PATTERNS}

    for level, patterns in CRISIS_PATTERNS.items():
        for pattern, weight in patterns:
            if re.search(pattern, t):
                scores[level] += weight

    finality = sum(3 for w in FINALITY_WORDS if w in t)
    action = sum(3 for w in ACTION_WORDS if w in t)
    release = sum(4 for w in RELEASE_WORDS if w in t)
    bonus = finality + action + release

    emergency = scores.get("emergency", 0) + bonus
    crisis = scores.get("crisis", 0) + scores.get("grief_crisis", 0) + scores.get("metaphor_crisis", 0) + scores.get("farewell", 0) + bonus
    self_harm = scores.get("self_harm", 0)
    distress = scores.get("distress", 0)

    enjoyment = any(w in t for w in ["feel good", "like", "enjoy", "love", "relief", "calm", "better"])

    if emergency >= 10:
        return "emergency"
    if self_harm >= 7 and not enjoyment:
        return "emergency"
    if self_harm >= 4 and enjoyment:
        return "crisis"
    if crisis >= 8:
        return "crisis"
    if self_harm >= 4:
        return "crisis"
    if crisis >= 4 or distress >= 7:
        return "borderline"
    if distress >= 3:
        return "distress"
    return "safe"

def detect_emotion(text):
    try:
        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])
        emotion = emotion_model.predict(vec)[0]
        proba = emotion_model.predict_proba(vec)[0]
        conf = round(float(max(proba)) * 100, 1)
        return emotion, conf
    except Exception as e:
        print("Emotion error:", e)
        return "emotional_support", 50.0

def detect_psychosis(text):
    """Detect psychosis-like experiences without diagnosing the user.

    Categories include hallucination, paranoia, delusional belief, disorganized/confusing thought,
    and grief-related dream content that escalates into a belief/urge to join a deceased person.
    """
    t = normalize_for_safety(text)
    detected = {}

    expanded_patterns = {
        "command_hallucination": [
            (r"(voice|voices|someone|they).{0,25}(told|telling|says?|saying|ordering|commanding).{0,30}(hurt|harm|kill|cut|burn|attack|hit|end|die|go|join)", 12),
            (r"(dead|died|passed away|deceased).{0,30}(calling|asking|telling).{0,30}(come|go|join)", 12),
            (r"(calling me).{0,30}(i should|should|want to|need to).{0,30}(go|join|come)", 11),
        ],
        "hallucination": [
            (r"\b(hear|hearing|heard)\b.{0,25}\b(voice|voices|calling|sounds?)\b", 8),
            (r"\b(see|seeing|saw)\b.{0,25}\b(person|people|shadow|figure|dead|ghost|spirit|someone)\b", 8),
            (r"\b(dead|died|passed away|deceased)\b.{0,25}\b(calling me|talking to me|speaking to me)\b", 8),
        ],
        "paranoia": [
            (r"\b(they|people|someone|neighbors?|everyone)\b.{0,30}\b(watching|following|tracking|spying|monitoring)\b.{0,20}\b(me|my family)?\b", 7),
            (r"\b(they|people|someone|neighbors?)\b.{0,30}\b(trying|planning|want).{0,20}(kill|harm|hurt|poison|attack)\b.{0,20}\bme\b", 9),
        ],
        "delusional_belief": [
            (r"\b(i know|i am sure|definitely|for sure)\b.{0,35}\b(controlling my mind|reading my thoughts|sending messages|special powers|chosen by)\b", 7),
            (r"\b(tv|radio|phone|signs|messages)\b.{0,30}\b(talking to me|sending me|warning me|controlling me)\b", 7),
        ],
        "disorganized_confusion": [
            (r"\b(my thoughts|mind)\b.{0,25}\b(broken|scattered|mixed up|not making sense|confused)\b", 4),
            (r"\b(i cannot tell|can't tell)\b.{0,25}\b(real|dream|true)\b", 6),
        ],
    }

    # Keep the original trained/loaded patterns too, if present.
    for category, patterns in PSYCHOSIS_PATTERNS.items():
        score = 0
        for pattern, weight in patterns:
            if re.search(pattern, t):
                score += weight
        if score > 0:
            detected[category] = detected.get(category, 0) + score

    for category, patterns in expanded_patterns.items():
        score = 0
        for pattern, weight in patterns:
            if re.search(pattern, t):
                score += weight
        if score > 0:
            detected[category] = detected.get(category, 0) + score

    if not detected:
        return {"psychosis_detected": False, "level": "safe"}

    top_category = max(detected, key=detected.get)
    top_score = detected[top_category]

    if top_category == "command_hallucination" or top_score >= 11:
        level = "EMERGENCY"
    elif top_score >= 7:
        level = "CRISIS"
    else:
        level = "DISTRESS"

    return {
        "psychosis_detected": True,
        "category": top_category,
        "score": top_score,
        "level": level,
    }


def detect_dream_or_hallucination_risk(text):
    """Separate grief dreams from possible psychosis/crisis escalation.

    A deceased person calling/crying inside a dream is treated as grief support unless the user
    expresses a direct urge to go, join, die, or harm themselves.
    """
    t = normalize_for_safety(text)

    dream_patterns = [
        r"\b(friend|mother|father|sister|brother|wife|husband|child|someone)\b.*\bdream\b.*\b(calling me|talking to me|came|appeared|crying)\b",
        r"\bi saw .* in my dream\b",
        r"\bsomeone came in my dream\b",
        r"\bmy dead .* (came|appeared|was there|calling me|crying)\b.*\bdream\b",
        r"\b(dead|died|passed away|deceased)\b.{0,35}\b(calling|crying|talking)\b.{0,35}\b(after waking|woke up|dream)\b",
    ]

    direct_intent_patterns = [
        r"\b(i should|should|want to|need to|have to|will)\b.{0,20}\b(go|join|come)\b.{0,20}\b(her|him|them|dead|friend|mother|father)\b",
        r"\bgo with (her|him|them)\b",
        r"\bjoin (her|him|them)\b",
        r"\b(want to die|kill myself|end my life|suicide)\b",
    ]

    awake_patterns = [
        r"\bi hear .* calling me\b",
        r"\bvoices? .* talking to me\b",
        r"\bshe is telling me to come\b",
        r"\bhe is telling me to come\b",
        r"\btelling me to join\b",
        r"\bsomeone standing in my room\b",
        r"\bwhile awake\b",
        r"\bwhen i am awake\b",
    ]

    if any(re.search(p, t) for p in direct_intent_patterns):
        return {"detected": True, "type": "psychosis_grief_escalation", "level": "crisis"}

    if any(re.search(p, t) for p in dream_patterns):
        return {
            "detected": True,
            "type": "dream_grief",
            "level": "support",
            "response": (
                "That sounds really upsetting, and I am glad you told me. Dreams about someone who has died can feel heavy, especially when your mind has been under a lot of strain. "
                "You do not have to sit with this alone. Would you feel okay talking to someone you trust about how this dream affected you?"
            )
        }

    if any(re.search(p, t) for p in awake_patterns):
        return {"detected": True, "type": "psychosis_or_command", "level": "crisis"}

    return {"detected": False}



def is_grief_dream_without_self_harm(text):
    """Detect grief/disturbing experiences about a deceased person without automatic crisis.

    This is intentionally a safety nuance, not normal response routing. Crisis still wins when
    the user says they want/need/should go, join the deceased person, die, or harm themselves.
    It also treats phrases like "after waking up" as dream context, because users often continue
    the same dream story without repeating the word "dream" every turn.
    """
    t = normalize_for_safety(text or "")

    deceased_context = any(x in t for x in [
        "dead friend", "dead mother", "dead father", "dead sister", "dead brother",
        "my friend died", "friend who died", "passed away", "deceased", "someone who died"
    ]) or ("dead" in t and any(x in t for x in ["friend", "mother", "father", "sister", "brother", "person", "someone"]))

    dream_context = any(x in t for x in [
        "dream", "dreams", "in my dream", "in my dreams", "after waking", "when i woke", "woke up"
    ])

    grief_content = any(x in t for x in [
        "calling me", "crying", "came", "appeared", "saw", "see", "talking to me", "i am scared", "scared"
    ])

    crisis_intent = any(x in t for x in [
        "i should go", "i want to go", "i need to go", "i have to go", "i will go",
        "go to her", "go to him", "join her", "join him", "join them",
        "go with her", "go with him", "want to die", "kill myself", "end my life", "suicide"
    ])

    awake_hallucination_context = any(x in t for x in [
        "while awake", "when i am awake", "i hear voices", "voices are", "voice is", "standing in my room"
    ])

    return bool(deceased_context and grief_content and not crisis_intent and not awake_hallucination_context and dream_context)

def build_grief_dream_support_response(user_text, name="", role="default"):
    """Warm support for grief-related dreams without over-triggering crisis."""
    display_name = (name or "").strip()
    prefix = f"{display_name}, " if display_name else ""

    if os.environ.get("OPENAI_API_KEY"):
        try:
            system_prompt = """
You are Empath AI, a warm mental-health support chatbot for IDPs in Pakistan.
The user is describing a grief-related dream about someone who died. They have NOT expressed wanting to die or join the deceased person.
Respond with emotional support, not emergency panic.
Do not validate the dream as supernatural or real. Do not dismiss it.
Include: gentle validation, one hopeful IDP-sensitive line, a short dua/prayer for the deceased if appropriate, and exactly one simple follow-up question.
Keep it 4-6 sentences, warm and natural.
""".strip()
            prompt = f"User name: {display_name or 'User'}\nRole: {role}\nUser message: {user_text}\nWrite the next response."
            result = client.chat.completions.create(
                model=os.environ.get("EMPATH_AI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.75,
                max_tokens=180,
            )
            response = result.choices[0].message.content.strip()
            return _ensure_one_followup(response, user_text) if '_ensure_one_followup' in globals() else response
        except Exception as e:
            print("Grief dream OpenAI response error:", e)

    options = [
        (
            f"{prefix}seeing your friend like that in a dream sounds really painful and unsettling. "
            "Dreams about someone we have lost can bring sadness back very strongly, especially when the heart is already carrying a lot. "
            "After everything you have been managing, your mind and heart may be asking for gentleness right now. "
            "You can make dua for your friend — may Allah grant them peace, mercy, and light. "
            "How did you feel when you woke up from that dream?"
        ),
        (
            f"{prefix}that dream sounds heavy, especially because it was about someone you cared about. "
            "When grief is still inside us, the mind can bring that person into dreams in emotional ways. "
            "Even with so much change around you, talking about it is a strong step instead of holding it alone. "
            "May Allah give your friend peace and give your heart comfort too. "
            "What part of the dream stayed with you the most?"
        ),
        (
            f"{prefix}I can understand why that would shake you — seeing your friend crying in a dream can feel very real and upsetting. "
            "It may be your grief showing how much you still care about them, not something you have to face alone. "
            "When life feels unsettled, painful memories and dreams can feel even stronger, but sharing them can bring a little ease. "
            "Please make a gentle dua for your friend, and be kind to yourself tonight too. "
            "Would you like to tell me what happened in the dream?"
        ),
    ]
    return random.choice(options)
def run_safety_checks(user_text):
    normalized = normalize_for_safety(user_text)

    # Grief-related dreams should receive emotional support unless the user expresses
    # a direct wish/plan to die, go with, or join the deceased person.
    if is_grief_dream_without_self_harm(normalized):
        return {"type": "grief_dream", "level": "support"}

    if direct_crisis_override(normalized):
        return {"type": "emergency", "level": "emergency"}

    # Prevent false-positive crisis detection for phrases like "kill this exam/project".
    if is_metaphor_kill(normalized):
        return {"type": "safe", "level": "safe", "metaphor_kill": True}

    guardrail = check_guardrail(normalized)
    if guardrail["blocked"]:
        return {"type": "blocked", "response": guardrail["response"]}

    # Psychosis-like experiences should be detected before general grief-crisis intent,
    # so the response can stay calm, grounding, and clinically safer.
    dream_risk = detect_dream_or_hallucination_risk(normalized)
    if dream_risk["detected"]:
        if dream_risk["level"] == "crisis":
            return {"type": "psychosis", "level": "crisis", "category": dream_risk.get("type", "psychosis_risk")}
        return {"type": "special_support", "level": "support", "response": dream_risk["response"]}

    psychosis = detect_psychosis(normalized)
    if psychosis["psychosis_detected"]:
        level = "emergency" if psychosis["level"] == "EMERGENCY" else "crisis" if psychosis["level"] == "CRISIS" else "distress"
        return {"type": "psychosis", "level": level, "category": psychosis["category"]}

    if has_grief_intent(normalized):
        return {"type": "crisis", "level": "crisis"}

    crisis_level = detect_crisis(normalized)
    if crisis_level == "emergency":
        return {"type": "emergency", "level": "emergency"}
    if crisis_level == "crisis":
        return {"type": "crisis", "level": "crisis"}

    return {"type": "safe", "level": crisis_level}

def get_severity(phq_score):
    if phq_score <= 4:
        return "Minimal"
    elif phq_score <= 9:
        return "Mild"
    elif phq_score <= 14:
        return "Moderate"
    return "Severe"

def select_quran_verse(user_text):
    text = normalize_for_safety(user_text)
    used = session.get("used_verses", [])

    if any(w in text for w in ["suicide", "kill myself", "want to die", "hopeless", "no point"]):
        preferred = "hope"
    elif any(w in text for w in ["burden", "too much", "exhausted", "tired"]):
        preferred = "burden"
    else:
        preferred = "hardship"

    candidates = [v for v in QURAN_HOPE_VERSES if v["topic"] == preferred and v["ref"] not in used]
    if not candidates:
        candidates = [v for v in QURAN_HOPE_VERSES if v["ref"] not in used]
    if not candidates:
        used = []
        session["used_verses"] = used
        candidates = QURAN_HOPE_VERSES[:]

    verse = random.choice(candidates)
    used.append(verse["ref"])
    session["used_verses"] = used
    return verse

def select_psychosis_verse(user_text):
    text = normalize_for_safety(user_text)
    used = session.get("used_psychosis_verses", [])

    if any(w in text for w in ["voices", "talking to me", "standing in my room", "watching me"]):
        preferred = "protection"
    elif any(w in text for w in ["scared", "fear", "frightening", "panic"]):
        preferred = "calm"
    else:
        preferred = "healing"

    candidates = [v for v in PSYCHOSIS_SUPPORT_VERSES if v["topic"] == preferred and v["ref"] not in used]
    if not candidates:
        candidates = [v for v in PSYCHOSIS_SUPPORT_VERSES if v["ref"] not in used]
    if not candidates:
        used = []
        session["used_psychosis_verses"] = used
        candidates = PSYCHOSIS_SUPPORT_VERSES[:]

    verse = random.choice(candidates)
    used.append(verse["ref"])
    session["used_psychosis_verses"] = used
    return verse


def get_religious_support_line(user_text=""):
    text = normalize_for_safety(user_text)

    if any(w in text for w in ["hopeless", "no hope", "nothing left", "end"]):
        return 'Allah says in the Quran: "Do not lose hope in the mercy of Allah." (Quran 39:53)'
    if any(w in text for w in ["burden", "too much", "cannot bear", "heavy"]):
        return 'Allah says in the Quran: "Allah does not burden a soul beyond what it can bear." (Quran 2:286)'
    if any(w in text for w in ["hard", "hardship", "pain", "suffering"]):
        return 'Allah says in the Quran: "Indeed, with hardship comes ease." (Quran 94:5-6)'

    return random.choice([
        'Allah says in the Quran: "Do not lose hope in the mercy of Allah." (Quran 39:53)',
        'Allah says in the Quran: "Indeed, with hardship comes ease." (Quran 94:5-6)',
        'Allah says in the Quran: "Allah does not burden a soul beyond what it can bear." (Quran 2:286)',
        'In Islam, even deep pain is seen by Allah, and a hurting heart is not forgotten.',
        'In our religion, hardship is not the end of a person’s story, and mercy is never closed.'
    ])


def build_crisis_response(user_text, name=""):
    session["crisis_step"] = "initial"
    faith_line = get_religious_support_line(user_text)
    display_name = f"{name}. " if name else ""

    return (
        f"{display_name}I'm really glad you reached out. "
        "You matter and your life is important. "
        "Are you alone, or is there someone near you who can sit with you? "
        "Please call Umang (0317-4288665) or 1122 right now — they can help you immediately."
        f"\n\n{faith_line}"
    )

def build_psychosis_support_response(user_text, name="", role="default"):
    """Calm, non-validating psychosis-safe response."""
    text = normalize_for_safety(user_text)
    display_name = f"{name}, " if name else ""

    idp_contexts = {
        "student": "With studies, routine, and safety feeling unsettled, your mind may be under a lot of strain right now.",
        "homemaker": "After carrying family needs and daily uncertainty, your mind may be under a lot of strain right now.",
        "daily_worker": "With work, safety, and daily survival feeling uncertain, your mind may be under a lot of strain right now.",
        "unemployed": "With uncertainty around work and family responsibilities, your mind may be under a lot of strain right now.",
        "parent": "When you are trying to protect family while carrying fear inside, your mind may be under a lot of strain right now.",
        "default": "With everything you have been going through, your mind may be under a lot of strain right now."
    }
    idp_line = idp_contexts.get(role, idp_contexts["default"])

    severe_markers = [
        "hurt", "harm", "kill", "end", "die", "suicide", "go to her", "go to him",
        "join her", "join him", "join them", "i should go", "i should join"
    ]
    is_severe = any(m in text for m in severe_markers)

    if any(w in text for w in ["watching", "following", "spying", "tracking", "trying to kill", "harm me"]):
        opening = f"{display_name}that sounds really frightening and unsettling. I am here with you."
        support = "Even if it feels very real, you do not have to handle this alone."
        help_line = "Please reach out to someone you trust now, and if you feel unsafe, call 1122 or go to a nearby hospital."
        question = "Is there someone near you who can stay with you for a while?"
    elif any(w in text for w in ["voice", "voices", "hearing", "hear", "talking to me"]):
        opening = f"{display_name}hearing something like that must feel really overwhelming. I am glad you told me."
        support = "Try to stay near a safe person or a familiar place right now."
        help_line = "It would be a good idea to speak with someone you trust and also a doctor or counsellor who can support you through this."
        question = "Are you alone right now, or is someone close by?"
    elif any(w in text for w in ["dead", "died", "passed", "calling me", "go to her", "join her", "go to him", "join him"]):
        opening = f"{display_name}that sounds deeply confusing and painful, especially when it feels connected to someone you have lost."
        if any(w in text for w in ["go to her", "join her", "go to him", "join him", "i should go", "i should join", "want to die", "kill myself", "end my life"]):
            support = "Please do not act on that feeling; stay where you are and keep yourself close to someone safe."
            help_line = "Tell a trusted person near you right now, and if the urge feels strong, call 1122 or Umang at 0317-4288665 immediately."
            question = "Can you sit with someone you trust right now?"
        else:
            support = "This can feel frightening, but you do not have to treat the dream as something you must follow."
            help_line = "Since you feel scared, stay close to someone you trust and talk to them about what you saw."
            question = "Can you tell your sister or someone nearby what the dream made you feel?"
    else:
        opening = f"{display_name}that sounds really distressing and confusing to experience. I am glad you shared it with me."
        support = "You are not alone in this moment; try to keep your body grounded by noticing where you are and taking one slow breath."
        help_line = "Please consider telling someone you trust and speaking with a doctor or counsellor soon."
        question = "What is happening around you right now?"

    if is_severe and "1122" not in help_line:
        help_line = "Please tell someone near you right now, and if you feel at risk of acting on it, call 1122 or Umang at 0317-4288665 immediately."

    return f"{opening}\n\n{idp_line}\n\n{support}\n\n{help_line}\n\n{question}"


def get_recent_bot_responses(limit=4):
    history = list(session.get("history", []))
    bot_messages = [item.get("content", "") for item in history if item.get("role") == "assistant"]
    return bot_messages[-limit:]


def pick_fresh_response(options, fallback=None):
    recent = " ".join(get_recent_bot_responses()).lower()
    cleaned_options = []
    for option in options:
        probe = option.lower()[:80]
        if probe and probe not in recent:
            cleaned_options.append(option)

    if cleaned_options:
        return random.choice(cleaned_options)
    if fallback:
        return fallback
    return random.choice(options)


def build_wellbeing_context(user_id):
    if not user_id:
        return ""

    latest_log = get_latest_wellbeing_log(user_id)
    if not latest_log:
        return ""

    parts = []

    sleep = latest_log["sleep_hours"]
    mood = latest_log["mood"]
    energy = latest_log["energy"]

    if sleep is not None:
        if sleep <= 4:
            parts.append(f"You have only been sleeping around {sleep:g} hours, so no wonder your body feels worn out.")
        elif sleep <= 6:
            parts.append(f"Your sleep has been around {sleep:g} hours, which is probably leaving you drained.")
        else:
            parts.append(f"Your sleep is around {sleep:g} hours lately.")

    mood_map = {
        "very_low": "Your mood has been feeling very low.",
        "low": "Your mood has been feeling heavy.",
        "anxious": "There has been a lot of worry sitting in your system.",
        "okay": "Your mood seems a little steadier right now.",
        "good": "There are still some signs of steadiness in you."
    }
    if mood in mood_map:
        parts.append(mood_map[mood])

    energy_map = {
        "very_low": "Your energy sounds completely drained.",
        "low": "Your energy has been low.",
        "medium": "Your energy is there in small amounts.",
        "high": "Your energy seems stronger than before."
    }
    if energy in energy_map:
        parts.append(energy_map[energy])

    return " ".join(parts[:2])


def maybe_add_soft_faith_line(user_message, crisis_level="safe"):
    text = normalize_for_safety(user_message)

    if crisis_level in ["crisis", "emergency"]:
        return ""

    if any(x in text for x in ["sleep", "tired", "heavy", "hard", "burden", "worried", "hopeless"]):
        return random.choice([
            "Even in hard seasons, ease does not disappear forever.",
            "Sometimes faith helps by reminding us that hard phases do move.",
            "A person can feel worn down and still not be at the end of their story."
        ])

    return ""


def build_local_fallback_response(user_message, sess, emotion="unknown"):
    name = sess.get("name", "")
    role = sess.get("role", "default")
    history = list(sess.get("history", []))

    hardships = detect_idp_hardships(history)
    session["idp_hardships"] = hardships

    context_line = get_idp_context_line(role, hardships)
    idp_line = context_line
    faith_line = maybe_add_soft_faith_line(user_message, sess.get("crisis_level", "safe"))
    opener = f"{name}, " if name else ""
    lowered = user_message.lower()

    sleep_words = ["sleep", "slept", "insomnia", "night", "awake"]
    tired_words = ["tired", "drained", "exhausted", "fatigue", "weak", "energy"]

    # Strict rule: only discuss sleep/rest when user brings sleep or tiredness first.
    if any(w in lowered for w in sleep_words + tired_words):
        options = [
            f"{opener}that sounds exhausting. {idp_line} What usually makes nights hardest for you?",
            f"{opener}when rest goes off like this, the whole day can feel heavier. {idp_line} How long has this been going on?",
            f"{opener}that kind of tiredness can build up quickly. Is it more broken sleep, worry, noise around you, or not getting the chance to rest at all?",
            f"{opener}your body sounds worn out. What happens first — racing thoughts, interruptions around you, or no real chance to rest?"
        ]
        return pick_fresh_response(options)

    # Mood / sadness: stay on mood, but keep a light Pakistan-IDP frame.
    if emotion in ["depression", "sadness", "grief"]:
        options = [
            f"{opener}that sounds really heavy. {idp_line} What has been weighing on you the most today?",
            f"{opener}I can tell this is not a small moment for you. Living with uncertainty after displacement can make feelings harder to name. What feels hardest right now?",
            f"{opener}you do not sound okay. {context_line} Is it more your thoughts, your heart, or everything together today?",
            f"{opener}{idp_line} When life feels unsettled, even small things can feel too much. What has been draining you the most lately?"
        ]
        return pick_fresh_response(options)

    if emotion in ["anxiety", "fear", "stress", "ptsd"]:
        options = [
            f"{opener}your mind sounds really overloaded right now. {idp_line} What keeps circling in your head the most?",
            f"{opener}that kind of pressure can make it hard to settle. If things around you are uncertain, what has been making it hardest to feel calm?",
            f"{opener}it feels like your system has been on edge for too long. {context_line} What part of the day feels most intense for you?",
            f"{opener}{idp_line} What has been putting the most pressure on you lately?"
        ]
        return pick_fresh_response(options)

    options = [
        f"{opener}I am with you. {idp_line} What feels most difficult today?",
        f"{opener}you can say it plainly here. When life is unsettled after displacement, feelings can become hard to explain. What has been sitting on your mind the most?",
        f"{opener}this seems bigger than one small bad moment. {faith_line} {idp_line} What has been wearing you down lately?",
        f"{opener}{context_line} What feels hardest to carry right now?"
    ]
    return pick_fresh_response(options)

def get_response(user_message, sess, crisis_level="safe", emotion="unknown"):
    name = sess.get("name", "")
    role = sess.get("role", "default")
    severity = sess.get("severity", "unknown")
    history = list(sess.get("history", []))
    user_id = sess.get("user_id")
    memory_profile = sess.get("user_memory_profile", "")

    cause = detect_displacement_cause(history)
    phrase = get_displacement_phrase(cause)
    session["displacement_cause"] = cause

    hardships = detect_idp_hardships(history)
    session["idp_hardships"] = hardships
    idp_context_line = get_idp_context_line(role, hardships)

    retrieved = retrieve_from_training_data(user_message, emotion, role, top_k=2)

    retrieval_context = ""
    if retrieved:
        retrieval_context = "\n\nMOST RELEVANT RESPONSES FROM OUR IDP TRAINING DATA:\n"
        for i, r in enumerate(retrieved, 1):
            retrieval_context += (
                f"\nExample {i} [source={r['source']}, similarity={r['similarity']}]:\n"
                f"Situation: {r['question'][:150]}\n"
                f"Counselor response: {r['answer'][:400]}\n"
            )
        retrieval_context += "\nIMPORTANT: Use these as style and support references. Rephrase naturally. Do not copy word for word."

    insight = IDP_SURVEY_INSIGHTS.get(role) or {}
    survey_note = ""
    if insight:
        survey_note = (
            f"\nOUR REAL IDP SURVEY DATA for '{role}' (n={insight['count']} actual Pakistani IDPs):\n"
            f"Most common depression issues: {', '.join(insight['top_phq'])}\n"
            f"Most common anxiety issues: {', '.join(insight['top_gad'])}\n"
        )

    wellbeing_note = ""
    if user_id:
        latest_log = get_latest_wellbeing_log(user_id)
        if latest_log:
            parts = []
            if latest_log["sleep_hours"] is not None:
                parts.append(f"last recorded sleep: {latest_log['sleep_hours']} hours")
            if latest_log["mood"]:
                parts.append(f"last recorded mood: {latest_log['mood']}")
            if latest_log["energy"]:
                parts.append(f"last recorded energy: {latest_log['energy']}")
            if parts:
                wellbeing_note = "\nPREVIOUS WELLBEING INFO FOR THIS RETURNING USER: " + ", ".join(parts)

    hardship_note = ""
    if hardships:
        hardship_note = f"\nDETECTED IDP HARDSHIPS FROM CONVERSATION: {', '.join(hardships)}\nSuggested emotional framing: {idp_context_line}\n"

    advice_note = ""
    last_advice = sess.get("last_advice", "")
    last_scores = sess.get("last_assessment_scores", {})
    if last_advice:
        advice_note = (
            f"\nRECENT ASSESSMENT RESULTS FOR THIS USER:\n"
            f"PHQ-9 (Depression): {score_to_10(last_scores.get('phq9', 0), 27)}/10 | "
            f"GAD-7 (Anxiety): {score_to_10(last_scores.get('gad7', 0), 21)}/10 | "
            f"WHO-5 (Wellbeing): {score_to_10(last_scores.get('who5', 0), 25)}/10\n"
            f"Advice already given:\n{last_advice}\n"
            f"INSTRUCTION: If the user mentions sleep, routines, anxiety coping, helplines, or professional support, "
            f"gently follow up on how they are doing with the guidance above. Do not repeat the full advice — "
            f"just check in naturally and adjust based on what they share.\n"
        )

    system_prompt = f"""
You are Empath AI — a supportive mental health chatbot designed specifically for internally displaced people (IDPs) in Pakistan.

PERSON:
Name={name or 'unknown'}
Role={role}
Emotion={emotion}
Crisis={crisis_level}
Severity={severity}

LEARNED USER MEMORY PROFILE (compact profile learned from previous chats, not full chat):
{memory_profile}

MEMORY RULES:
- Use this profile gently when relevant.
- Do not quote or repeat old conversations.
- Do not say "last time you said" unless the user asks.
- Adapt support using learned role, displacement, living place, stressors, and wellbeing patterns.

DISPLACEMENT CONTEXT:
Displacement cause detected silently: {cause}
Natural phrase if needed: "{phrase}"
IMPORTANT: This chatbot is built for Pakistani IDPs. Use a light IDP-aware frame when the user is distressed, especially around camp, shelter, NGO, relatives, temporary living, lost routine, privacy, studies, family pressure, or uncertainty.
Do not overuse displacement wording in every sentence. If the user has shared a specific living place, use that exact wording and do not replace it with another place.
If the cause is unknown, do not assume flood, conflict, or any specific reason.
{hardship_note}
{survey_note}
{wellbeing_note}
{advice_note}
{retrieval_context}

YOUR STYLE:
1. Speak like a real person, not like a counselor reading a script.
2. Use natural English with short, direct sentences.
3. Sound warm and grounded, not formal.
4. For general sadness or confusion, include a gentle Pakistan-IDP context when natural, such as disrupted routine, temporary place, camp, shelter, NGO, relatives, privacy, studies, family pressure, or uncertainty.
5. Tailor the response to the person's role and recent wellbeing data if available.
6. Keep the reply to 3 or 4 short paragraphs or 4 to 6 sentences total.
7. Ask no more than one gentle question.
8. Avoid filler phrases like "I'm sorry to hear that", "I understand", "that sounds difficult", or "would you like to share more".
9. Avoid repetitive crisis warnings or repeating the same structure as the last reply.
10. Do not sound clinical, preachy, or overly polished.
11. Support first, assessment later.
12. Never provide self-harm information.
13. If crisis is present, mention Umang: 0317-4288665.
14. Strict rule: do not introduce sleep, sleeping hours, night rest, or energy unless the user explicitly mentions sleep, tiredness, rest, fatigue, or energy first.
15. If this is a returning user, you may gently notice patterns over time, but do not sound technical.
16. If faith fits naturally, add only one soft line. Keep it brief and natural.

BAD STYLE:
- robotic empathy
- formal therapy wording
- repeated sentence patterns
- too many questions
- long lectures
"""

    try:
        messages = []
        for item in history[-6:]:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append(item)

        messages.append({"role": "user", "content": user_message})

        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_prompt,
            input=messages,
            max_output_tokens=220,
        )

        return response.output_text.strip()

    except Exception as e:
        print(f"OpenAI error: {e}")
        if retrieved:
            return retrieved[0]["answer"][:400]
        return build_local_fallback_response(user_message, sess, emotion)

def get_crisis_followup_response(name, user_message):
    text = normalize_for_safety(user_message)
    display_name = f"{name}, " if name else ""
    previous_step = session.get("crisis_step", "initial")
    faith_line = get_religious_support_line(user_message)

    if any(k in text for k in ["i am alone", "alone", "nobody", "no one"]):
        session["crisis_step"] = "alone"
        session["crisis_active"] = True
        return (
            f"{display_name}I hear you. Being alone can make a painful night feel even heavier. "
            "Stay with me for this moment. Put both feet on the ground if you can, and take one slow breath in and one slow breath out. "
            f"{faith_line} "
            "Tell me softly — what feels heaviest right now?"
        )

    if any(k in text for k in ["dont want anyone", "don't want anyone", "dont want to tell", "don't want to tell", "i dont want anyone", "i don't want anyone"]):
        session["crisis_step"] = "refuses_support"
        session["crisis_active"] = True
        return (
            f"{display_name}I understand. Sometimes pain feels so personal that even another person feels like too much. "
            "I will stay with you here, and we will keep this moment small. "
            f"{faith_line} "
            "Tell me — is the pain sitting more in your thoughts, your chest, your sleep, or something that happened today?"
        )

    if any(k in text for k in ["i dont know what to do", "i don't know what to do", "dont know what to do", "don't know what to do"]):
        session["crisis_step"] = "not_sure_what_to_do"
        session["crisis_active"] = True
        return (
            f"{display_name}You do not have to figure everything out right now. "
            "For this moment, just stay here with me. Take one slow breath in and one slow breath out. "
            f"{faith_line} "
            "Tell me — is the pain feeling heavier in your thoughts, your chest, your sleep, or because of something that happened today?"
        )

    if any(k in text for k in ["i dont know", "i don't know", "dont know", "don't know"]):
        session["crisis_step"] = "not_sure"
        session["crisis_active"] = True
        return (
            f"{display_name}That is okay. You do not have to understand all of it right now. "
            "Sometimes pain is just heavy before it becomes clear. "
            f"{faith_line} "
            "For this moment, just stay with me and tell me — is your mind racing, is your chest tight, or do you just feel empty?"
        )

    if any(k in text for k in ["yes my family is there", "my family is there", "family is there", "family is with me", "yes my family is with me", "yes family is with me", "yes someone is with me"]):
        session["crisis_step"] = "with_family"
        session["crisis_active"] = False
        return (
            f"{display_name}I’m relieved to hear your family is there with you. "
            "Please stay close to them tonight, even if you do not want to explain everything. "
            f"{faith_line} "
            "You do not have to solve everything right now. Tell me — what has been hurting you the most today?"
        )

    if any(k in text for k in ["yes i am safe", "i am safe", "yes safe", "safe now", "yes i am with my family"]):
        session["crisis_step"] = "safe_confirmed"
        session["crisis_active"] = False
        return (
            f"{display_name}I’m relieved to hear that you are safe right now. "
            "Thank you for staying here with me. Let us keep this moment gentle and simple. "
            f"{faith_line} "
            "Tell me — what has been weighing on your heart the most tonight?"
        )

    if text in {"yes", "yess", "yeah", "haan", "han"}:
        if previous_step in {"initial", "alone", "refuses_support", "not_sure", "not_sure_what_to_do"}:
            session["crisis_step"] = "yes_but_unspecified"
            session["crisis_active"] = True
            return (
                f"{display_name}Thank you for staying with me. "
                "Just help me understand one small thing — is a family member with you right now, or are you by yourself?"
            )
        if previous_step in {"with_family", "safe_confirmed"}:
            session["crisis_active"] = False
            return (
                f"{display_name}I’m glad you stayed with me. "
                "Since you are safe right now, let us focus on what is hurting underneath all this. "
                "You can say it in one line only if that feels easier."
            )

    if any(k in text for k in ["i want to die", "kill myself", "suicide", "end my life", "don't want to live", "dont want to live"]):
        session["crisis_active"] = True
        return build_crisis_response(user_message, name)

    if previous_step in {"with_family", "safe_confirmed"}:
        session["crisis_active"] = False
        return (
            f"{display_name}I’m still here with you, and I’m glad you stayed. "
            "Since you are safe right now, let us focus on what is hurting underneath all this. "
            "You can say it in one line only if that feels easier."
        )

    session["crisis_active"] = True
    return (
        f"{display_name}I’m here with you. Let us slow this down together for one moment. "
        "Take one small breath in and one slow breath out. "
        f"{faith_line} "
        "Tell me — what feels heaviest right now?"
    )

def save_assessment_json(session_data):
    try:
        record = {
            "resourceType": "MentalHealthAssessment",
            "patient_name": session_data.get("name", "Unknown"),
            "role": session_data.get("role", "Unknown"),
            "displacement_cause": session_data.get("displacement_cause", "Unknown"),
            "idp_hardships": session_data.get("idp_hardships", []),
            "assessment_date": datetime.now().isoformat(),
            "phq9_scores": dict(session_data.get("phq", {})),
            "gad7_scores": dict(session_data.get("gad", {})),
            "who5_scores": dict(session_data.get("who", {})),
            "phq9_total": sum(session_data.get("phq", {}).values()),
            "gad7_total": sum(session_data.get("gad", {}).values()),
            "who5_total": sum(session_data.get("who", {}).values()),
            "severity": session_data.get("severity", "Unknown"),
            "emotion_detected": session_data.get("emotion", "Unknown"),
            "flagged_for_followup": session_data.get("severity") in ["Moderate", "Severe"],
        }

        os.makedirs(os.path.join(BASE_PATH, "assessments"), exist_ok=True)
        fname = os.path.join(
            BASE_PATH,
            "assessments",
            f"{session_data.get('name', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    except Exception as e:
        print("Save JSON error:", e)

def ask_first_assessment_question(prev=None):
    session["stage"] = STAGE_ASSESS
    session["q_index"] = 0
    session["assessment_started"] = True
    session["phq"] = {}
    session["gad"] = {}
    session["who"] = {}

    if prev:
        intro = (
            "It has been a week since your last check-in. Let us see how you have been doing. "
            "Same questions — just answer as honestly as you can.\n\n"
        )
    else:
        intro = (
            "I want to understand how you have been feeling. I will ask you some questions — "
            "there are no right or wrong answers.\n\n"
        )
    return intro + SA_QUESTIONS[0]["text"] + "\n(0=Not at all  1=Several days  2=More than half the days  3=Nearly every day)"


SA_FOLLOWUP_QUESTIONS = {
    "crisis_risk": [
        "I saw from your recent self-assessment that you were going through something very serious. How have you been feeling since then? Are you safe right now?",
        "I wanted to check in — have you been able to reach out to anyone, or call Umang (0317-4288665) like I suggested?",
        "How is today feeling compared to when you did the assessment? I am here if something still feels very heavy.",
    ],
    "depression_high": [
        "I noticed from your recent self-assessment that low mood and energy have been affecting you. How have things been feeling since then?",
        "From your check-in, keeping a small daily routine was one of the suggestions. Have you been able to try anything small — even a short walk or a fixed prayer time?",
        "How is your mood this week compared to when you did the self-assessment? Has anything shifted, even a little?",
    ],
    "anxiety_high": [
        "Your recent self-assessment showed that worry has been quite high for you. What has been sitting most heavily on your mind lately?",
        "One of the suggestions from your check-in was the breathing technique — breathe in 4, hold 4, out 6. Have you had a chance to try it when worry builds up?",
        "How has the anxiety been this week compared to when you did the assessment? Is it feeling any different?",
    ],
    "wellbeing_low": [
        "Your self-assessment showed that your daily wellbeing has been very low. Have there been any small moments this week where things felt even slightly okay?",
        "From your check-in, one thing I suggested was identifying one small comfort — a familiar routine, a short rest, or a conversation with someone safe. Has anything like that helped?",
        "How are you feeling in yourself today? Any small change since you did the self-assessment?",
    ],
    "mild": [
        "You completed a self-assessment recently. How have things been feeling since then?",
        "From your check-in, things were manageable but still difficult. Has anything changed or felt heavier this week?",
        "How are you doing today compared to when you filled out the check-in?",
    ],
    "minimal": [
        "You completed a self-assessment recently. How have you been getting on since then?",
        "Your scores were in a reasonable range, but I know that does not always capture everything. Is there anything that has been on your mind this week?",
        "How are you feeling today? Anything you would like to talk through?",
    ],
}


def load_recent_self_assessment(user_id):
    """Load the most recent self-assessment from DB into main chat session for follow-up."""
    last = get_last_assessment_for_user(user_id)
    if not last or last["phq9_total"] is None:
        return
    if days_since_assessment(last) >= 7:
        return

    phq_total = last["phq9_total"]
    gad_total = last["gad7_total"]
    who_total = last["who5_total"] or 0

    session["last_assessment_scores"] = {"phq9": phq_total, "gad7": gad_total, "who5": who_total}
    session["last_advice"] = last["advice_given"] or ""
    session["sa_followup_count"] = 0
    session["sa_followup_needed"] = True
    session["sa_listen_turns"] = 0

    # Determine dominant concern for targeted follow-up questions
    if phq_total >= 20 or gad_total >= 18:
        session["sa_dominant_concern"] = "crisis_risk"
    elif phq_total >= 10:
        session["sa_dominant_concern"] = "depression_high"
    elif gad_total >= 10:
        session["sa_dominant_concern"] = "anxiety_high"
    elif who_total * 4 < 40:
        session["sa_dominant_concern"] = "wellbeing_low"
    elif phq_total >= 5 or gad_total >= 5:
        session["sa_dominant_concern"] = "mild"
    else:
        session["sa_dominant_concern"] = "minimal"
    session["crisis_step"] = ""


def _check_and_start_assessment():
    """Returns response text. Enforces 7-day gate; loads prev assessment for comparison intro."""
    user_id = session.get("user_id")
    if user_id:
        last = get_last_assessment_for_user(user_id)
        days = days_since_assessment(last)
        if days < 7:
            next_date = (
                datetime.fromisoformat(str(last["assessment_date"])) + timedelta(days=7)
            ).strftime("%B %d")
            saved_advice = (last["advice_given"] or "").strip()
            result_text = (
                f"You have already completed your assessment this week. "
                f"Your next one will be available on {next_date}.\n\n"
                f"Saved results:\n"
                f"PHQ-9 (Depression): {score_to_10(last['phq9_total'], 27)}/10\n"
                f"GAD-7 (Anxiety): {score_to_10(last['gad7_total'], 21)}/10\n"
                f"WHO-5 (Wellbeing): {score_to_10(last['who5_total'], 25)}/10"
            )
            if saved_advice:
                result_text += f"\n\nSaved comments and guidance:\n{saved_advice}"
            result_text += "\n\nIf something feels urgent right now, I am here to talk."
            return result_text
        prev = last if (last and last["phq9_total"] is not None) else None
        return ask_first_assessment_question(prev=prev)
    return ask_first_assessment_question()

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("account_id"):
            return redirect(url_for("auth_page"))
        return f(*args, **kwargs)
    return decorated

@app.route("/")
def index():
    return render_template("intro.html")

@app.route("/login")
def auth_page():
    if session.get("account_id"):
        return redirect(url_for("dashboard"))
    return render_template("auth.html")

@app.route("/signup", methods=["POST"])
def signup():
    init_db()
    data = request.get_json()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    dob = data.get("dob") or ""

    if not name or not email or not password or not dob:
        return jsonify({"error": "All fields are required."}), 400

    # Age check server-side
    try:
        birth = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        if age < 18:
            return jsonify({"error": f"You must be 18 or older. You are {age} years old."}), 400
    except ValueError:
        return jsonify({"error": "Invalid date of birth."}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM accounts WHERE email = ?", (email,))
    if cursor.fetchone():
        return jsonify({"error": "An account with this email already exists. Please login."}), 400

    pw_hash = hash_password(password)
    cursor.execute(
        "INSERT INTO accounts (full_name, email, password_hash, dob) VALUES (?,?,?,?)",
        (name, email, pw_hash, dob)
    )
    db.commit()
    account_id = cursor.lastrowid

    session.permanent = True
    session["account_id"] = account_id
    session["account_name"] = name
    return jsonify({"message": "Account created.", "name": name}), 200

@app.route("/login", methods=["POST"])
def login():
    init_db()
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    db = get_db()
    cursor = db.cursor()
    pw_hash = hash_password(password)
    cursor.execute(
        "SELECT id, full_name FROM accounts WHERE email = ? AND password_hash = ?",
        (email, pw_hash)
    )
    row = cursor.fetchone()
    if not row:
        return jsonify({"error": "No account found with these credentials. Please sign up first."}), 401

    session.permanent = True
    session["account_id"] = row["id"]
    session["account_name"] = row["full_name"]
    return jsonify({"message": "Logged in.", "name": row["full_name"]}), 200

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))
@app.route("/delete_account", methods=["POST"])
def delete_account():
    print("DELETE ACCOUNT ROUTE HIT")

    account_id = session.get("account_id")
    if not account_id:
        return jsonify({
            "success": False,
            "message": "No logged-in account found."
        }), 401

    db = get_db()
    user = get_user_by_account_id(account_id)

    try:
        if user:
            db.execute("DELETE FROM wellbeing_logs WHERE user_id = ?", (user["id"],))
            db.execute("DELETE FROM chat_history WHERE user_id = ?", (user["id"],))
            db.execute("DELETE FROM assessments WHERE user_id = ?", (user["id"],))
            db.execute("DELETE FROM users WHERE id = ?", (user["id"],))

        db.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
        db.commit()
        session.clear()

        print("ACCOUNT DELETED SUCCESSFULLY:", account_id)

        return jsonify({
            "success": True,
            "message": "Your account has been deleted successfully.",
            "redirect": url_for("index")
        }), 200

    except Exception as e:
        db.rollback()
        print("DELETE ACCOUNT ERROR:", str(e))
        return jsonify({
            "success": False,
            "message": f"Could not delete account: {str(e)}"
        }), 500




def get_crisis_support_message(name, memory=None):
    opener = f"{name}, " if name else ""

    # --- variations ---
    openings = [
        "I want to slow this moment down with you for a second.",
        "Let’s pause here together for a moment.",
        "Take a breath with me for a second.",
        "I’m here with you right now — let’s go gently."
    ]

    hope_lines = [
        "This shall pass too, even if it doesn’t feel like it right now.",
        "What you’re feeling is heavy, but it is not permanent.",
        "Even the longest nights eventually turn into morning.",
        "This moment is not your whole life."
    ]

    strength_lines = [
        "You’ve already survived difficult moments before.",
        "There is strength in you, even if it feels hidden right now.",
        "The fact that you are here shows your strength.",
        "You are doing better than you think."
    ]

    islamic_lines = [
        "Allah reminds us: 'With hardship comes ease.'",
        "Allah does not burden a soul beyond what it can bear.",
        "Even in hardship, there is a path toward ease.",
        "Allah sees your struggle, even the quiet ones."
    ]

    support_lines = [
        "You don’t have to carry everything alone.",
        "It’s okay to take things one small step at a time.",
        "Right now, just one calm breath is enough.",
        "You don’t need to fix everything today."
    ]

    closing_lines = [
        "We can talk more… tell me what’s on your mind.",
        "I’m here to listen… what feels heaviest right now?",
        "You can share anything with me… I’m here.",
        "Tell me what has been weighing on you the most."
    ]

    # --- optional personalization (IDP memory) ---
    displacement_line = ""
    if memory and memory.get("living_place"):
        place = memory.get("living_place")
        displacement_line = f"Being in {place} can make everything feel even heavier sometimes.\n\n"

    # --- build message ---
    message = (
        f"{opener}{random.choice(openings)}\n\n"
        f"{random.choice(hope_lines)}\n\n"
        f"{random.choice(strength_lines)}\n\n"
        f"{support_lines[0]}\n\n"
        f"{displacement_line}"
        f"{random.choice(islamic_lines)}\n\n"
        f"If you feel unsafe, please call Umang 0317-4288665 or 1122.\n\n"
        f"{random.choice(support_lines)}\n\n"
        f"{random.choice(closing_lines)}"
    )

    return message

@app.route("/app")
@login_required
def chat_app():
    session_name = session.get("account_name", "")
    crisis_transition = request.args.get("crisis") == "1"
    # Keep account info but clear chat session state
    account_id = session.get("account_id")
    account_name = session.get("account_name")
    session.clear()
    session.permanent = True
    session["account_id"] = account_id
    session["account_name"] = account_name
    if crisis_transition:
        session["crisis_transition_pending"] = True
    return render_template("index.html", user_name=session_name)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user_name=session.get("account_name", ""))
# ================== PRO RESPONSE ENGINE ==================

def simple_memory_update(session, msg):
    msg = msg.lower()

    memory = session.get("simple_memory", {})

    if "money" in msg:
        memory["stress"] = "financial"
    elif any(w in msg for w in ["study","exam","focus"]):
        memory["stress"] = "academic"
    elif detect_family_stress(msg):
        memory["stress"] = "family"

    if "tired" in msg:
        memory["energy"] = "low"

    if any(w in msg for w in ["sad","low","down"]):
        memory["mood"] = "low"

    session["simple_memory"] = memory
    return memory




def smart_response(user_message, session):
    msg = user_message.lower()

    # Neutral/supportive family presence should not be treated as family stress
    if is_neutral_family_presence(msg):
        return "That is good to hear. Having your family with you can be a source of support.\n\nHow are you feeling personally right now?"

    if "happy" in msg:
        return random.choice([
            "That’s really nice to hear. What made today better?",
            "I’m glad things feel lighter today. What helped?"
        ])

    if "money" in msg:
        return random.choice([
            "Money stress can feel really heavy. What part is worrying you most?",
            "Financial pressure can stay on your mind. What feels hardest right now?"
        ])

    if "study" in msg:
        return random.choice([
            "Trying to study like this is really hard. What part feels most difficult?",
            "It’s not easy to focus when your mind is carrying so much. What’s hardest right now?"
        ])

    if "tired" in msg:
        return random.choice([
            "That sounds exhausting. Is it more physical or mental tiredness?",
            "Feeling that tired can drain everything. What’s taking most of your energy?"
        ])

    if any(w in msg for w in ["sad","low","down"]):
        place = session.get("living_place", "")
        if place:
            return f"Some days feel heavier, especially living in {place}. What kind of low is it today?"
        return "Some days just feel low. What kind of low is it today?"

    return None
@app.route("/chat", methods=["GET"])
@login_required
def chat_page_redirect():
    if request.args.get("crisis") == "1":
        return redirect(url_for("chat_app", crisis=1))
    return redirect(url_for("chat_app"))

@app.route("/init", methods=["GET"])
def init():
    account_id = session.get("account_id")
    account_name = (session.get("account_name") or "").strip()

    init_db()

    if not account_id or not account_name:
        init_session()
        welcome = "Hello. I am here to listen and support you. What should I call you?"
        add_to_history("assistant", welcome)
        return jsonify({
            "response": welcome,
            "stage": session.get("stage"),
            "name": session.get("name", ""),
            "is_returning_user": False,
            "type": "normal"
        })

    session.permanent = True
    session["account_id"] = account_id
    session["account_name"] = account_name
    init_session()

    first_name = account_name.split()[0].capitalize()
    session["name"] = first_name

    linked_user = get_user_by_account_id(account_id)

    if not linked_user:
        linked_user = find_existing_user(first_name)
        if linked_user and users_has_account_id() and not linked_user["account_id"]:
            try:
                db = get_db()
                db.execute("UPDATE users SET account_id = ? WHERE id = ?", (account_id, linked_user["id"]))
                db.commit()
                linked_user = get_user_by_id(linked_user["id"])
            except Exception:
                pass

    if linked_user:
        session["user_id"] = linked_user["id"]
        session["role"] = linked_user["role"] if linked_user["role"] else "default"
        session["living_place"] = get_row_value(linked_user, "living_place", "")
        session["user_memory_profile"] = get_user_memory_profile(linked_user["id"])
        update_user_last_seen(linked_user["id"])
        load_recent_self_assessment(linked_user["id"])

        real_history = has_real_user_history(linked_user)

        if real_history:
            session["is_returning_user"] = True
            session["stage"] = STAGE_LISTEN
            if session.get("user_memory_profile"):
                welcome = (
                    f"Welcome back, {first_name}. I remember the support context you have shared before, "
                    f"so we can continue gently from there. How are things feeling today?"
                )
            else:
                welcome = (
                    f"Welcome back, {first_name}. "
                    f"How have you been feeling lately?"
                )
        else:
            session["is_returning_user"] = False
            session["stage"] = STAGE_ROLE
            welcome = (
                f"Hello, {first_name}. It's nice to meet you. "
                f"Can you tell me a little about yourself — are you a student, working, "
                f"a housewife/homemaker, unemployed, or something else?"
            )
    else:
        new_user_id = create_user(first_name, account_id=account_id)
        session["user_id"] = new_user_id
        session["is_returning_user"] = False
        session["role"] = "default"
        load_recent_self_assessment(new_user_id)

        session["stage"] = STAGE_ROLE
        welcome = (
            f"Hello, {first_name}. It's nice to meet you. "
            f"Can you tell me a little about yourself — are you a student, working, "
            f"a housewife/homemaker, unemployed, or something else?"
        )

    if session.get("crisis_transition_pending"):
        session["crisis_transition_pending"] = False
        session["stage"] = STAGE_LISTEN
        session["is_returning_user"] = True
        welcome = get_crisis_support_message(session.get("name", first_name))

    add_to_history("assistant", welcome)

    return jsonify({
        "response": welcome,
        "stage": session.get("stage"),
        "name": session.get("name", first_name),
        "is_returning_user": session.get("is_returning_user", False),
        "type": "normal"
    })




# ─────────────────────────────────────────────
# NATURAL RESPONSE ENGINE
# ─────────────────────────────────────────────
def _recent_dialogue_for_prompt(limit=8):
    history = list(session.get("history", []))[-limit:]
    lines = []
    for item in history:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _compact_user_context(sess, emotion="unknown", wellbeing_info=None):
    name = sess.get("name", "") or "the user"
    role = sess.get("role", "default") or "default"
    memory = sess.get("user_memory_profile", "") or ""
    living_place = sess.get("living_place", "") or ""

    context = [
        f"Name: {name}",
        f"Role/profession: {role}",
        "Project context: Empath AI supports internally displaced people in Pakistan; keep this as gentle background context.",
        "Do not ask whether the user is an IDP and do not repeatedly say 'as an IDP' or 'because you are displaced'.",
    ]
    if living_place:
        context.append(f"Current place/living context: {living_place}")
    if memory:
        context.append(f"Saved memory summary: {memory}")
    if emotion and emotion != "unknown":
        context.append(f"Detected emotional signal: {emotion}")
    if wellbeing_info:
        context.append(f"Wellbeing info extracted from latest message: {wellbeing_info}")
    return "\n".join(context)


def _remove_extra_questions(text):
    """Keep max one natural follow-up question."""
    text = (text or "").strip()
    if text.count("?") <= 1:
        return text
    chars = []
    seen_question = False
    for ch in text:
        if ch == "?":
            if not seen_question:
                chars.append(ch)
                seen_question = True
            else:
                chars.append(".")
        else:
            chars.append(ch)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _avoid_repeated_opening(text):
    """Reduce repeated phrases that made the bot feel template-based."""
    repeated = [
        "I am really glad to hear that.",
        "Your positive thinking is meaningful",
        "I hope this happiness stays with you",
        "That is good to hear.",
    ]
    alternatives = [
        "That sounds like a genuinely lighter moment.",
        "There is something very gentle in what you just shared.",
        "That feels like a hopeful shift.",
        "I can hear a little ease in your words.",
    ]
    for phrase in repeated:
        if phrase.lower() in text.lower():
            text = re.sub(re.escape(phrase), random.choice(alternatives), text, flags=re.IGNORECASE)
    return text

def _hopeful_idp_line(user_message="", sess=None):
    """Return one soft IDP-sensitive line that validates real-life context without reducing hope."""
    t = normalize_for_safety(user_message or "")

    if any(w in t for w in ["happy", "good news", "thankful", "grateful", "better", "hope", "peace", "peaceful", "handled", "managed"]):
        options = [
            "Even with so much change around you, this brighter moment is worth holding onto.",
            "After everything you have been managing, noticing a good moment shows real strength.",
            "When life has felt unsettled, moments like this can become small signs of hope.",
            "In difficult circumstances, this kind of light feeling can quietly give you strength."
        ]
    elif any(w in t for w in ["sleep", "tired", "exhausted", "drained", "energy", "weak"]):
        options = [
            "When life has been unsettled, rest can become harder, but your body still deserves gentleness.",
            "After everything you have been carrying, even small rest and care can help you regain strength.",
            "With so much change around you, low energy makes sense, and it can improve step by step.",
            "This tiredness is not weakness; it may be your body asking for care after a difficult stretch."
        ]
    elif any(w in t for w in ["job", "work", "money", "income", "unemployed", "earn", "fees", "expense"]):
        options = [
            "When stability has been shaken, work and money worries can feel heavier, but small practical steps still matter.",
            "After everything you have been managing, this pressure is real, yet it does not define your worth.",
            "In uncertain circumstances, even one small step toward support or work can bring a little hope back.",
            "With so much responsibility around you, it makes sense to feel pressure, and you still deserve support too."
        ]
    elif any(w in t for w in ["family", "children", "parents", "responsibility", "pressure"]):
        options = [
            "After everything your family has been managing, even small acts of care can hold a lot of meaning.",
            "When life around the family feels unsettled, love and responsibility can both feel heavy, but they also show your strength.",
            "With so much change around you, caring for family is not easy, and your effort still matters.",
            "In difficult circumstances, trying to stay present for family is already a meaningful form of support."
        ]
    elif any(w in t for w in ["sad", "low", "worse", "worst", "hopeless", "empty", "cry", "stress", "worried", "anxious", "fear", "panic"]):
        options = [
            "When life has been unsettled for a while, heavy feelings can grow louder, but they can still be worked through slowly.",
            "After everything you have been carrying, this feeling makes sense, and you do not have to face it alone.",
            "With so much change around you, even small worries can feel bigger, but small steady steps can still help.",
            "In difficult circumstances, feeling overwhelmed does not mean you are failing; it means you need support and kindness."
        ]
    else:
        options = [
            "Even with so much change around you, your feelings still deserve space and care.",
            "After everything you have been managing, taking this one moment gently can still help.",
            "When life feels unsettled, being heard clearly can become one small source of steadiness.",
            "In difficult circumstances, small honest conversations can bring a little more clarity."
        ]
    return random.choice(options)


def _has_idp_sensitive_line(text):
    """Avoid adding duplicate IDP-context lines if the response already has one."""
    t = (text or "").lower()
    markers = [
        "after everything", "with so much change", "when life has", "when life feels",
        "life has been", "unsettled", "difficult circumstances", "stability has been shaken",
        "everything you have been managing", "everything you’ve been managing"
    ]
    return any(m in t for m in markers)


def _add_hopeful_idp_line(text, user_message="", sess=None):
    """Insert exactly one hopeful IDP-sensitive line before the final follow-up question."""
    text = (text or "").strip()
    if not text or _has_idp_sensitive_line(text):
        return text

    line = _hopeful_idp_line(user_message, sess)

    # Put the IDP-sensitive line before the final question, so the response still ends with one question.
    qpos = text.rfind("?")
    if qpos != -1:
        before = text[:qpos + 1]
        split_at = max(before.rfind("\n", 0, qpos), before.rfind(". ", 0, qpos), before.rfind("! ", 0, qpos))
        if split_at != -1:
            if before[split_at:split_at+2] in [". ", "! "]:
                question = before[split_at+2:].strip()
                main = text[:split_at+1].strip()
            else:
                question = before[split_at:].strip()
                main = text[:split_at].strip()
            if question.endswith("?"):
                return (main + "\n\n" + line + "\n\n" + question).strip()

    return (text.rstrip() + "\n\n" + line).strip()


def _ensure_one_followup(text, user_message):
    """Ensure the response ends with exactly one relevant, natural follow-up question."""
    text = _remove_extra_questions(_avoid_repeated_opening(text))
    if "?" in text:
        return _add_hopeful_idp_line(text, user_message)
    t = normalize_for_safety(user_message or "")
    if any(w in t for w in ["peace", "peaceful", "prayer", "praying", "allah", "dua", "namaz"]):
        q = "What part of that feeling brought you the most comfort?"
    elif any(w in t for w in ["good news", "happy", "joy", "thankful", "grateful"]):
        q = "What made that moment feel special for you?"
    elif any(w in t for w in ["job", "work", "money", "income", "unemployed"]):
        q = "What feels like the most urgent pressure for you right now?"
    elif any(w in t for w in ["family", "children", "parents", "responsibility"]):
        q = "What kind of support does your family need most right now?"
    elif any(w in t for w in ["sleep", "tired", "energy", "exhausted", "drained"]):
        q = "Has this been affecting your whole day or mostly certain times?"
    elif any(w in t for w in ["sad", "low", "hopeless", "stress", "worried", "anxious"]):
        q = "What feels heaviest in this moment?"
    else:
        q = "What would you like to share next?"
    text = text.rstrip(".") + "\n\n" + q
    return _add_hopeful_idp_line(text, user_message)


# ─────────────────────────────────────────────
# ADAPTIVE MEMORY AND RESPONSE LEARNING — NO EMBEDDINGS / DATASET-GUIDED
# ─────────────────────────────────────────────
def _safe_lower(value):
    return str(value or "").lower().strip()


def _tone_tags(text):
    """Light semantic tags for memory summaries. These guide the LLM; they are not rigid replies."""
    t = normalize_for_safety(text or "")
    tags = set()
    groups = {
        "faith": ["allah", "dua", "prayer", "praying", "namaz", "quran", "sabr", "tawakkul", "shukar", "thank allah"],
        "family": ["family", "children", "kids", "parents", "mother", "father", "wife", "husband", "responsibility"],
        "work_money": ["job", "work", "money", "income", "unemployed", "earn", "fees", "expense", "provide"],
        "sleep_energy": ["sleep", "slept", "tired", "energy", "exhausted", "drained", "weak"],
        "sadness": ["sad", "low", "worse", "worst", "hopeless", "empty", "cry", "depressed"],
        "anxiety": ["stress", "stressed", "worried", "anxious", "panic", "fear", "overthinking"],
        "hope_positive": ["better", "happy", "hope", "hopeful", "good news", "peace", "peaceful", "thankful", "grateful", "handled", "managed"],
        "idp_life": ["camp", "shelter", "tent", "flood", "relief", "temporary", "lost home", "left home", "displaced"],
    }
    for tag, words in groups.items():
        if any(w in t for w in words):
            tags.add(tag)
    return tags


def _classify_assistant_style(text):
    """Estimate previous assistant style so future replies can adapt and avoid repetition."""
    t = _safe_lower(text)
    tags = []
    if any(w in t for w in ["allah", "quran", "dua", "sabr", "prayer", "may allah"]):
        tags.append("faith-based")
    if any(w in t for w in ["try", "small step", "routine", "one thing", "practical", "plan", "support"]):
        tags.append("practical")
    if any(w in t for w in ["i hear", "sounds", "makes sense", "not easy", "heavy", "with you"]):
        tags.append("validation")
    if any(w in t for w in ["reflect", "notice", "inside", "meaning", "heart", "calm", "gentle"]):
        tags.append("reflective")
    words = t.split()
    if len(words) <= 45:
        tags.append("short")
    elif len(words) >= 95:
        tags.append("deep")
    return tags or ["supportive"]


def _user_engaged_after(text):
    """Approximate engagement from the user's reply following an assistant message."""
    t = _safe_lower(text)
    if not t:
        return "none"
    if len(t.split()) >= 7:
        return "engaged"
    if any(w in t for w in ["yes", "thanks", "thank", "ok", "okay", "good", "right", "true"]):
        return "light"
    return "brief"


def build_adaptive_memory_context(user_id, current_message=""):
    """Summarise long-term memory from SQLite without embeddings.

    Uses chat_history, wellbeing_logs, assessments, and previous assistant responses.
    The summary is sent to the LLM as guidance only; it should never be dumped to the user.
    """
    if not user_id:
        return "No saved long-term memory yet."
    try:
        db = get_db()
        rows = db.execute("""
            SELECT sender, message, timestamp
            FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 40
        """, (user_id,)).fetchall()
        rows = list(reversed(rows))
        user_msgs = [str(r["message"]) for r in rows if r["sender"] == "user"]
        assistant_msgs = [str(r["message"]) for r in rows if r["sender"] == "assistant"]
        all_user_text = " ".join(user_msgs[-20:])
        current_tags = _tone_tags(current_message)
        past_tags = _tone_tags(all_user_text)

        recurring = []
        if "family" in past_tags:
            recurring.append("family/responsibility")
        if "work_money" in past_tags:
            recurring.append("work, income, or financial pressure")
        if "sleep_energy" in past_tags:
            recurring.append("sleep or low energy")
        if "sadness" in past_tags:
            recurring.append("low mood/heaviness")
        if "anxiety" in past_tags:
            recurring.append("stress or worry")
        if "faith" in past_tags:
            recurring.append("faith as comfort")
        if "idp_life" in past_tags:
            recurring.append("living/adjustment pressures")

        current_norm = normalize_for_safety(current_message or "")
        trend_hint = ""
        if any(w in current_norm for w in ["better", "calm", "peace", "peaceful", "handled", "managed", "happy", "hopeful"]):
            if any(tag in past_tags for tag in ["sadness", "anxiety", "sleep_energy", "work_money", "family"]):
                trend_hint = "The user may be showing improvement or a lighter moment compared with earlier difficulty. Acknowledge growth gently."
        elif any(w in current_norm for w in ["again", "still", "worse", "worst", "hard again", "not better"]):
            trend_hint = "The user may be describing a repeated or worsening pattern. Recognize it gently without sounding like a database lookup."

        wellbeing_rows = db.execute("""
            SELECT sleep_hours, mood, energy, source_text, log_date
            FROM wellbeing_logs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 5
        """, (user_id,)).fetchall()
        wellbeing_bits = []
        if wellbeing_rows:
            latest = wellbeing_rows[0]
            if latest["sleep_hours"] is not None:
                wellbeing_bits.append(f"latest sleep around {latest['sleep_hours']} hours")
            if latest["mood"]:
                wellbeing_bits.append(f"latest mood: {latest['mood']}")
            if latest["energy"]:
                wellbeing_bits.append(f"latest energy: {latest['energy']}")
            sleep_values = [float(r["sleep_hours"]) for r in wellbeing_rows if r["sleep_hours"] is not None]
            if len(sleep_values) >= 2:
                if sleep_values[0] > sleep_values[-1]:
                    wellbeing_bits.append("sleep appears somewhat improved compared with earlier logs")
                elif sleep_values[0] < sleep_values[-1]:
                    wellbeing_bits.append("sleep may have become lower than earlier logs")

        assess = db.execute("""
            SELECT phq9_total, gad7_total, who5_total, severity, advice_given, assessment_date
            FROM assessments
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,)).fetchone()
        assessment_bit = ""
        if assess and assess["phq9_total"] is not None:
            assessment_bit = f"Last assessment: PHQ9={assess['phq9_total']}, GAD7={assess['gad7_total']}, WHO5={assess['who5_total']}, severity={assess['severity']}."

        style_scores = {}
        openings = []
        for i, r in enumerate(rows):
            if r["sender"] == "assistant":
                msg = str(r["message"])
                first_sentence = re.split(r"[.!?]", msg.strip())[0].strip()
                if first_sentence:
                    openings.append(first_sentence[:120])
                next_user = ""
                for nxt in rows[i+1:]:
                    if nxt["sender"] == "user":
                        next_user = str(nxt["message"])
                        break
                engagement = _user_engaged_after(next_user)
                weight = 2 if engagement == "engaged" else 1 if engagement == "light" else 0
                for style in _classify_assistant_style(msg):
                    style_scores[style] = style_scores.get(style, 0) + weight

        preferred_styles = [k for k, v in sorted(style_scores.items(), key=lambda x: x[1], reverse=True) if v > 0][:3]
        if not preferred_styles:
            preferred_styles = ["warm validation", "gentle practical support"]
        avoid_openings = []
        seen = set()
        for op in reversed(openings[-8:]):
            key = op.lower()
            if key and key not in seen:
                avoid_openings.append(op)
                seen.add(key)
            if len(avoid_openings) >= 3:
                break

        profile = get_user_memory_profile(user_id) or ""
        lines = []
        if profile:
            lines.append("Saved profile: " + profile)
        if recurring:
            lines.append("Recurring themes: " + ", ".join(recurring) + ".")
        if current_tags:
            lines.append("Current message signals: " + ", ".join(sorted(current_tags)) + ".")
        if trend_hint:
            lines.append("Trend hint: " + trend_hint)
        if wellbeing_bits:
            lines.append("Wellbeing trend: " + "; ".join(wellbeing_bits) + ".")
        if assessment_bit:
            lines.append(assessment_bit)
        lines.append("Preferred response style inferred from user engagement: " + ", ".join(preferred_styles) + ".")
        if avoid_openings:
            lines.append("Avoid repeating these recent openings: " + " | ".join(avoid_openings) + ".")
        if user_msgs[-6:]:
            lines.append("Recent user statements: " + " | ".join(user_msgs[-6:]) + ".")
        return "\n".join(lines)
    except Exception as e:
        print("Adaptive memory context error:", e)
        return "Memory exists, but it could not be summarized safely."


def _fallback_natural_response(user_message, sess, emotion="unknown"):
    """Non-LLM fallback: broad, varied, context-aware, and not tied to one exact keyword reply."""
    name = sess.get("name", "")
    opener = f"{name}, " if name else ""
    t = normalize_for_safety(user_message or "")

    if is_metaphor_kill(user_message):
        base = random.choice([
            f"{opener}it sounds like this exam or project is taking a lot of energy from you, and you want to tackle it strongly. Even when life feels unsettled, breaking one hard thing into smaller steps can make it feel more possible.",
            f"{opener}I hear the pressure behind your words — you want to handle this properly, not let it defeat you. After everything you have been managing, even one focused step can be meaningful.",
            f"{opener}that sounds like determination mixed with stress. When responsibilities pile up, it can help to choose the next small task instead of carrying the whole project at once."
        ])
    elif any(w in t for w in ["peace", "peaceful", "prayer", "praying", "namaz", "allah", "dua", "tawakkul"]):
        base = random.choice([
            f"{opener}there is a calm strength in what you shared. Faith can sometimes give the heart a place to rest when life feels uncertain.",
            f"{opener}that sounds like a quiet kind of peace. When prayer settles the heart, even a difficult day can feel a little easier to face.",
            f"{opener}I can hear how much comfort this brings you. Holding onto trust in Allah can become a real source of steadiness."
        ])
    elif any(w in t for w in ["happy", "good news", "thankful", "grateful", "better", "hope", "will get better", "positive"]):
        base = random.choice([
            f"{opener}I can hear a brighter feeling in your words. After everything you have been managing, even one hopeful moment matters.",
            f"{opener}that sounds like something your heart really needed today. It is good to notice these lighter moments when they come.",
            f"{opener}there is hope in the way you are speaking. That does not erase the hard parts, but it gives you something steady to hold."
        ])
    elif any(w in t for w in ["job", "work", "money", "income", "unemployed", "earn", "support my family"]):
        base = random.choice([
            f"{opener}wanting to provide for your family while work is uncertain can feel deeply painful. Your care for them is clear, even before money or a job is solved.",
            f"{opener}that pressure makes sense. When income is uncertain, it can feel like your worth is being measured by what you can provide, but you are more than that.",
            f"{opener}I hear the responsibility behind your words. Supporting family can also mean staying steady, helping with small daily things, and taking one practical step at a time."
        ])
    elif any(w in t for w in ["family", "children", "parents", "responsibility", "pressure"]):
        base = random.choice([
            f"{opener}family responsibilities can sit very heavily on the heart, especially when life around you is already unsettled.",
            f"{opener}I can hear how much your family matters to you. Carrying that love and pressure together can be exhausting.",
            f"{opener}that sounds like a lot to hold quietly. Wanting to protect your family while managing your own feelings is not easy."
        ])
    elif any(w in t for w in ["sad", "low", "worse", "worst", "hopeless", "empty", "cry"]):
        base = random.choice([
            f"{opener}that sounds really heavy. When life has been unstable for a while, low feelings can come with extra weight.",
            f"{opener}I am listening. This does not sound like a small mood change; it sounds like something has been pressing on you.",
            f"{opener}some days feel hard to explain, but the heaviness is still real. You do not have to make it sound perfect here."
        ])
    elif any(w in t for w in ["stress", "stressed", "anxious", "worried", "fear", "panic", "overthinking"]):
        base = random.choice([
            f"{opener}your mind sounds crowded right now. When there is uncertainty around you, stress can become louder than usual.",
            f"{opener}that kind of pressure can make it hard to breathe freely. Let us slow it down together for a moment.",
            f"{opener}I hear the tension in this. Stress often feels bigger when there are too many things to manage at once."
        ])
    elif any(w in t for w in ["sleep", "tired", "exhausted", "drained", "energy", "weak"]):
        base = random.choice([
            f"{opener}that sounds draining. When rest is disturbed, emotions and daily responsibilities both become harder to carry.",
            f"{opener}low energy can quietly affect everything — mood, patience, focus, and hope.",
            f"{opener}your body and mind may be asking for gentleness, not more pressure."
        ])
    else:
        base = random.choice([
            f"{opener}I am with you. I want to understand what this moment feels like for you, not just reply quickly.",
            f"{opener}you can say it simply here. I will try to stay with the meaning behind your words.",
            f"{opener}I hear you. Let us take this one piece at a time, without making it feel like a form or test."
        ])
    return _ensure_one_followup(base, user_message)



def build_dataset_guidance_context(user_message, role="default", emotion="unknown"):
    """Use local training/testing assets as understanding signals, not as fixed replies.

    This keeps the chatbot from becoming rule-based: datasets and ML models guide
    tone, likely concerns, and risk awareness, while the final wording is generated
    naturally by the LLM/fallback layer.
    """
    parts = []
    try:
        role_key = role or "default"
        if 'IDP_SURVEY_INSIGHTS' in globals() and role_key in IDP_SURVEY_INSIGHTS:
            insight = IDP_SURVEY_INSIGHTS.get(role_key, {})
            top_phq = insight.get("top_phq", [])
            top_gad = insight.get("top_gad", [])
            if top_phq or top_gad:
                parts.append(
                    "Role-based survey signal: for this role, commonly observed concerns include "
                    + ", ".join((top_phq + top_gad)[:4])
                    + ". Use this only as background guidance."
                )
    except Exception as e:
        print("Dataset survey guidance error:", e)

    try:
        # Use retrieval system only as a lightweight topic signal, never as a copy-paste answer.
        retrieved = retrieve_from_training_data(user_message, emotion=emotion, role=role, top_k=2)
        if retrieved:
            topics = []
            for item in retrieved:
                topic = item.get("topic", "general")
                source = item.get("source", "training")
                sim = item.get("similarity", "")
                topics.append(f"{topic} ({source}, similarity {sim})")
            parts.append(
                "Training-data semantic signal: similar examples relate to "
                + "; ".join(topics)
                + ". Do not copy dataset answers; use only to understand likely context."
            )
    except Exception as e:
        print("Dataset semantic guidance error:", e)

    if emotion and emotion != "unknown":
        parts.append(f"ML emotion signal: {emotion}. Treat this as a hint, not a label to repeat to the user.")

    return "\n".join(parts) if parts else "No strong dataset signal; rely on the user's exact words and saved memory."

def generate_natural_chat_response(user_message, sess, emotion="unknown", wellbeing_info=None):
    """Generate a warm, contextual reply for normal chat.

    Safety/guardrail checks happen before this function in /chat. Self-assessment remains separate.
    If OPENAI_API_KEY is set, this uses an LLM-style response. If not, it falls back gracefully.
    """
    api_key_available = bool(os.environ.get("OPENAI_API_KEY"))
    recent = _recent_dialogue_for_prompt(limit=10)
    user_context = _compact_user_context(sess, emotion=emotion, wellbeing_info=wellbeing_info)
    adaptive_context = build_adaptive_memory_context(sess.get("user_id"), current_message=user_message)
    dataset_context = build_dataset_guidance_context(user_message, role=sess.get("role", "default"), emotion=emotion)
    metaphor_context = ""
    if is_metaphor_kill(user_message):
        metaphor_context = (
            "Language note: the user is using 'kill' as a harmless metaphor for doing well or tackling an exam/project/task. "
            "Do not treat it as self-harm, violence, danger, or 'extreme thoughts'. Respond to the underlying stress or determination naturally."
        )
    last_assistant = ""
    for item in reversed(list(sess.get("history", []))):
        if item.get("role") == "assistant":
            last_assistant = str(item.get("content", ""))[:600]
            break

    if api_key_available:
        try:
            system_prompt = """
You are Empath AI, a warm mental-health support chatbot for internally displaced people in Pakistan.
Your job is to respond like a thoughtful supportive friend: gentle, emotionally intelligent, natural, respectful, and specific to the user's exact message.

Core behaviour:
- Use the user's role, recent conversation, saved memory, wellbeing trends, assessments, ML emotion signal, and dataset guidance to understand the situation; generate fresh wording rather than fixed templates.
- Keep IDP context as quiet background. Assume life may be unsettled, but do NOT repeatedly say "as an IDP", "because you are displaced", or ask whether they are displaced.
- Include exactly one hopeful, soft IDP-sensitive sentence in every normal reply. It should validate the user's real-life situation without making them lose hope. Use natural wording like "after everything you have been managing", "when life has been unsettled", or "even with so much change around you".
- Do NOT sound clinical, robotic, scripted, or like a survey.
- Do NOT reuse the same opening, structure, or emotional phrase. Avoid the recent assistant patterns provided below.
- Slightly reflect the user's exact words in the first sentence so they feel understood.
- Ask exactly ONE natural follow-up question at the end, based on this exact message.
- Keep the answer concise: usually 3 to 6 sentences. Sometimes short is better.
- Give practical support when the user asks for help or seems stuck, but do not overwhelm with long lists.
- For positive messages, distinguish the feeling: happiness, hope, peace, achievement, gratitude, good news, or faith. Make each response unique.
- If the user shows improvement compared with previous chats, acknowledge growth gently. If struggle repeats, recognize the pattern softly.
- Learn silently from earlier assistant responses and user engagement: adapt tone, depth, faith references, and practicality without saying you are learning.
- Do not start PHQ-9/GAD-7/WHO-5 unless the user clearly asks for assessment.
- Crisis and psychosis handling are checked before you. If the current message still sounds unsafe or detached from reality, stay calm, do not validate unusual beliefs as real, and gently encourage trusted/professional human support.
""".strip()
            user_prompt = f"""
User/session context:
{user_context}

Adaptive memory and response-learning context from database (use subtly; do not dump it):
{adaptive_context}

Dataset/ML guidance for understanding only (do not copy as a response):
{dataset_context}

Context/language note if any:
{metaphor_context}

Recent conversation:
{recent}

Last assistant response to avoid repeating:
{last_assistant}

Current user message:
{user_message}

Write the next Empath AI reply now.
""".strip()
            result = client.chat.completions.create(
                model=os.environ.get("EMPATH_AI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.85,
                max_tokens=220,
            )
            response = result.choices[0].message.content.strip()
            return _ensure_one_followup(response, user_message)
        except Exception as e:
            print("Natural OpenAI response error:", e)

    return _fallback_natural_response(user_message, sess, emotion=emotion)


# CHAT SAFETY FAIL-SAFE HELPERS
# Only protects /chat safety routing; normal flow remains unchanged.
def _crisis_plain_fallback(name=""):
    display_name = f"{name}, " if name else ""
    return (
        f"{display_name}I can hear this is a very dangerous and painful moment. "
        "Right now, your safety matters more than anything else. Please do not stay alone; sit near someone you trust or call them now. "
        "You can also call 1122 for emergency help or Umang at 0317-4288665 for immediate support. "
        "Can you move close to another person right now?"
    )

def _safe_record_user_message(user_message):
    try:
        add_to_history("user", user_message)
    except Exception as e:
        print("Safe user history record error:", e)
    try:
        if session.get("user_id"):
            save_chat_message(session["user_id"], "user", user_message)
    except Exception as e:
        print("Safe user DB record error:", e)

def _safe_return_chat_response(response, response_type="normal", stage=None):
    try:
        add_to_history("assistant", response)
    except Exception as e:
        print("Safe assistant history record error:", e)
    try:
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
    except Exception as e:
        print("Safe assistant DB record error:", e)

    payload = {"response": response, "type": response_type}
    if stage is not None:
        payload["stage"] = stage
    return jsonify(payload)

def _early_safety_response_if_needed(user_message):
    """Return urgent safety response before DB/OpenAI work."""
    try:
        safety = run_safety_checks(user_message)
    except Exception as e:
        print("Early safety check error:", e)
        norm = normalize_for_safety(user_message)
        if direct_crisis_override(norm):
            _safe_record_user_message(user_message)
            response = _crisis_plain_fallback(session.get("name", ""))
            session["crisis_active"] = True
            session["crisis_level"] = "emergency"
            return _safe_return_chat_response(response, "emergency", session.get("stage"))
        return None

    stype = safety.get("type")
    if stype not in {"blocked", "grief_dream", "psychosis", "special_support", "emergency", "crisis"}:
        return None

    _safe_record_user_message(user_message)

    try:
        if stype in ["emergency", "crisis"]:
            session["crisis_active"] = True
            session["crisis_level"] = safety.get("level", "crisis")
            try:
                response = build_crisis_response(user_message, session.get("name", ""))
            except Exception as e:
                print("Crisis response builder error:", e)
                response = _crisis_plain_fallback(session.get("name", ""))
            return _safe_return_chat_response(response, stype, session.get("stage"))

        if stype == "blocked":
            return _safe_return_chat_response(safety.get("response", "I care about your wellbeing, so I cannot help with that. But I can listen to what you are going through right now."), "blocked", session.get("stage"))

        if stype == "grief_dream":
            response = build_grief_dream_support_response(user_message, session.get("name", ""), session.get("role", "default"))
            return _safe_return_chat_response(response, "grief_dream", session.get("stage"))

        if stype == "psychosis":
            response = build_psychosis_support_response(user_message, session.get("name", ""), session.get("role", "default"))
            return _safe_return_chat_response(response, "psychosis", session.get("stage"))

        if stype == "special_support":
            return _safe_return_chat_response(safety.get("response", "That sounds really unsettling, and I am glad you told me. You do not have to carry this alone. Is someone you trust nearby right now?"), "special_support", session.get("stage"))

    except Exception as e:
        print("Early safety route error:", e)
        response = _crisis_plain_fallback(session.get("name", ""))
        session["crisis_active"] = True
        session["crisis_level"] = "crisis"
        return _safe_return_chat_response(response, "crisis", session.get("stage"))

    return None

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "I did not catch that. Could you say that again?"})

    # CRITICAL: urgent safety messages must never depend on DB writes or OpenAI/API calls.
    early_safety_response = _early_safety_response_if_needed(user_message)
    if early_safety_response is not None:
        return early_safety_response

    if "stage" not in session:
        init_db()
        init_session()
        if session.get("account_id") and session.get("account_name"):
            account_id = session.get("account_id")
            first_name = session.get("account_name", "").split()[0].capitalize() if session.get("account_name") else ""
            session["name"] = first_name
            linked_user = get_user_by_account_id(account_id)
            if linked_user:
                session["user_id"] = linked_user["id"]
                session["is_returning_user"] = True
                session["role"] = linked_user["role"] if linked_user["role"] else "default"
                session["living_place"] = get_row_value(linked_user, "living_place", "")
                session["user_memory_profile"] = get_user_memory_profile(linked_user["id"])
                update_user_last_seen(linked_user["id"])
                load_recent_self_assessment(linked_user["id"])
                session["stage"] = STAGE_LISTEN
            elif first_name:
                new_user_id = create_user(first_name, account_id=account_id)
                session["user_id"] = new_user_id
                session["is_returning_user"] = False
                session["role"] = "default"
                session["stage"] = STAGE_ROLE

    stage = session.get("stage", STAGE_NAME)
    add_to_history("user", user_message)

    # Save user message to DB if known
    if session.get("user_id"):
        save_chat_message(session["user_id"], "user", user_message)

    # Save wellbeing only if naturally mentioned
    wellbeing_info = None
    if session.get("user_id"):
        wellbeing_info = save_wellbeing_if_mentioned(session["user_id"], user_message)

    # Learn compact user profile from this message, not the full chat.
    if session.get("user_id"):
        try:
            update_user_memory_profile(session["user_id"], user_message, session.get("emotion", "unknown"), session.get("crisis_level", "safe"))
            session["user_memory_profile"] = get_user_memory_profile(session["user_id"])
        except Exception as e:
            print("User memory update error:", e)

    # Save displacement living-place memory when user mentions it.
    detected_place = detect_living_place(user_message)
    if detected_place:
        session["living_place"] = detected_place
        if session.get("user_id"):
            update_user_living_place(session["user_id"], detected_place)

    safety = run_safety_checks(user_message)

    if safety["type"] in ["emergency", "crisis"]:
        session["crisis_active"] = True
        session["crisis_level"] = safety.get("level", "crisis")

    if safety["type"] == "blocked":
        response = safety["response"]
        add_to_history("assistant", response)
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
        return jsonify({"response": response, "type": "blocked"})

    if safety["type"] == "grief_dream":
        response = build_grief_dream_support_response(
            user_message,
            session.get("name", ""),
            session.get("role", "default")
        )
        add_to_history("assistant", response)
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
        return jsonify({"response": response, "type": "grief_dream"})

    if safety["type"] == "psychosis":
        response = build_psychosis_support_response(
            user_message,
            session.get("name", ""),
            session.get("role", "default")
        )
        add_to_history("assistant", response)
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
        return jsonify({"response": response, "type": "psychosis"})

    if safety["type"] == "special_support":
        response = safety["response"]
        add_to_history("assistant", response)
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
        return jsonify({"response": response, "type": "special_support"})

    if safety["type"] == "emergency":
        response = build_crisis_response(user_message, session.get("name", ""))
        add_to_history("assistant", response)
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
        return jsonify({"response": response, "type": "emergency"})

    if safety["type"] == "crisis":
        response = build_crisis_response(user_message, session.get("name", ""))
        add_to_history("assistant", response)
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
        return jsonify({"response": response, "type": "crisis"})

    crisis_level = session.get("crisis_level", safety.get("level", "safe"))
    emotion, conf = detect_emotion(user_message)
    session["emotion"] = emotion
    session["confidence"] = conf
    session["is_positive"] = is_positive_message(user_message)
    session["idp_hardships"] = detect_idp_hardships(session.get("history", []))

    if session.get("crisis_active", False):
        response = get_crisis_followup_response(session.get("name", ""), user_message)
        add_to_history("assistant", response)
        if session.get("user_id"):
            save_chat_message(session["user_id"], "assistant", response)
        return jsonify({"response": response, "type": "crisis", "stage": session.get("stage")})

    response = ""

    if stage == STAGE_NAME:
        if looks_like_name(user_message):
            name = user_message.split()[0].capitalize()
            session["name"] = name

            account_id = session.get("account_id")
            existing_user = get_user_by_account_id(account_id) if account_id else find_existing_user(name)

            if existing_user:
                session["user_id"] = existing_user["id"]
                session["is_returning_user"] = True
                session["role"] = existing_user["role"] if existing_user["role"] else "default"
                session["living_place"] = get_row_value(existing_user, "living_place", "")
                update_user_last_seen(existing_user["id"])
                load_recent_self_assessment(existing_user["id"])
                response = (
                    f"Welcome back, {name}. We talked last time. "
                    "Tell me, how have you been feeling lately?"
                )
                session["stage"] = STAGE_LISTEN
            else:
                new_user_id = create_user(name, account_id=session.get("account_id"))
                session["user_id"] = new_user_id
                session["is_returning_user"] = False
                session["role"] = "default"
                load_recent_self_assessment(new_user_id)
                response = (
                    f"Hello, {name}. It's nice to meet you. "
                    "Can you tell me a little about your daily life right now — "
                    "are you studying, working, caring for family, or managing something else?"
                )
                session["stage"] = STAGE_ROLE

        else:
            session["stage"] = STAGE_LISTEN
            response = "I am here with you. What has been weighing on your heart these days?"

    elif stage == STAGE_ROLE:
        role = detect_role(user_message)
        session["role"] = role

        if session.get("user_id"):
            update_user_role(session["user_id"], role)

        if session.get("sa_followup_needed"):
            response = (
                f"Thank you, {session.get('name', '')}. I also noticed you completed a self-assessment recently. "
                f"Before we continue, how have you been feeling lately?"
            )
        else:
            response = f"Thank you, {session.get('name', '')}. How have you been feeling lately?"

        session["stage"] = STAGE_LISTEN

    elif stage == STAGE_LISTEN:
        if wants_assessment(user_message):
            response = _check_and_start_assessment()
            add_to_history("assistant", response)
            if session.get("user_id"):
                save_chat_message(session["user_id"], "assistant", response)
            return jsonify({"response": response, "type": "normal", "stage": session.get("stage")})

        # Natural response engine: uses LLM-style contextual generation when API key is available,
        # with a warm local fallback if not. This prevents rigid rule-based replies.
        response = generate_natural_chat_response(
            user_message=user_message,
            sess=session,
            emotion=emotion,
            wellbeing_info=wellbeing_info
        )
        session["stage"] = STAGE_LISTEN

        # ── Self-assessment follow-up questions (max 3, at turns 1 / 4 / 8) ──
        if session.get("sa_followup_needed") and session.get("sa_followup_count", 0) < 3:
            sa_listen_turns = session.get("sa_listen_turns", 0) + 1
            session["sa_listen_turns"] = sa_listen_turns
            trigger_turns = [1, 4, 8]
            if sa_listen_turns in trigger_turns:
                concern = session.get("sa_dominant_concern", "mild")
                followup_list = SA_FOLLOWUP_QUESTIONS.get(concern, SA_FOLLOWUP_QUESTIONS["mild"])
                idx = session.get("sa_followup_count", 0)
                if idx < len(followup_list):
                    response += "\n\n" + followup_list[idx]
                    session["sa_followup_count"] = idx + 1
                    if session["sa_followup_count"] >= 3:
                        session["sa_followup_needed"] = False
        else:
            session["sa_listen_turns"] = session.get("sa_listen_turns", 0) + 1

        # One-time gentle assessment nudge after 4+ exchanges, for emotional users not yet suggested
        user_turns = sum(1 for m in session.get("history", []) if m.get("role") == "user")
        if (
            not session.get("assessment_suggested")
            and user_turns >= 4
            and emotion in ["depression", "sadness", "grief", "anxiety", "stress", "fear", "ptsd"]
        ):
            user_id_for_check = session.get("user_id")
            already_done_this_week = False
            if user_id_for_check:
                last = get_last_assessment_for_user(user_id_for_check)
                already_done_this_week = days_since_assessment(last) < 7

            if not already_done_this_week:
                response += (
                    "\n\nWhenever you feel ready, I can also walk you through a short weekly check-in — "
                    "it takes about 5 minutes and helps me understand how you have been feeling this week. "
                    "Just say 'start assessment' if you would like to try it, or visit the Self Assessment page."
                )
                session["assessment_suggested"] = True

    elif stage == STAGE_ASSESS:
        name = session.get("name", "")

        if not session.get("assessment_started", False):
            response = _check_and_start_assessment()
            add_to_history("assistant", response)
            if session.get("user_id"):
                save_chat_message(session["user_id"], "assistant", response)
            return jsonify({"response": response, "type": "normal", "stage": session.get("stage")})

        q_index = session.get("q_index", 0)
        number_match = re.search(r"\b([0-4])\b", user_message)

        if number_match is None:
            # Repeat current question with hint
            if q_index < len(SA_QUESTIONS):
                cur = SA_QUESTIONS[q_index]
                hint = "(0–4)" if cur["scale"] == "who" else "(0–3)"
                response = f"Please reply with a number {hint}.\n\n{cur['text']}"
            else:
                response = "Please reply with a number to continue."
        else:
            score = int(number_match.group(1))

            # Save answer for current question
            if q_index < len(SA_QUESTIONS):
                cur = SA_QUESTIONS[q_index]
                max_val = 4 if cur["scale"] == "who" else 3
                clamped = min(max(score, 0), max_val)

                if cur["scale"] == "phq":
                    phq = dict(session.get("phq", {}))
                    phq[cur["key"]] = clamped
                    session["phq"] = phq
                    if cur["key"] == "phq9" and clamped >= 2:
                        session["crisis_active"] = True
                        session["crisis_level"] = "crisis"
                        session["q_index"] = q_index + 1
                        response = build_crisis_response(user_message, name)
                        add_to_history("assistant", response)
                        if session.get("user_id"):
                            save_chat_message(session["user_id"], "assistant", response)
                        return jsonify({"response": response, "type": "crisis"})
                elif cur["scale"] == "gad":
                    gad = dict(session.get("gad", {}))
                    gad[cur["key"]] = clamped
                    session["gad"] = gad
                elif cur["scale"] == "who":
                    who = dict(session.get("who", {}))
                    who[cur["key"]] = clamped
                    session["who"] = who

                session["q_index"] = q_index + 1
                q_index += 1

            # All 21 questions answered?
            if q_index >= len(SA_QUESTIONS):
                phq_total = sum(session.get("phq", {}).values())
                gad_total = sum(session.get("gad", {}).values())
                who_total = sum(session.get("who", {}).values())
                phq_q9 = session.get("phq", {}).get("phq9", 0)

                phq_sev = sa_get_phq9_severity(phq_total)[0]
                gad_sev = sa_get_gad7_severity(gad_total)[0]
                session["severity"] = phq_sev

                prev_assess = None
                if session.get("user_id"):
                    prev_assess = get_last_assessment_for_user(session["user_id"])
                    if prev_assess and prev_assess["phq9_total"] is None:
                        prev_assess = None

                advice, risk_level = generate_assessment_advice(phq_total, gad_total, who_total, phq_q9, prev_assess)
                session["last_advice"] = advice
                session["last_assessment_scores"] = {"phq9": phq_total, "gad7": gad_total, "who5": who_total}

                save_assessment_json(session)
                if session.get("user_id"):
                    save_assessment_to_db(session["user_id"], session, advice=advice, prev=prev_assess)

                session["stage"] = STAGE_RESPOND

                comparison = ""
                if prev_assess and prev_assess["phq9_total"] is not None:
                    chg = phq_total - prev_assess["phq9_total"]
                    if chg >= 3:
                        comparison = f" Your depression score has risen {chg} points since last week."
                    elif chg <= -3:
                        comparison = f" Your depression score improved by {abs(chg)} points from last week."

                response = (
                    f"Thank you for sharing all of that, {name}. Here is a summary of your results:\n\n"
                    f"PHQ-9 (Depression): {score_to_10(phq_total, 27)}/10 — {phq_sev}\n"
                    f"GAD-7 (Anxiety): {score_to_10(gad_total, 21)}/10 — {gad_sev}\n"
                    f"WHO-5 (Wellbeing): {score_to_10(who_total, 25)}/10{comparison}\n\n"
                    f"Here is some personal guidance based on your scores:\n\n{advice}"
                )

                if risk_level == "crisis":
                    session["crisis_active"] = True
                    session["crisis_level"] = "crisis"
            else:
                # Ask next question
                nxt = SA_QUESTIONS[q_index]
                if nxt["scale"] == "who":
                    hint = "(0=At no time  1=Some of the time  2=Less than half  3=More than half  4=All of the time)"
                else:
                    hint = "(0=Not at all  1=Several days  2=More than half the days  3=Nearly every day)"
                response = nxt["text"] + "\n" + hint

    elif stage == STAGE_RESPOND:
        if wants_assessment(user_message):
            response = _check_and_start_assessment()
            add_to_history("assistant", response)
            if session.get("user_id"):
                save_chat_message(session["user_id"], "assistant", response)
            return jsonify({"response": response, "type": "normal", "stage": session.get("stage")})

        response = get_response(user_message, session, crisis_level, emotion)

    else:
        response = "I am here with you. Can you tell me what has been feeling heaviest for you lately?"

    # Final strict topic guard: if the user did not mention sleep/rest/tiredness/energy,
    # do not let model-generated replies suddenly move to sleep.
    _user_l = user_message.lower()
    _sleep_terms = ["sleep", "slept", "sleeping", "insomnia", "night", "awake", "rest", "tired", "drained", "exhausted", "fatigue", "weak", "energy"]
    _response_l = (response or "").lower()
    if not any(t in _user_l for t in _sleep_terms) and any(t in _response_l for t in ["sleep", "sleeping", "slept", "insomnia"]):
        response = build_local_fallback_response(user_message, session, emotion)

    add_to_history("assistant", response)

    if session.get("user_id"):
        save_chat_message(session["user_id"], "assistant", response)

    return jsonify({
        "response": response,
        "type": "normal",
        "stage": session.get("stage"),
        "is_returning_user": session.get("is_returning_user", False)
    })

@app.route("/logs/<name>", methods=["GET"])
def view_logs(name):
    user = find_existing_user(name)
    if not user:
        return jsonify({"error": "User not found"}), 404

    logs = get_recent_wellbeing_logs(user["id"], limit=20)
    return jsonify({
        "user": {
            "id": user["id"],
            "name": user["name"],
            "role": user["role"],
            "first_seen": user["first_seen"],
            "last_seen": user["last_seen"]
        },
        "logs": [dict(row) for row in logs]
    })



# ─────────────────────────────────────────────
# FULLY ADAPTIVE IDP RESPONSE OVERRIDES
# Added to keep replies IDP-specific without sounding robotic.
# These definitions override earlier generic versions at runtime.
# ─────────────────────────────────────────────
def classify_message_tone(user_message):
    """Classify current message so IDP support adapts to the user's tone."""
    t = normalize_for_safety(user_message or "")
    positive_patterns = [
        "i am feeling okay", "i feel okay", "feeling okay", "i am okay", "i'm okay",
        "i am fine", "i feel fine", "feeling fine", "i am good", "i feel good",
        "better today", "calm today", "okay today", "fine today", "good today"
    ]
    low_patterns = [
        "little low", "a little low", "feeling low", "feel low", "low mood", "sad",
        "down", "not good", "not okay", "not fine", "upset", "empty",
        "nothing interesting", "nothing feels interesting", "worst", "not feeling well", "not well"
    ]
    stress_patterns = [
        "stress", "stressed", "pressure", "anxious", "worried", "worry", "panic",
        "overthinking", "fear", "scared", "tense", "burden"
    ]
    study_patterns = ["study", "studies", "student", "exam", "class", "university", "focus", "semester"]
    family_patterns = ["family", "children", "kids", "parents", "mother", "father", "responsibility"]

    if any(p in t for p in positive_patterns):
        return "steady"
    if any(p in t for p in stress_patterns):
        return "stress"
    if any(p in t for p in study_patterns):
        return "study"
    # Only call it family pressure when stress/problem words are present.
    if detect_family_stress(t) or any(p in t for p in ["family pressure", "family problem", "too much responsibility"]):
        return "family"
    if any(p in t for p in low_patterns):
        return "low"
    return "general"


def get_adaptive_idp_line(sess, tone="general"):
    """Return one natural IDP-specific line based on role, place, and mood tone."""
    role = sess.get("role", "default")
    place = sess.get("living_place", "")
    memory_profile = (sess.get("user_memory_profile", "") or "").lower()

    if not place:
        if "staying in an ngo" in memory_profile:
            place = "an NGO"
        elif "staying in a camp" in memory_profile:
            place = "a camp"
        elif "staying in a shelter" in memory_profile:
            place = "a shelter"
        elif "with relatives" in memory_profile:
            place = "with relatives"

    # Do not force displacement language unless the user has actually shared IDP context.
    if not user_is_idp_context(sess):
        return ""

    if place:
        if tone == "steady":
            return f"Even a steady day matters when you are managing life in {place}."
        if tone == "study" or role == "student":
            return f"Trying to study while staying in {place} can make focus and routine much harder."
        if tone == "family" or role in ["parent", "homemaker"]:
            return f"Managing family responsibilities while staying in {place} can quietly drain a person."
        if tone == "stress":
            return f"Being in {place} can keep your mind alert, especially when privacy, noise, or uncertainty are around you."
        if tone == "low":
            return f"Being in {place} after displacement can make low days feel heavier than they look from outside."
        return f"Being in {place} after displacement can affect routine, privacy, and peace of mind."

    if role == "student":
        if tone == "steady":
            return "Even a calmer day matters when studies and routine have been disturbed after displacement."
        return "For a student, displacement can disturb focus, routine, and hope for the future."
    if role in ["homemaker", "parent"]:
        return "After displacement, caring for family while carrying your own feelings can become very heavy."
    if role == "daily_worker":
        return "After displacement, uncertainty around daily work and income can keep stress present all day."
    if role == "unemployed":
        return "After displacement, losing routine and work can make a person feel stuck."
    if role == "elderly":
        return "After displacement, losing familiar surroundings can make ordinary days feel lonely or unsettled."

    if tone == "steady":
        return "Even small calm moments matter when life has been disrupted after displacement."
    if tone == "stress":
        return "Displacement can keep the mind on alert because daily life feels less predictable."
    if tone == "low":
        return "After displacement, emotions can feel heavier because home, routine, and privacy have all been affected."
    return "Displacement can affect peace, routine, privacy, family life, and emotional balance in quiet ways."


def build_adaptive_idp_response(user_message, sess, emotion="unknown"):
    """Adaptive first-response layer for common user states."""
    name = sess.get("name", "")
    opener = f"{name}, " if name else ""
    tone = classify_message_tone(user_message)
    idp_line = get_adaptive_idp_line(sess, tone)
    text = normalize_for_safety(user_message or "")

    # Normal-life positive/food/family-presence messages should not enter IDP emotional templates.
    if is_food_or_recipe_message(text) or is_happy_message(text) or is_neutral_family_presence(text):
        return None
    sleep_mentioned = any(w in text for w in [
        "sleep", "slept", "sleeping", "night", "awake", "insomnia", "rest",
        "tired", "drained", "exhausted", "fatigue", "energy"
    ])

    if tone == "steady":
        return pick_fresh_response([
            f"{opener}that is good to hear. {idp_line} How has your day been going where you are staying these days?",
            f"{opener}I am glad today feels a little steadier. {idp_line} What helped you feel okay today?",
            f"{opener}that sounds like a calmer moment. {idp_line} What part of today felt most manageable?"
        ])

    if tone == "low":
        return pick_fresh_response([
            f"{opener}that sounds like a heavy day. {idp_line} What kind of low is it today — sadness, pressure, emptiness, or just not feeling like yourself?",
            f"{opener}some days feel low without one clear reason. {idp_line} What has been sitting heaviest on your heart today?",
            f"{opener}with everything around you changed, feeling low can make sense. {idp_line} Is this more about studies, family pressure, or the place you are staying?"
        ])

    if tone == "stress":
        return pick_fresh_response([
            f"{opener}that pressure sounds tiring. {idp_line} What is making your mind feel most crowded right now?",
            f"{opener}stress can feel sharper when daily life is uncertain. {idp_line} Is the pressure coming more from family, studies, money, or your living situation?",
            f"{opener}your mind sounds overloaded. {idp_line} What has been the hardest part of today?"
        ])

    if tone == "study":
        return pick_fresh_response([
            f"{opener}trying to study after displacement is not simple. {idp_line} What part feels hardest right now — focus, motivation, space, or fear about the future?",
            f"{opener}it makes sense that studying feels harder when routine has been shaken. {idp_line} Which subject or task feels most difficult today?",
            f"{opener}{idp_line} We do not need to fix all of studies at once. What is one small study task that feels possible today?"
        ])

    if tone == "family":
        return pick_fresh_response([
            f"{opener}family pressure can feel heavier after displacement. {idp_line} What responsibility has been weighing on you most?",
            f"{opener}{idp_line} Carrying family needs while managing your own emotions is a lot. What part feels hardest today?",
            f"{opener}it sounds like you are carrying more than your own feelings. {idp_line} Who in the family are you most worried about right now?"
        ])

    if sleep_mentioned:
        return pick_fresh_response([
            f"{opener}that sounds exhausting. {idp_line} What usually makes rest hardest for you — worry, noise, interruptions, or no quiet space?",
            f"{opener}when rest is disturbed, the whole day can feel heavier. {idp_line} How long has this been going on?",
            f"{opener}your body sounds worn out. {idp_line} Is it more broken sleep, racing thoughts, or the place around you making it hard to rest?"
        ])

    return None


def build_wellbeing_followup(user_message, extracted, name="", role="default"):
    if not extracted or not extracted.get("has_any"):
        return None

    # Let the priority response engine handle happy / cooking / recipe / family-presence messages.
    if is_food_or_recipe_message(user_message) or is_happy_message(user_message) or is_neutral_family_presence(user_message):
        return None

    display_name = f"{name}, " if name else ""
    temp_sess = dict(session)
    if name:
        temp_sess["name"] = name
    if role:
        temp_sess["role"] = role

    tone = classify_message_tone(user_message)
    idp_line = get_adaptive_idp_line(temp_sess, tone)

    # Positive/neutral mood should never be treated like sadness.
    if extracted.get("mentioned_mood") and extracted.get("mood") in ["okay", "good"] and not extracted.get("mentioned_sleep"):
        return pick_fresh_response([
            f"{display_name}that is good to hear. {idp_line} How has your day been going where you are staying these days?",
            f"{display_name}I am glad today feels a little steadier. {idp_line} What helped you feel okay today?",
            f"{display_name}that sounds like a calmer moment. {idp_line} What part of today felt most manageable?"
        ])

    sleep_missing_hours = extracted.get("mentioned_sleep") and extracted.get("sleep_hours") is None
    mood_missing = extracted.get("mentioned_sleep") and not extracted.get("mentioned_mood") and extracted.get("mood") is None
    energy_missing = extracted.get("mentioned_sleep") and not extracted.get("mentioned_energy") and extracted.get("energy") is None

    if sleep_missing_hours and mood_missing and energy_missing:
        return pick_fresh_response([
            f"{display_name}rest being disturbed can make the whole day heavier. {idp_line} Roughly how many hours are you managing these days, and has it affected your mood or energy?",
            f"{display_name}when nights go badly, the rest of the day usually feels heavier too. {idp_line} About how many hours are you sleeping?",
            f"{display_name}that kind of sleep trouble can build up. {idp_line} How many hours are you actually getting these days?"
        ])

    if sleep_missing_hours:
        return pick_fresh_response([
            f"{display_name}roughly how many hours are you sleeping these days?",
            f"{display_name}about how many hours of sleep are you getting most nights?",
            f"{display_name}how many hours are you managing on a usual night?"
        ])

    if extracted.get("mentioned_sleep") and extracted.get("sleep_hours") is not None:
        h = extracted["sleep_hours"]
        if extracted.get("mood") is None and extracted.get("energy") is None:
            return pick_fresh_response([
                f"{display_name}{h:g} hours is quite low. {idp_line} Has that been affecting your mood or energy during the day?",
                f"{display_name}sleeping around {h:g} hours can really wear a person down. {idp_line} How has your mood and energy been?",
                f"{display_name}with only around {h:g} hours, it makes sense if the day feels heavier. {idp_line} Is your mood low too, or is it mainly tiredness?"
            ])
        if extracted.get("mood") is None:
            return pick_fresh_response([
                f"{display_name}and how has your mood been with that kind of rest?",
                f"{display_name}has that been affecting your mood too?",
                f"{display_name}what has your mood been like lately with rest being like this?"
            ])
        if extracted.get("energy") is None:
            return pick_fresh_response([
                f"{display_name}has your energy been low too?",
                f"{display_name}and what about your energy through the day?",
                f"{display_name}is that also leaving you drained during the day?"
            ])

    if extracted.get("mentioned_mood") and extracted.get("mood") is not None and not extracted.get("mentioned_sleep"):
        if extracted.get("mood") in ["very_low", "low", "anxious"]:
            return pick_fresh_response([
                f"{display_name}that sounds like a heavy day. {idp_line} What kind of low is it today — sadness, pressure, emptiness, or just not feeling like yourself?",
                f"{display_name}some days feel low without one clear reason. {idp_line} What has been sitting heaviest on your heart today?",
                f"{display_name}with everything around you changed, feeling low can make sense. {idp_line} Is this more about studies, family pressure, or the place you are staying?"
            ])

    if extracted.get("mentioned_energy") and extracted.get("energy") is not None and not extracted.get("mentioned_sleep"):
        return pick_fresh_response([
            f"{display_name}that sounds draining. {idp_line} Has this tiredness been there all day, or does it come and go?",
            f"{display_name}low energy can wear a person down quietly. {idp_line} What do you think has been behind it lately?",
            f"{display_name}that sounds tiring. {idp_line} Has this been more physical tiredness, mental heaviness, or both?"
        ])

    return None





# ─────────────────────────────────────────────
# ADAPTIVE CONVERSATION LEARNING ENGINE
# ─────────────────────────────────────────────
def update_conversation_learning(memory, msg):
    """Update small session-based learning memory from the latest user message."""
    if not isinstance(memory, dict):
        memory = {}

    msg = (msg or "").lower()

    # detect recurring stress pattern
    if "money" in msg or any(w in msg for w in ["fees", "expense", "income", "financial"]):
        memory["stress_type"] = "financial"
    elif any(w in msg for w in ["study", "studies", "exam", "focus", "university", "class"]):
        memory["stress_type"] = "academic"
    elif detect_family_stress(msg):
         memory["stress_type"] = "family"

    if "tired" in msg or any(w in msg for w in ["exhausted", "drained", "weak", "fatigue"]):
        memory["energy_pattern"] = "low"

    if any(w in msg for w in ["sad", "low", "down", "hopeless", "empty"]):
        memory["mood_pattern"] = "low"

    if any(w in msg for w in ["displaced", "camp", "shelter", "ngo", "temporary place", "left my home", "lost my home"]):
        memory["displaced"] = True

    return memory


def detect_intent(msg):
    """Lightweight intent detector used before the heavier OpenAI natural-response fallback."""
    msg = (msg or "").lower().strip()

    return {
        "positive": bool(re.search(r"\b(happy|good|yay|great|possible|better|fine)\b", msg)) and not bool(re.search(r"\b(not good|not fine|not okay|not better)\b", msg)),
        "neutral": bool(re.search(r"\b(okay|fine|normal)\b", msg)),
        "low": bool(re.search(r"\b(low|sad|down|hopeless|empty|upset)\b", msg)),
        "stress": "stress" in msg or "pressure" in msg,
        "tired": "tired" in msg or any(w in msg for w in ["exhausted", "drained", "weak", "fatigue"]),
        "money": "money" in msg or any(w in msg for w in ["fees", "expense", "income", "financial"]),
        "study": any(w in msg for w in ["study", "studies", "exam", "focus", "university", "class"]),
        "family": detect_family_stress(msg),
        "crisis": any(w in msg for w in ["harm", "suicide", "die", "kill myself", "end my life"]),
        "unsure": msg in ["nothing", "everything", "i don't know", "idk", "dont know", "don't know"]
    }



def is_happy_message(text):
    """Detect clearly positive/happy messages without forcing a follow-up question."""
    t = normalize_for_safety(text)

    negative_phrases = [
        "not happy", "not good", "not okay", "not fine", "not better",
        "sad", "low", "depressed", "hopeless", "empty", "tired", "stressed"
    ]
    if any(p in t for p in negative_phrases):
        return False

    happy_patterns = [
        r"\bi am happy\b",
        r"\bi'm happy\b",
        r"\bvery happy\b",
        r"\bfeeling happy\b",
        r"\bi feel happy\b",
        r"\bi feel good\b",
        r"\bi am good\b",
        r"\bi'm good\b",
        r"\bfeeling better\b",
        r"\bi feel better\b",
        r"\bgreat\b",
        r"\bexcited\b",
        r"\bpositive\b",
        r"\bcalm\b",
        r"\bpeaceful\b",
        r"\bnice day\b",
        r"\bgood day\b",
    ]
    return any(re.search(p, t) for p in happy_patterns)


def is_food_or_recipe_message(text):
    """Detect food/recipe/cooking requests so they are not treated as distress."""
    t = normalize_for_safety(text)

    food_words = [
        "recipe", "recipes", "rice", "daal", "dal", "chawal", "pulao", "biryani",
        "khichri", "khichdi", "sabzi", "cooking", "cook", "food", "dish",
        "made a recipe", "made food", "made a good recipe", "made a very good recipe",
        "make a good recipe", "i make a good recipe", "i made a good recipe",
        "chicken", "karahi"
    ]

    return any(w in t for w in food_words)


def build_positive_response_for_user(msg, memory):
    """Positive replies should give hope, not remind the user of pain unnecessarily."""
    name = session.get("name", "")
    opener = f"{name}, " if name else ""

    is_idp = bool(memory.get("displaced")) or user_is_idp_context(session)

    if is_idp:
        return (
            f"{opener}I am really glad to hear that. "
            "It is beautiful that you are still noticing good moments even in your current situation. "
            "That shows hope and strength. I hope you keep finding more peaceful and happy moments."
        )

    return (
        f"{opener}I am really glad to hear that. "
        "You are thinking positively today, and that is a good thing. "
        "I hope you always stay happy and peaceful."
    )

def generate_response(msg, memory):
    """Deprecated rule-based priority engine.

    Kept for compatibility with older calls, but intentionally returns None so the
    natural response engine can handle normal conversation.
    """
    return None



_original_get_response_for_adaptive_idp = get_response



# ==============================
# ✅ SELF ASSESSMENT ROUTE (FIX 404)
# ==============================
@app.route("/self_assessment")
def self_assessment():
    return render_template("self_assessment.html")


# ─────────────────────────────────────────────
# SELF-ASSESSMENT CHAT (PHQ-9 + GAD-7 + WHO-5)
# ─────────────────────────────────────────────

SA_QUESTIONS = [
    # PHQ-9
    {"key": "phq1", "scale": "phq", "text": "Over the past one week — how often have you had little interest or pleasure in doing things?"},
    {"key": "phq2", "scale": "phq", "text": "How often have you been feeling down, depressed, or hopeless?"},
    {"key": "phq3", "scale": "phq", "text": "How often have you had trouble falling or staying asleep, or sleeping too much?"},
    {"key": "phq4", "scale": "phq", "text": "How often have you felt tired or had little energy?"},
    {"key": "phq5", "scale": "phq", "text": "How often have you had poor appetite or been overeating?"},
    {"key": "phq6", "scale": "phq", "text": "How often have you felt bad about yourself — or that you are a failure or have let yourself or your family down?"},
    {"key": "phq7", "scale": "phq", "text": "How often have you had trouble concentrating on things, such as reading or watching something?"},
    {"key": "phq8", "scale": "phq", "text": "How often have you been moving or speaking more slowly than usual, or been so restless that others have noticed?"},
    {"key": "phq9", "scale": "phq", "text": "How often have thoughts come up that you would be better off dead, or thoughts of hurting yourself in some way?"},
    # GAD-7
    {"key": "gad1", "scale": "gad", "text": "Now a few questions about worry and anxiety. How often have you felt nervous, anxious, or on edge?"},
    {"key": "gad2", "scale": "gad", "text": "How often have you been unable to stop or control worrying?"},
    {"key": "gad3", "scale": "gad", "text": "How often have you been worrying too much about different things?"},
    {"key": "gad4", "scale": "gad", "text": "How often have you had trouble relaxing?"},
    {"key": "gad5", "scale": "gad", "text": "How often have you been so restless that it was hard to sit still?"},
    {"key": "gad6", "scale": "gad", "text": "How often have you become easily annoyed or irritable?"},
    {"key": "gad7", "scale": "gad", "text": "How often have you felt afraid — as if something awful might happen?"},
    # WHO-5
    {"key": "who1", "scale": "who", "text": "Last few questions about wellbeing. Over the past two weeks, how often have you felt cheerful and in good spirits?"},
    {"key": "who2", "scale": "who", "text": "How often have you felt calm and relaxed?"},
    {"key": "who3", "scale": "who", "text": "How often have you felt active and full of energy?"},
    {"key": "who4", "scale": "who", "text": "How often did you wake up feeling fresh and rested?"},
    {"key": "who5", "scale": "who", "text": "How often has your daily life been filled with things that interest you?"},
]

PHQ_GAD_OPTIONS = [
    {"label": "0 — Not at all", "value": "0"},
    {"label": "1 — Several days", "value": "1"},
    {"label": "2 — More than half the days", "value": "2"},
    {"label": "3 — Nearly every day", "value": "3"},
]

WHO_OPTIONS = [
    {"label": "0 — At no time", "value": "0"},
    {"label": "1 — Some of the time", "value": "1"},
    {"label": "2 — Less than half the time", "value": "2"},
    {"label": "3 — More than half the time", "value": "3"},
    {"label": "4 — All of the time", "value": "4"},
]


def score_to_10(raw, max_raw):
    """Normalise a raw score to a 0-10 scale, rounded to 1 decimal."""
    return round(raw / max_raw * 10, 1)

def sa_get_phq9_severity(score):
    if score <= 4:   return ("Minimal",          "Your score suggests minimal depression symptoms right now.")
    if score <= 9:   return ("Mild",              "Your score suggests mild depression. These feelings are real and worth addressing.")
    if score <= 14:  return ("Moderate",          "Your score suggests moderate depression. Speaking with a counselor is strongly recommended.")
    if score <= 19:  return ("Moderately Severe", "Your score suggests moderately severe depression. Please seek professional support soon.")
    return              ("Severe",           "Your score suggests severe depression. Please reach out for professional help immediately.")


def sa_get_gad7_severity(score):
    if score <= 4:  return ("Minimal",  "Your anxiety levels appear minimal right now.")
    if score <= 9:  return ("Mild",     "Your score suggests mild anxiety. Managing stress and rest can help.")
    if score <= 14: return ("Moderate", "Your score suggests moderate anxiety. Talking to someone can make a real difference.")
    return              ("Severe",  "Your score suggests severe anxiety. Professional support is strongly recommended.")


def sa_get_who5_interpretation(score):
    pct = score * 4
    if pct >= 72:  return "Your wellbeing score is good. You are showing positive signs of mental wellness."
    if pct >= 52:  return "Your wellbeing score is moderate. There may be room to improve your daily sense of ease and energy."
    if pct >= 28:  return "Your wellbeing score is low. This may suggest emotional fatigue or risk of depression — support is available."
    return             "Your wellbeing score is very low. This strongly suggests a need for professional mental health support."


def sa_build_result(phq_scores, gad_scores, who_scores):
    phq_total = sum(phq_scores.values())
    gad_total = sum(gad_scores.values())
    who_total = sum(who_scores.values())
    phq_q9   = phq_scores.get("phq9", 0)

    phq_sev = sa_get_phq9_severity(phq_total)[0]
    gad_sev = sa_get_gad7_severity(gad_total)[0]
    crisis_flag = phq_q9 >= 2

    # ── Score summary ─────────────────────────────────────
    lines = [
        "Thank you for completing this assessment. Here is a summary of your results:\n",
        f"PHQ-9 (Depression) — Score: {score_to_10(phq_total, 27)}/10 — {phq_sev}",
        f"GAD-7 (Anxiety)    — Score: {score_to_10(gad_total, 21)}/10 — {gad_sev}",
        f"WHO-5 (Wellbeing)  — Score: {score_to_10(who_total, 25)}/10",
    ]

    # ── Personalised guidance (same engine as main chat) ──
    advice, risk_level = generate_assessment_advice(phq_total, gad_total, who_total, phq_q9)
    if advice:
        lines.append(f"\n── Personal Guidance ──\n\n{advice}")

    # ── CTA to main chat ──────────────────────────────────
    if risk_level == "crisis" or crisis_flag:
        lines.append(
            "\nFor immediate, ongoing support please open the main chat — "
            "Empath AI will follow up on your results and guide you step by step. "
            "[Go to Main Chat →](/)"
        )
    elif risk_level in ("high", "moderate"):
        lines.append(
            "\nFor more personalised guidance and to keep track of your progress over time, "
            "visit the main chat where Empath AI can talk with you directly. "
            "[Go to Main Chat →](/)"
        )
    else:
        lines.append(
            "\nIf anything feels heavy or you want to talk through what you are feeling, "
            "the main chat is always open for you. "
            "[Go to Main Chat →](/)"
        )

    return "\n".join(lines), crisis_flag


def sa_init_session():
    session["sa_q_index"] = 0
    session["sa_phq"] = {}
    session["sa_gad"] = {}
    session["sa_who"] = {}
    session["sa_done"] = False
    session["sa_awaiting_name"] = session.get("user_id") is None
    session["sa_user_id"] = session.get("user_id")


def _sa_already_done_response(last_row, name=""):
    next_date = (
        datetime.fromisoformat(str(last_row["assessment_date"])) + timedelta(days=7)
    ).strftime("%B %d")
    greeting = f"Hi {name}. " if name else ""
    saved_advice = (last_row["advice_given"] or "").strip()
    saved_advice = (last_row["advice_given"] or "").strip()
    response_text = (
        f"{greeting}You have already completed your weekly assessment. "
        f"Your next one will be available on {next_date}.\n\n"
        f"Saved scores:\n"
        f"PHQ-9 (Depression): {score_to_10(last_row['phq9_total'], 27)}/10\n"
        f"GAD-7 (Anxiety): {score_to_10(last_row['gad7_total'], 21)}/10\n"
        f"WHO-5 (Wellbeing): {score_to_10(last_row['who5_total'], 25)}/10"
    )
    if saved_advice:
        response_text += f"\n\nSaved comments and guidance:\n{saved_advice}"
    response_text += "\n\nIf something feels urgent, please return to the main chat or call Umang: 0317-4288665."
    return jsonify({
        "response": response_text,
        "options": [],
        "type": "already_done",
    })


def _sa_start_questions(name=""):
    first_q = SA_QUESTIONS[0]
    greeting = f"Hello{', ' + name if name else ''}. "
    return jsonify({
        "response": (
            greeting +
            "I will ask you 21 short questions covering depression, anxiety, and wellbeing. "
            "There are no right or wrong answers — just answer as honestly as you can.\n\n" +
            first_q["text"]
        ),
        "options": PHQ_GAD_OPTIONS,
    })


@app.route("/assessment_chat_init", methods=["GET"])
def assessment_chat_init():
    sa_init_session()

    user_id = session.get("user_id")
    if user_id:
        last = get_last_assessment_for_user(user_id)
        if days_since_assessment(last) < 7:
            session["sa_done"] = True
            user = get_user_by_id(user_id)
            return _sa_already_done_response(last, user["name"] if user else "")
        user = get_user_by_id(user_id)
        session["sa_awaiting_name"] = False
        return _sa_start_questions(user["name"] if user else "")

    return jsonify({
        "response": "Welcome to your weekly self-assessment. Before we begin, what is your name?",
        "options": [],
    })


@app.route("/assessment_chat", methods=["POST"])
def assessment_chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "I did not catch that. Please choose an option or type a number.", "options": []})

    if session.get("sa_done"):
        return jsonify({"response": "You have already completed this assessment. Press Restart to begin again.", "options": []})

    # ── Name step ──────────────────────────────────────────────────
    if session.get("sa_awaiting_name"):
        raw_name = user_message.strip().split()[0].capitalize()
        if len(raw_name) < 2:
            return jsonify({"response": "Please enter your name to continue.", "options": []})

        existing = find_existing_user(raw_name)
        if existing:
            sa_uid = existing["id"]
            update_user_last_seen(sa_uid)
        else:
            sa_uid = create_user(raw_name)

        session["sa_user_id"] = sa_uid
        session["sa_awaiting_name"] = False

        last = get_last_assessment_for_user(sa_uid)
        if days_since_assessment(last) < 7:
            session["sa_done"] = True
            return _sa_already_done_response(last, raw_name)

        return _sa_start_questions(raw_name)

    # ── Question answering step ─────────────────────────────────────
    q_index = session.get("sa_q_index", 0)

    number_match = re.search(r"\b([0-4])\b", user_message)
    score = int(number_match.group(1)) if number_match else None

    if score is None:
        current_q = SA_QUESTIONS[q_index] if q_index < len(SA_QUESTIONS) else None
        options = WHO_OPTIONS if (current_q and current_q["scale"] == "who") else PHQ_GAD_OPTIONS
        return jsonify({"response": "Please choose one of the options below, or type a number.", "options": options})

    if q_index < len(SA_QUESTIONS):
        prev_q = SA_QUESTIONS[q_index]
        clamped = min(max(score, 0), 4 if prev_q["scale"] == "who" else 3)

        if prev_q["scale"] == "phq":
            phq = dict(session.get("sa_phq", {}))
            phq[prev_q["key"]] = clamped
            session["sa_phq"] = phq
            if prev_q["key"] == "phq9" and clamped >= 2:
                session["sa_q_index"] = q_index + 1
                return jsonify({
                    "response": (
                        "I want to pause here for a moment. Your answer suggests you may be going through something very heavy. "
                        "Please know this can be a phase, and life does not stay the same forever. "
                        "You deserve support right now — please call Umang: 0317-4288665, reach out to someone you trust, and come back only when you feel a little safer."
                    ),
                    "options": [],
                    "type": "crisis",
                })
        elif prev_q["scale"] == "gad":
            gad = dict(session.get("sa_gad", {}))
            gad[prev_q["key"]] = clamped
            session["sa_gad"] = gad
        elif prev_q["scale"] == "who":
            who = dict(session.get("sa_who", {}))
            who[prev_q["key"]] = clamped
            session["sa_who"] = who

        session["sa_q_index"] = q_index + 1
        q_index += 1

    # ── All questions answered ──────────────────────────────────────
    if q_index >= len(SA_QUESTIONS):
        session["sa_done"] = True
        phq_scores = session.get("sa_phq", {})
        gad_scores = session.get("sa_gad", {})
        who_scores = session.get("sa_who", {})
        result_text, crisis = sa_build_result(phq_scores, gad_scores, who_scores)

        sa_uid = session.get("sa_user_id") or session.get("user_id")
        phq_total = sum(phq_scores.values())
        gad_total = sum(gad_scores.values())
        who_total = sum(who_scores.values())
        phq_q9 = phq_scores.get("phq9", 0)
        prev_assess = get_last_assessment_for_user(sa_uid) if sa_uid else None
        if prev_assess and prev_assess["phq9_total"] is None:
            prev_assess = None
        advice_text, risk_level = generate_assessment_advice(phq_total, gad_total, who_total, phq_q9, prev_assess)

        session["last_advice"] = advice_text
        session["last_assessment_scores"] = {"phq9": phq_total, "gad7": gad_total, "who5": who_total}
        session["sa_followup_count"] = 0
        session["sa_followup_needed"] = True
        session["sa_listen_turns"] = 0
        if phq_total >= 20 or gad_total >= 18:
            session["sa_dominant_concern"] = "crisis_risk"
        elif phq_total >= 10:
            session["sa_dominant_concern"] = "depression_high"
        elif gad_total >= 10:
            session["sa_dominant_concern"] = "anxiety_high"
        elif who_total * 4 < 40:
            session["sa_dominant_concern"] = "wellbeing_low"
        elif phq_total >= 5 or gad_total >= 5:
            session["sa_dominant_concern"] = "mild"
        else:
            session["sa_dominant_concern"] = "minimal"

        if sa_uid:
            try:
                db = get_db()
                db.execute("""
                    INSERT INTO assessments (
                        user_id, phq9_total, gad7_total, who5_total,
                        severity, emotion_detected, displacement_cause, idp_hardships,
                        advice_given, prev_phq9, prev_gad7, prev_who5
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sa_uid,
                    phq_total,
                    gad_total,
                    who_total,
                    sa_get_phq9_severity(phq_total)[0],
                    session.get("emotion", "Unknown"),
                    session.get("displacement_cause", "Unknown"),
                    json.dumps([]),
                    advice_text,
                    prev_assess["phq9_total"] if prev_assess else None,
                    prev_assess["gad7_total"] if prev_assess else None,
                    prev_assess["who5_total"] if prev_assess else None,
                ))
                db.commit()
            except Exception as e:
                print("SA DB save error:", e)

        return jsonify({"response": result_text, "options": [], "type": "crisis" if (crisis or risk_level == 'crisis') else "result"})

    # ── Next question ───────────────────────────────────────────────
    next_q = SA_QUESTIONS[q_index]
    options = WHO_OPTIONS if next_q["scale"] == "who" else PHQ_GAD_OPTIONS
    return jsonify({"response": next_q["text"], "options": options})

@app.route("/assessment_chat_reset", methods=["POST"])
def assessment_chat_reset():
    sa_init_session()
    return jsonify({"status": "reset"})


@app.route("/wellness")
@login_required
def wellness():
    db = get_db()
    user_id = session.get("user_id")

    wellbeing_rows = db.execute("""
        SELECT * FROM wellbeing_logs
        WHERE user_id = ?
        ORDER BY log_date ASC
    """, (user_id,)).fetchall()

    assessment_rows_raw = db.execute("""
        SELECT * FROM assessments
        WHERE user_id = ?
        ORDER BY assessment_date ASC
    """, (user_id,)).fetchall()

    # ---- SUMMARY ----
    avg_sleep = 0
    sleep_vals = [float(r["sleep_hours"]) for r in wellbeing_rows if r["sleep_hours"] is not None]
    if sleep_vals:
        avg_sleep = round(sum(sleep_vals) / len(sleep_vals), 1)

    latest_mood = "-"
    latest_energy = "-"
    if wellbeing_rows:
        for r in reversed(wellbeing_rows):
            if latest_mood == "-" and r["mood"]:
                latest_mood = str(r["mood"]).replace("_", " ").title()
            if latest_energy == "-" and r["energy"]:
                latest_energy = str(r["energy"]).replace("_", " ").title()
            if latest_mood != "-" and latest_energy != "-":
                break

    # ---- CHART DATA ----
    sleep_labels = []
    sleep_values = []
    sleep_colors = []

    mood_labels = []
    mood_values = []

    energy_labels = []
    energy_values = []

    for r in wellbeing_rows:
        date = str(r["log_date"])[:10]

        if r["sleep_hours"] is not None:
            sleep_labels.append(date)
            sleep_values.append(float(r["sleep_hours"]))

            if float(r["sleep_hours"]) >= 7:
                sleep_colors.append("green")
            elif float(r["sleep_hours"]) >= 5:
                sleep_colors.append("orange")
            else:
                sleep_colors.append("red")

        if r["mood"]:
            mood_labels.append(date)
            mood_values.append({
                "very_low": 1,
                "low": 2,
                "anxious": 2,
                "okay": 3,
                "good": 4,
                "better": 4
            }.get(str(r["mood"]).lower(), 2))

        if r["energy"]:
            energy_labels.append(date)
            energy_values.append({
                "very_low": 1,
                "low": 2,
                "medium": 3,
                "high": 4
            }.get(str(r["energy"]).lower(), 2))

    # ---- ASSESSMENT CHART DATA ----
    assessment_labels = []
    phq = []
    gad = []
    who = []

    for r in assessment_rows_raw:
        assessment_labels.append(str(r["assessment_date"])[:10])
        phq.append(round((float(r["phq9_total"]) / 27) * 10, 1) if r["phq9_total"] is not None else 0)
        gad.append(round((float(r["gad7_total"]) / 21) * 10, 1) if r["gad7_total"] is not None else 0)
        who.append(round((float(r["who5_total"]) / 25) * 10, 1) if r["who5_total"] is not None else 0)

    # ---- TABLE DATA ----
    assessment_rows = []
    for r in assessment_rows_raw:
        assessment_rows.append({
            "display_date": str(r["assessment_date"])[:10],
            "phq10": round((float(r["phq9_total"]) / 27) * 10, 1) if r["phq9_total"] is not None else 0,
            "gad10": round((float(r["gad7_total"]) / 21) * 10, 1) if r["gad7_total"] is not None else 0,
            "who10": round((float(r["who5_total"]) / 25) * 10, 1) if r["who5_total"] is not None else 0,
            "severity": r["severity"] or "—",
            "emotion_detected": r["emotion_detected"] or "—"
        })

    return render_template(
        "wellness.html",
        summary={
            "avg_sleep": avg_sleep,
            "latest_mood": latest_mood,
            "latest_energy": latest_energy,
            "total_assessments": len(assessment_rows_raw)
        },
        chart_data={
            "sleep_labels": sleep_labels,
            "sleep_values": sleep_values,
            "sleep_colors": sleep_colors,

            "mood_labels": mood_labels,
            "mood_values": mood_values,
            "mood_colors": [
                "#d9534f" if v == 1 else
                "#f0ad4e" if v == 2 else
                "#5bc0de" if v == 3 else
                "#5cb85c"
                for v in mood_values
            ],

            "energy_labels": energy_labels,
            "energy_values": energy_values,
            "energy_colors": [
                "#d9534f" if v == 1 else
                "#f0ad4e" if v == 2 else
                "#5bc0de" if v == 3 else
                "#5cb85c"
                for v in energy_values
            ],

            "assessment_labels": assessment_labels,
            "phq_values": phq,
            "gad_values": gad,
            "who_values": who
        },
        assessment_rows=assessment_rows[::-1],
        has_data=bool(wellbeing_rows or assessment_rows_raw)
    )

# ─────────────────────────────────────────────
# QUOTES ROUTE — AI generated, IDP specific
# ─────────────────────────────────────────────
@app.route("/quotes")
@login_required
def quotes_page():
    return render_template("quotes.html")

@app.route("/api/quotes", methods=["GET"])
@login_required
def get_quotes():
    categories = ["Hope", "Resilience", "Family", "Patience", "Strength", "Home"]
    import random
    chosen = random.sample(categories, 3)

    system_prompt = """You generate short, soft, comforting quotes for internally displaced people (IDPs) in Pakistan who have lost their homes due to floods, conflict, or earthquakes. These people are going through grief, trauma, displacement, and loss.

RULES:
- Each quote must feel warm, gentle, and human — NOT clinical or motivational-speaker style
- Mix sources: Quran verses, Hadith of Prophet Muhammad (PBUH), Quaid-e-Azam Muhammad Ali Jinnah, Allama Iqbal, and famous international figures (Maya Angelou, Nelson Mandela, Victor Hugo etc.)
- Quran and Hadith quotes should be the actual meaning in simple English — not overly formal
- Do NOT use hard or dramatic language like "conquer", "warrior", "battle", "fight"
- Keep each quote under 30 words
- Return ONLY valid JSON, no markdown, no extra text

Return exactly this JSON structure:
[
  {
    "quote": "...",
    "person": "...",
    "title": "...",
    "category": "Hope",
    "type": "international"
  }
]

type must be one of: "quran", "hadith", "quaid", "iqbal", "international"
"""

    user_prompt = f"Generate exactly 3 quotes, one for each of these categories: {', '.join(chosen)}. Return only the JSON array."

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_prompt,
            input=[{"role": "user", "content": user_prompt}],
            max_output_tokens=600,
        )
        raw = response.output_text.strip()
        # strip markdown fences if any
        raw = raw.replace("```json", "").replace("```", "").strip()
        quotes = json.loads(raw)
        return jsonify({"quotes": quotes, "status": "ok"})
    except Exception as e:
        print(f"Quotes API error: {e}")
        # fallback quotes
        fallback = [
            {"quote": "Verily, with every hardship comes ease.", "person": "The Holy Quran", "title": "Surah Al-Inshirah 94:6", "category": "Hope", "type": "quran"},
            {"quote": "You never know how strong you are until being strong is your only choice.", "person": "Bob Marley", "title": "Musician & Poet", "category": "Resilience", "type": "international"},
            {"quote": "With faith, discipline and selfless devotion to duty, there is nothing worthwhile you cannot achieve.", "person": "Quaid-e-Azam M.A. Jinnah", "title": "Founder of Pakistan", "category": "Strength", "type": "quaid"},
        ]
        return jsonify({"quotes": fallback, "status": "fallback"})
        
# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True, port=5000)
    
    
