import os
import re
import pickle
import json
import random
from datetime import datetime

from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "mental_health_fyp_2026"

# OpenAI client uses OPENAI_API_KEY from environment
client = OpenAI()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")

print("Loading models...")

with open(os.path.join(DATA_PATH, "crisis_detector.pkl"), "rb") as f:
    crisis_data = pickle.load(f)

with open(os.path.join(DATA_PATH, "emotion_detector_model.pkl"), "rb") as f:
    emotion_model = pickle.load(f)

with open(os.path.join(DATA_PATH, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(DATA_PATH, "psychosis_detector.pkl"), "rb") as f:
    psychosis_data = pickle.load(f)

with open(os.path.join(DATA_PATH, "guardrail.pkl"), "rb") as f:
    guardrail_data = pickle.load(f)

# Optional templates fallback
TEMPLATES = {}
templates_path = os.path.join(DATA_PATH, "response_templates.pkl")
if os.path.exists(templates_path):
    with open(templates_path, "rb") as f:
        TEMPLATES = pickle.load(f)

CRISIS_PATTERNS = crisis_data["patterns"]
FINALITY_WORDS = crisis_data["finality_words"]
ACTION_WORDS = crisis_data["action_words"]
RELEASE_WORDS = crisis_data["release_words"]

PSYCHOSIS_PATTERNS = psychosis_data["patterns"]
PSYCHOSIS_RESPONSES = psychosis_data["responses"]

BLOCKED_TOPICS = guardrail_data["blacklist_combos"]
FRAMING_PATTERNS = guardrail_data["framing_patterns"]
JAILBREAK_PATTERNS = guardrail_data["jailbreak_patterns"]
ESCALATION_PATTERNS = guardrail_data["escalation_patterns"]
SYNONYM_PATTERNS = guardrail_data["synonym_patterns"]
MANIPULATION_PATTERNS = guardrail_data["manipulation_patterns"]

print("All models loaded!")

# ─────────────────────────────────────────────
# STAGES
# ─────────────────────────────────────────────
STAGE_NAME = "name"
STAGE_ROLE = "role"
STAGE_LISTEN = "listen"
STAGE_ASSESS = "assess"
STAGE_RESPOND = "respond"

DISTRESS_EMOTIONS = [
    "depression", "anxiety", "ptsd", "OCD",
    "emotional_support", "stress", "grief", "anger", "fear", "sadness"
]

ROLE_KEYWORDS = {
    "student": [
        "student", "university", "college", "school", "studies",
        "study", "exam", "education", "semester", "notes"
    ],
    "homemaker": [
        "house wife", "housewife", "home maker", "homemaker",
        "mother", "mom", "maa", "children", "kids", "baby", "bacha"
    ],
    "daily_worker": [
        "worker", "labor", "labour", "construction", "daily wage",
        "mazdoor", "earner", "jobless"
    ],
    "elderly": [
        "retired", "old", "elder", "grandfather", "grandmother",
        "haji", "buzurg", "old man", "old woman"
    ],
    "government": [
        "government", "employee", "office", "job", "officer"
    ],
}

GRIEF_INTENT_PATTERNS = [
    r"(should|going to|want to|will).{0,10}(go|join).{0,15}(them|him|her|dead|died|passed)",
    r"she is asking me to come",
    r"he is asking me to come",
    r"(dead|died|passed).{0,20}(asking|calling).{0,15}(come|join|go)",
    r"i should go.{0,15}(join|him|her|them)",
]

NEGATIVE_HINTS = [
    "sad", "bad", "upset", "alone", "tired", "hopeless", "crying",
    "depressed", "anxious", "stress", "stressed", "worried",
    "not good", "not fine", "not okay", "not well", "hurt",
    "displaced", "violence", "lost my home", "camp", "failing",
    "worthless", "ashamed", "no point", "cant provide", "can't provide",
    "hungry", "hunger", "exhaust", "exhausted"
]

QURAN_HOPE_VERSES = [
    {
        "ref": "Quran 39:53",
        "topic": "hope",
        "arabic": "لَا تَقْنَطُوا مِن رَّحْمَةِ اللَّهِ",
        "english": "Do not lose hope in Allah’s mercy."
    },
    {
        "ref": "Quran 2:286",
        "topic": "burden",
        "arabic": "لَا يُكَلِّفُ ٱللَّهُ نَفْسًا إِلَّا وُسْعَهَا",
        "english": "Allah does not burden a soul beyond what it can bear."
    },
    {
        "ref": "Quran 94:5-6",
        "topic": "hardship",
        "arabic": "فَإِنَّ مَعَ ٱلْعُسْرِ يُسْرًا ۝ إِنَّ مَعَ ٱلْعُسْرِ يُسْرًا",
        "english": "Surely, with hardship comes ease."
    },
]

PSYCHOSIS_SUPPORT_VERSES = [
    {
        "ref": "Quran 13:28",
        "topic": "calm",
        "arabic": "أَلَا بِذِكْرِ اللَّهِ تَطْمَئِنُّ الْقُلُوبُ",
        "english": "Surely in the remembrance of Allah do hearts find comfort."
    },
    {
        "ref": "Quran 17:82",
        "topic": "healing",
        "arabic": "وَنُنَزِّلُ مِنَ ٱلْقُرْءَانِ مَا هُوَ شِفَآءٞ وَرَحْمَةٞ لِّلْمُؤْمِنِينَ",
        "english": "We send down the Quran as a healing and mercy for the believers."
    },
    {
        "ref": "Quran 2:255",
        "topic": "protection",
        "arabic": "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَيُّ ٱلْقَيُّومُ",
        "english": "Allah! There is no god worthy of worship except Him, the Ever-Living, All-Sustaining."
    }
]

SIGNATURE_LINES = [
    "Stay with me for a moment.",
    "Let’s hold this gently.",
    "I’m listening with care.",
    "You do not have to carry all of this at once.",
    "Let’s keep this moment simple and safe.",
    "We can take this one step at a time."
]

# ─────────────────────────────────────────────
# SESSION / HELPERS
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
    session["daily_checkins"] = []


def add_to_history(role, content):
    history = list(session.get("history", []))
    history.append({"role": role, "content": content})
    session["history"] = history[-12:]


def detect_role(text):
    text = text.lower().strip()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return role
    return "default"


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
        "suicidal": "suicide",
        "commmit suicide": "commit suicide",
        "kill my self": "kill myself",
        "end my self": "end myself",
        "dont want to live": "don't want to live",
        "dont wanna live": "don't want to live",
        "wanna die": "want to die",
        "wan to die": "want to die",
        "diee": "die",
        "myslef": "myself",
        "kll myself": "kill myself",
        "kil myself": "kill myself",
        "sui cide": "suicide",
        "sui-side": "suicide",
        "freind": "friend",
        "frend": "friend",
        "freinds": "friends",
        "frends": "friends",
        "dreem": "dream",
        "dreams": "dream",
        "cant": "can't",
        "wont": "won't",
        "dont": "don't",
    }

    for wrong, correct in replacements.items():
        t = t.replace(wrong, correct)

    t = normalize_text(t)
    return t


def has_grief_intent(text):
    t = text.lower()
    return any(re.search(p, t) for p in GRIEF_INTENT_PATTERNS)


def is_positive_message(text):
    t = clean_text(text)

    negative_patterns = [
        r"\bnot\s+good\b",
        r"\bnot\s+fine\b",
        r"\bnot\s+okay\b",
        r"\bnot\s+well\b",
        r"\bnot\s+better\b",
        r"\bi am sad\b",
        r"\bi feel sad\b",
        r"\bi am not okay\b",
        r"\bi am not fine\b",
        r"\bi am not good\b",
        r"\bbad day\b",
        r"\bfeel bad\b",
        r"\bi am depressed\b",
        r"\bi feel alone\b",
        r"\bno point\b",
        r"\bworthless\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, t):
            return False

    positive_patterns = [
        r"\bi am happy\b",
        r"\bi feel good\b",
        r"\bi am good\b",
        r"\bi am fine\b",
        r"\bi am okay\b",
        r"\bi feel better\b",
        r"\bi am doing well\b",
        r"\bi feel great\b",
        r"\bi am grateful\b",
        r"\bi am alright\b",
    ]
    for pattern in positive_patterns:
        if re.search(pattern, t):
            return True

    return False


def wants_assessment(text):
    t = text.lower().strip()
    triggers = [
        "start assessment",
        "start screening",
        "mental health assessment",
        "check my mental health",
        "screen me",
        "ask me questions",
        "start phq",
        "start gad",
        "yes assessment",
        "yes screening"
    ]
    return any(trigger in t for trigger in triggers)


def looks_like_name(text):
    t = text.strip()
    if len(t.split()) > 3:
        return False

    lower_t = normalize_for_safety(t)
    bad_name_signals = [
        "kill", "die", "suicide", "sad", "depressed", "help",
        "anxious", "alone", "lost", "home", "stress", "cry",
        "hopeless", "not okay", "hurt", "dream", "calling me",
        "voices", "blood"
    ]

    if any(word in lower_t for word in bad_name_signals):
        return False

    return bool(re.fullmatch(r"[A-Za-z ]{2,30}", t))


def direct_crisis_override(text):
    t = normalize_for_safety(text)

    high_risk_phrases = [
        "i want to die",
        "i want to kill myself",
        "kill myself",
        "commit suicide",
        "suicide",
        "end my life",
        "i don't want to live",
        "i do not want to live",
        "i want to end it",
        "i want to disappear forever",
        "i want to kill my self",
        "i will kill myself",
        "i am going to kill myself",
        "i should kill myself",
    ]

    return any(phrase in t for phrase in high_risk_phrases)


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

        all_patterns = (
            FRAMING_PATTERNS
            + JAILBREAK_PATTERNS
            + ESCALATION_PATTERNS
            + SYNONYM_PATTERNS
            + MANIPULATION_PATTERNS
        )
        for pattern in all_patterns:
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
    crisis = (
        scores.get("crisis", 0)
        + scores.get("grief_crisis", 0)
        + scores.get("metaphor_crisis", 0)
        + scores.get("farewell", 0)
        + bonus
    )
    self_harm = scores.get("self_harm", 0)
    distress = scores.get("distress", 0)

    enjoyment_words = ["feel good", "feels good", "like", "enjoy", "love", "crave", "alive", "relief", "calm", "real", "better"]
    is_enjoyment = any(w in t for w in enjoyment_words)

    if emergency >= 10:
        return "emergency"
    if self_harm >= 7 and not is_enjoyment:
        return "emergency"
    if self_harm >= 4 and is_enjoyment:
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
        print("User text:", text)
        print("Cleaned text:", cleaned)
        print("Predicted emotion:", emotion)
        print("Confidence:", conf)
        return emotion, conf
    except Exception as e:
        print("Emotion detection error:", e)
        return "emotional_support", 50.0


def detect_psychosis(text):
    t = text.lower().strip()
    detected = {}

    for category, patterns in PSYCHOSIS_PATTERNS.items():
        score = 0
        for pattern, weight in patterns:
            if re.search(pattern, t):
                score += weight
        if score > 0:
            detected[category] = score

    if not detected:
        return {"psychosis_detected": False}

    top_cat = max(detected, key=detected.get)
    top_score = detected[top_cat]
    level = "emergency" if top_cat == "command_hallucination" or top_score >= 10 else "crisis"

    return {
        "psychosis_detected": True,
        "category": top_cat,
        "level": level,
        "response": PSYCHOSIS_RESPONSES.get(
            top_cat,
            PSYCHOSIS_RESPONSES.get(
                "hallucination",
                ["I am concerned about what you are experiencing. Please tell a nearby trusted person or health worker right now."]
            )
        )[0],
    }


def detect_dream_or_hallucination_risk(text):
    t = normalize_for_safety(text)

    dream_patterns = [
        r"\b(friend|mother|father|sister|brother|wife|husband|child|someone)\b.*\bdream\b.*\bcalling me\b",
        r"\bi saw .* in my dream\b",
        r"\bsomeone came in my dream\b",
        r"\bmy friend came in my dream\b",
        r"\bmy friend is calling me\b",
        r"\bmy dead .* is calling me\b",
        r"\bshe is calling me\b",
        r"\bhe is calling me\b",
        r"\bblood\b.*\bdream\b",
        r"\bdream\b.*\bblood\b",
    ]

    awake_command_patterns = [
        r"\bi hear .* calling me\b",
        r"\bvoices? .* talking to me\b",
        r"\bvoices? .* calling me\b",
        r"\bshe is telling me to come\b",
        r"\bhe is telling me to come\b",
        r"\bthey are telling me to come\b",
        r"\basking me to come\b",
        r"\btelling me to join\b",
        r"\bcalling me to come\b",
        r"\bi should go with her\b",
        r"\bi should join her\b",
        r"\bsomeone standing in my room\b",
        r"\bi cant see them\b",
        r"\bi can't see them\b",
        r"\bno one was there\b",
    ]

    dream_hit = any(re.search(p, t) for p in dream_patterns)
    awake_hit = any(re.search(p, t) for p in awake_command_patterns)

    if awake_hit:
        return {
            "detected": True,
            "type": "psychosis_or_command",
            "level": "crisis",
            "response": "psychosis"
        }

    if dream_hit:
        return {
            "detected": True,
            "type": "dream_grief",
            "level": "support",
            "response": (
                "That sounds emotional and unsettling, and I’m glad you shared it with me. "
                "Sometimes dreams like this can come when someone is carrying grief, fear, or a lot of stress. "
                "Did this happen only in a dream, or does it feel like it is happening while you are awake too?"
            )
        }

    return {"detected": False}


def select_quran_verse(user_text):
    text = normalize_for_safety(user_text)
    used = session.get("used_verses", [])

    if any(word in text for word in ["suicide", "kill myself", "want to die", "end my life", "hopeless", "no point"]):
        preferred_topic = "hope"
    elif any(word in text for word in ["burden", "too much", "can't handle", "cant handle", "exhausted", "tired"]):
        preferred_topic = "burden"
    else:
        preferred_topic = "hardship"

    candidates = [v for v in QURAN_HOPE_VERSES if v["topic"] == preferred_topic and v["ref"] not in used]

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

    if any(w in text for w in ["voices", "talking to me", "standing in my room", "night"]):
        preferred = "protection"
    elif any(w in text for w in ["scared", "fear", "frightening", "unsettling", "panic"]):
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


def build_crisis_response(user_text, name=""):
    verse = select_quran_verse(user_text)

    first_part = (
        f"I’m really sorry you’re feeling this much pain{', ' + name if name else ''}… "
        f"I’m really glad you reached out to me 🤍\n"
        f"You matter, and your life is important. Right now, the most important thing is keeping you safe.\n"
        f"Are you alone, or is there someone near you who can sit with you?\n"
        f"Please call Umang (0317-4288665) or 1122 right now — they can help you immediately."
    )

    second_part = (
        f"\n\n{verse['ref']}\n"
        f"{verse['arabic']}\n"
        f"{verse['english']}"
    )

    return first_part + second_part


def build_psychosis_support_response(user_text, name=""):
    verse = select_psychosis_verse(user_text)
    text = normalize_for_safety(user_text)
    soft_line = random.choice(SIGNATURE_LINES)

    if any(w in text for w in ["voices", "talking to me"]):
        main = (
            f"{name}, nights can feel very intense when your mind is already carrying fear or shock. "
            f"What you’re describing sounds frightening, and I’m really glad you said it out loud. "
            f"{soft_line} Are you safe right now, and is there someone nearby who can sit with you?"
        )
    elif any(w in text for w in ["standing in my room", "no one was there", "cant see them", "can't see them"]):
        main = (
            f"{name}, that would shake anyone, and I’m sorry you had to feel that fear. "
            f"Sometimes when stress, grief, or shock build up, the mind can make a moment feel very real and very frightening. "
            f"{soft_line} Did this happen just once, or has it happened more than once?"
        )
    else:
        main = (
            f"{name}, that sounds confusing and heavy to carry alone. "
            f"You do not need to figure it all out by yourself right now. "
            f"{soft_line} Are you alone right now, or is there someone nearby you trust?"
        )

    ayah = f"\n\n{verse['ref']}\n{verse['arabic']}\n{verse['english']}"
    return main + ayah


def run_safety_checks(user_text):
    normalized_text = normalize_for_safety(user_text)

    if direct_crisis_override(normalized_text):
        return {"type": "emergency", "level": "emergency"}

    guardrail = check_guardrail(normalized_text)
    if guardrail["blocked"]:
        return {"type": "blocked", "response": guardrail["response"]}

    if has_grief_intent(normalized_text):
        return {"type": "crisis", "level": "crisis"}

    dream_risk = detect_dream_or_hallucination_risk(normalized_text)
    if dream_risk["detected"]:
        if dream_risk["level"] == "crisis":
            return {
                "type": "psychosis",
                "level": "crisis",
                "response": dream_risk["response"]
            }
        return {
            "type": "special_support",
            "level": "support",
            "response": dream_risk["response"]
        }

    psychosis = detect_psychosis(normalized_text)
    if psychosis["psychosis_detected"]:
        return {
            "type": "psychosis",
            "level": psychosis["level"],
            "response": psychosis["response"]
        }

    crisis_level = detect_crisis(normalized_text)
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


def get_template_response(context):
    if not TEMPLATES:
        name = context.get("name", "")
        return (
            f"{name}, I hear you. That sounds difficult. Can you tell me a little more about what has been happening?"
            if name else
            "I hear you. That sounds difficult. Can you tell me a little more about what has been happening?"
        )

    emotion = context.get("emotion", "emotional_support")
    severity = context.get("severity", "Mild")
    crisis = context.get("crisis_level", "safe")
    role = context.get("role", "default")
    name = context.get("name", "")

    try:
        if crisis == "emergency":
            response = TEMPLATES["emergency"][0]
        elif crisis == "crisis":
            response = TEMPLATES["crisis"][0]
        else:
            emotion_templates = TEMPLATES.get(emotion, TEMPLATES["emotional_support"])
            if isinstance(emotion_templates, dict):
                severity_templates = emotion_templates.get(
                    severity,
                    emotion_templates.get("default", TEMPLATES["emotional_support"]["default"])
                )
            else:
                severity_templates = emotion_templates

            response = severity_templates[0]

            if role != "default" and "roles" in TEMPLATES:
                role_responses = TEMPLATES["roles"].get(role)
                if role_responses:
                    response = role_responses[0] + " " + response

        if name and not response.lower().startswith(name.lower()):
            response = f"{name}, {response[0].lower() + response[1:]}"
        return response
    except Exception as e:
        print("Template response error:", e)
        return (
            f"{name}, I hear you. That sounds difficult. Can you tell me a little more about what has been happening?"
            if name else
            "I hear you. That sounds difficult. Can you tell me a little more about what has been happening?"
        )


def get_openai_response(user_message, sess, crisis_level="safe", emotion="unknown"):
    try:
        name = sess.get("name", "")
        role = sess.get("role", "default")
        severity = sess.get("severity", "unknown")
        history = list(sess.get("history", []))

        system_prompt = f"""
You are Empath AI, a warm and emotionally intelligent support chatbot for internally displaced adults in Pakistan.

Known context:
- Name: {name if name else "not provided"}
- Role/background: {role}
- Emotion detected: {emotion}
- Severity: {severity}
- Crisis level: {crisis_level}

Voice and identity:
- You are not a generic assistant.
- You speak like a calm, caring companion sitting beside the person.
- You sound grounded, gentle, observant, and human.
- You avoid generic chatbot phrases like "that sounds difficult" and "would you like to share more".
- You sometimes use original comforting lines like:
  * Stay with me for a moment.
  * Let’s hold this gently.
  * I’m listening with care.
  * You do not have to carry all of this at once.
  * Let’s keep this moment simple and safe.
- You can make small emotional observations such as:
  * Sometimes grief returns through dreams before it returns through words.
  * When the mind is tired, night can feel louder than day.
  * Shock can stay in the mind longer than people expect.
  * Displacement can disturb not only home, but also rest, focus, and safety.
- You should feel distinct from common chatbots.

Rules:
1. Maximum 4 short sentences
2. Always begin with empathy or emotional reflection
3. Then give one small supportive grounding thought
4. Then ask one natural follow-up question
5. Do not sound like a therapist, survey, or customer support bot
6. Do not start PHQ-9 or GAD-7 unless the user directly asks for assessment
7. Never ask the same question again if the user already answered it
8. If role is homemaker, understand children, burden, guilt, hunger, shelter stress
9. If role is daily_worker, understand provider pressure, shame, work loss
10. If role is student, understand study disruption, guilt, future anxiety
11. If role is elderly, understand grief, home loss, silence, identity loss
12. If crisis level is crisis, gently mention Umang helpline: 0317-4288665
13. Never provide self-harm methods
14. Avoid saying "everything will be okay"
15. English only
16. Sound original, soft, and emotionally aware
"""

        messages = []
        for item in history[-8:]:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append(item)

        messages.append({"role": "user", "content": user_message})

        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_prompt,
            input=messages,
            max_output_tokens=180,
        )

        return response.output_text.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return None


def get_crisis_followup_response(name, user_message):
    text = normalize_for_safety(user_message)

    if any(k in text for k in ["i am alone", "alone", "nobody", "no one", "no one is here"]):
        return (
            f"{name}, I’m really glad you told me. "
            f"Being alone can make this feel even heavier. "
            f"Is there anyone nearby in your home, NGO, camp, building, or neighborhood who can sit with you right now?"
        )

    if any(k in text for k in ["dont want to tell any body", "don't want to tell anybody", "dont want to tell anybody", "i dont want to tell anyone", "i don't want to tell anyone"]):
        return (
            f"{name}, I understand that telling someone can feel very hard right now. "
            f"You do not need to explain everything — having one person physically near you can still help keep you safe. "
            f"Would you be able to call Umang or ask one person to stay with you for a little while?"
        )

    if any(k in text for k in ["lost my family", "only one survive", "only one survived", "survive the flood", "survived the flood", "family died", "my family died"]):
        return (
            f"{name}, what you survived sounds deeply painful, and I’m really sorry. "
            f"You matter, and I want to focus on your safety right now. "
            f"Is there a trusted person or health worker nearby who can stay with you tonight?"
        )

    if any(k in text for k in ["i want to die", "kill myself", "commit suicide", "suicide", "end my life", "don't want to live"]):
        return build_crisis_response(user_message, name)

    return (
        f"{name}, I’m really glad you’re still here with me. "
        f"What you’re carrying sounds overwhelming, and your safety matters right now. "
        f"Is there someone physically nearby who can stay with you?"
    )


def get_supportive_reply(name, role, user_message, crisis_level, emotion):
    api_response = get_openai_response(user_message, session, crisis_level, emotion)
    if api_response:
        return api_response

    line = random.choice(SIGNATURE_LINES)
    return (
        f"{name}, I’m listening with care. "
        f"This sounds heavier than one small bad moment. "
        f"{line} What feels hardest right now?"
    )


def save_assessment(session_data):
    try:
        record = {
            "resourceType": "MentalHealthAssessment",
            "patient_name": session_data.get("name", "Unknown"),
            "role": session_data.get("role", "Unknown"),
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
        print(f"Save error: {e}")


def ask_first_assessment_question():
    session["stage"] = STAGE_ASSESS
    session["q_index"] = 0
    session["assessment_started"] = True
    return (
        "I want to understand you a little better so I can support you well. "
        "Over the past two weeks, how often have you felt down or hopeless? "
        "(0=not at all, 1=several days, 2=more than half the days, 3=nearly every day)"
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/init", methods=["GET"])
def init():
    session.clear()
    init_session()
    welcome = "Hello. I am here to listen and support you. What is your name?"
    add_to_history("assistant", welcome)
    return jsonify({
        "response": welcome,
        "stage": session.get("stage")
    })


@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"status": "reset"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "I did not catch that. Could you say that again?"})

    if "stage" not in session:
        init_session()

    stage = session["stage"]
    add_to_history("user", user_message)

    safety = run_safety_checks(user_message)
    name = session.get("name", "")

    if safety["type"] in ["emergency", "crisis"]:
        session["crisis_active"] = True
        session["crisis_level"] = safety.get("level", "crisis")

    if safety["type"] == "blocked":
        response = safety["response"]
        add_to_history("assistant", response)
        return jsonify({"response": response, "type": "blocked"})

    if safety["type"] == "psychosis":
        response = build_psychosis_support_response(user_message, session.get("name", ""))
        add_to_history("assistant", response)
        return jsonify({"response": response, "type": "psychosis"})

    if safety["type"] == "special_support":
        response = safety["response"]
        add_to_history("assistant", response)
        return jsonify({"response": response, "type": "special_support"})

    if safety["type"] == "emergency":
        session["crisis_active"] = True
        session["crisis_level"] = "emergency"
        response = build_crisis_response(user_message, session.get("name", ""))
        add_to_history("assistant", response)
        return jsonify({"response": response, "type": "emergency"})

    if safety["type"] == "crisis":
        session["crisis_active"] = True
        session["crisis_level"] = "crisis"
        response = build_crisis_response(user_message, session.get("name", ""))
        add_to_history("assistant", response)
        return jsonify({"response": response, "type": "crisis"})

    crisis_level = session.get("crisis_level", safety.get("level", "safe"))
    emotion, conf = detect_emotion(user_message)
    session["emotion"] = emotion
    session["confidence"] = conf

    msg_lower = user_message.lower()
    is_pos = is_positive_message(user_message)
    session["is_positive"] = is_pos

    is_distressed = (
        emotion in DISTRESS_EMOTIONS
        or any(hint in msg_lower for hint in NEGATIVE_HINTS)
    ) and not is_pos

    if session.get("crisis_active", False):
        response = get_crisis_followup_response(session.get("name", ""), user_message)
        add_to_history("assistant", response)
        return jsonify({
            "response": response,
            "type": "crisis",
            "stage": session.get("stage"),
        })

    response = ""

    if stage == STAGE_NAME:
        if looks_like_name(user_message):
            name = user_message.split()[0].capitalize()
            session["name"] = name
            response = (
                f"Nice to meet you, {name}. "
                f"Can you tell me a little about yourself — are you a student, working, a homemaker, or something else?"
            )
            session["stage"] = STAGE_ROLE
        else:
            session["stage"] = STAGE_LISTEN
            response = "I’m here with you. Before anything else, would you like to tell me what has been weighing on you lately?"

    elif stage == STAGE_ROLE:
        role = detect_role(user_message)
        session["role"] = role
        name = session.get("name", "")
        response = f"Thank you, {name}. I am here to listen. How have you been feeling lately?"
        session["stage"] = STAGE_LISTEN

    elif stage == STAGE_LISTEN:
        name = session.get("name", "")
        role = session.get("role", "default")

        if wants_assessment(user_message):
            session["stage"] = STAGE_ASSESS
            session["q_index"] = 0
            session["assessment_started"] = False
            response = ask_first_assessment_question()
            add_to_history("assistant", response)
            return jsonify({
                "response": response,
                "type": "normal",
                "stage": session.get("stage"),
            })

        response = get_supportive_reply(name, role, user_message, crisis_level, emotion)
        session["stage"] = STAGE_LISTEN

    elif stage == STAGE_ASSESS:
        name = session.get("name", "")
        questions = [
            ("phq3", "Over the past two weeks, how often have you felt down or hopeless? (0=not at all, 1=several days, 2=more than half the days, 3=nearly every day)"),
            ("phq5", "How has your sleep been lately? (0-3)"),
            ("phq4", "How is your energy day to day? (0-3)"),
            ("phq7", "Have you been finding it hard to focus on things? (0-3)"),
            ("phq9", "Sometimes when people go through hard times, thoughts of not wanting to be here can come up. Has anything like that come up for you? (0=not at all, 1=several days, 2=more than half the days, 3=nearly every day)"),
            ("gad1", "How much has worry been affecting you lately? (0-3)"),
            ("gad2", "Do you find the worry hard to control or stop? (0-3)"),
            ("who1", "Have there been any moments recently where you felt okay or even a little better? (0=never, 4=most of the time)"),
            ("who5", "Is there anything in your life right now that still interests or engages you even a little? (0-4)"),
        ]

        if not session.get("assessment_started", False):
            response = ask_first_assessment_question()
            add_to_history("assistant", response)
            return jsonify({
                "response": response,
                "type": "normal",
                "stage": session.get("stage"),
            })

        q_index = session.get("q_index", 0)

        if q_index > 0:
            prev_key = questions[q_index - 1][0]
            number_match = re.search(r"\b([0-4])\b", user_message)

            if number_match:
                score = int(number_match.group(1))
                score = min(max(score, 0), 4)

                if prev_key.startswith("phq"):
                    phq = dict(session.get("phq", {}))
                    phq[prev_key] = score
                    session["phq"] = phq
                elif prev_key.startswith("gad"):
                    gad = dict(session.get("gad", {}))
                    gad[prev_key] = score
                    session["gad"] = gad
                elif prev_key.startswith("who"):
                    who = dict(session.get("who", {}))
                    who[prev_key] = score
                    session["who"] = who

                if prev_key == "phq9" and score >= 2:
                    session["crisis_active"] = True
                    session["crisis_level"] = "crisis"
                    response = build_crisis_response(user_message, name)
                    add_to_history("assistant", response)
                    return jsonify({"response": response, "type": "crisis"})

        if q_index < len(questions):
            _, question = questions[q_index]
            response = question
            session["q_index"] = q_index + 1
        else:
            phq_score = sum(session.get("phq", {}).values())
            severity = get_severity(phq_score)
            session["severity"] = severity
            session["stage"] = STAGE_RESPOND

            if severity in ["Moderate", "Severe"]:
                response = (
                    f"Thank you for sharing all of that, {name}. "
                    f"It sounds like you have been carrying a lot for some time. "
                    f"I would encourage you to speak with a mental health professional, and Umang: 0317-4288665 can help."
                )
            elif severity == "Mild":
                response = (
                    f"Thank you for sharing, {name}. "
                    f"It sounds like things have been somewhat difficult lately. "
                    f"Talking about it is an important step."
                )
            else:
                response = (
                    f"Thank you for sharing, {name}. "
                    f"I am glad you reached out and answered those questions."
                )

            save_assessment(session)

    elif stage == STAGE_RESPOND:
        name = session.get("name", "")
        role = session.get("role", "default")

        if wants_assessment(user_message):
            session["stage"] = STAGE_ASSESS
            session["q_index"] = 0
            session["assessment_started"] = False
            response = ask_first_assessment_question()
            add_to_history("assistant", response)
            return jsonify({
                "response": response,
                "type": "normal",
                "stage": session.get("stage"),
            })

        response = get_supportive_reply(name, role, user_message, crisis_level, emotion)

    else:
        response = "I am here with you. Can you tell me a little more?"

    add_to_history("assistant", response)

    return jsonify({
        "response": response,
        "type": "normal",
        "stage": session.get("stage"),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)