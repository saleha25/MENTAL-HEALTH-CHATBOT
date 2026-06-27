"""
build_retrieval.py
Run this once to build the retrieval system from training data
"""

import os
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Set working directory
BASE_PATH = r"D:\Empath AI"
os.chdir(BASE_PATH)

print("Loading data...")

# Load datasets
try:
    cc = pd.read_csv("data/external/counsel_chat.csv")
    print(f"Loaded counsel_chat: {len(cc)} rows")
except Exception as e:
    print(f"Warning: counsel_chat.csv not found or error: {e}")
    cc = pd.DataFrame()

try:
    mh = pd.read_csv("data/external/mental_health_counseling.csv")
    print(f"Loaded mental_health_counseling: {len(mh)} rows")
except Exception as e:
    print(f"Warning: mental_health_counseling.csv not found or error: {e}")
    mh = pd.DataFrame()

try:
    mc = pd.read_csv("data/external/mental_chat_16k.csv")
    print(f"Loaded mental_chat_16k: {len(mc)} rows")
except Exception as e:
    print(f"Warning: mental_chat_16k.csv not found or error: {e}")
    mc = pd.DataFrame()

# Load vectorizer
try:
    with open("data/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    print("Loaded TF-IDF vectorizer")
except FileNotFoundError:
    print("ERROR: tfidf_vectorizer.pkl not found. Run training first!")
    exit(1)

# Load crisis detector
try:
    with open("data/crisis_detector.pkl", "rb") as f:
        crisis_data = pickle.load(f)
    CRISIS_PATTERNS = crisis_data["patterns"]
    FINALITY_WORDS  = crisis_data["finality_words"]
    ACTION_WORDS    = crisis_data["action_words"]
    RELEASE_WORDS   = crisis_data["release_words"]
    print("Loaded crisis detector")
except FileNotFoundError:
    print("ERROR: crisis_detector.pkl not found. Run training first!")
    exit(1)

# ─────────────────────────────────────────────
# CRISIS DETECTION FUNCTION
# ─────────────────────────────────────────────
def detect_crisis(text):
    t = text.lower().strip()
    scores = {k: 0 for k in CRISIS_PATTERNS}
    for level, patterns in CRISIS_PATTERNS.items():
        for pattern, weight in patterns:
            if re.search(pattern, t):
                scores[level] += weight
    
    finality  = sum(3 for w in FINALITY_WORDS if w in t)
    action    = sum(3 for w in ACTION_WORDS   if w in t)
    release   = sum(4 for w in RELEASE_WORDS  if w in t)
    bonus     = finality + action + release
    
    emergency = scores.get("emergency", 0) + bonus
    crisis    = scores.get("crisis", 0) + scores.get("grief_crisis", 0) + scores.get("metaphor_crisis", 0) + scores.get("farewell", 0) + bonus
    self_harm = scores.get("self_harm", 0)
    distress  = scores.get("distress", 0)
    
    enjoyment = any(w in t for w in ["feel good","like","enjoy","love","relief","calm","better"])
    
    if emergency >= 10:                      return "EMERGENCY"
    if self_harm >= 7 and not enjoyment:     return "EMERGENCY"
    if self_harm >= 4 and enjoyment:         return "CRISIS"
    if crisis >= 8:                          return "CRISIS"
    if self_harm >= 4:                       return "CRISIS"
    if crisis >= 4 or distress >= 7:         return "BORDERLINE"
    if distress >= 3:                        return "DISTRESS"
    return "SAFE"

# ─────────────────────────────────────────────
# BUILD QA PAIRS
# ─────────────────────────────────────────────
qa_pairs = []

# From counsel_chat
if not cc.empty:
    for _, row in cc.iterrows():
        q = str(row.get('questionText', '')).strip()
        a = str(row.get('answerText', '')).strip()
        t = str(row.get('topic', 'general')).strip().lower()
        if len(q) > 20 and len(a) > 20:
            qa_pairs.append({'question': q, 'answer': a, 'topic': t, 'source': 'counsel_chat'})
    print(f"Added {len([p for p in qa_pairs if p['source']=='counsel_chat'])} counsel_chat pairs")

# From mental_health_counseling
if not mh.empty:
    for _, row in mh.iterrows():
        q = str(row.get('Context', '')).strip()
        a = str(row.get('Response', '')).strip()
        if len(q) > 20 and len(a) > 20:
            qa_pairs.append({'question': q, 'answer': a, 'topic': 'general', 'source': 'mental_health'})
    print(f"Added {len([p for p in qa_pairs if p['source']=='mental_health'])} mental_health pairs")

# From mental_chat_16k
if not mc.empty:
    for _, row in mc.iterrows():
        q = str(row.get('input', '')).strip()
        a = str(row.get('output', '')).strip()
        if len(q) > 20 and len(a) > 20:
            qa_pairs.append({'question': q, 'answer': a, 'topic': 'general', 'source': 'mental_chat'})
    print(f"Added {len([p for p in qa_pairs if p['source']=='mental_chat'])} mental_chat pairs")

# ─────────────────────────────────────────────
# IDP SPECIFIC QA PAIRS (Expanded)
# ─────────────────────────────────────────────
IDP_QA = [
    # Displacement and Loss
    {"question": "I lost my home in the flood and I feel like I have nothing left",
     "answer": "Losing your home in a flood is not just losing a building — it is losing safety, memory, and belonging all at once. The grief of displacement is real and deserves to be acknowledged. Can you tell me what feels most missing for you right now?",
     "topic": "displacement", "source": "idp"},
    
    {"question": "We are living in a camp and I feel ashamed",
     "answer": "Living in a camp after displacement is not something to be ashamed of — it is a sign that you survived something devastating. Shame after displacement belongs to the situation, not to you. What part of camp life feels hardest right now?",
     "topic": "displacement", "source": "idp"},
    
    {"question": "I cannot sleep at night in this new place",
     "answer": "After displacement, sleep rarely comes easily — new sounds, new fears, and old memories all arrive at night. Your body is still adjusting to an unsafe feeling even if the danger has passed. How long have you been struggling with sleep?",
     "topic": "sleep", "source": "idp"},
    
    # Homemaker/Mother specific
    {"question": "My children are hungry and I cannot provide for them",
     "answer": "The weight of watching your children go hungry is one of the heaviest a parent can carry. This is not a failure of your love — it is the result of a disaster. Are you connected with any relief organization that could help with food right now?",
     "topic": "homemaker", "source": "idp"},
    
    {"question": "I feel like I am failing my children because I cannot take care of them",
     "answer": "A mother who worries this deeply about her children is not failing them — she is loving them through an impossible situation. The fact that you are here, still asking how to do better, says everything. What is one small thing you managed to do for your children today?",
     "topic": "homemaker", "source": "idp"},
    
    {"question": "My husband lost his job and I feel helpless",
     "answer": "When displacement takes away livelihood, the whole family feels the weight. Your worry for your husband and family shows how deeply you care. What helps you get through the hardest moments of the day?",
     "topic": "homemaker", "source": "idp"},
    
    {"question": "I am a housewife and I feel useless now without a proper home",
     "answer": "A homemaker's value is not in the walls of a house but in the care she gives. You are still a homemaker — your home is wherever you and your family are. What part of caring for your family still brings you even a small sense of purpose?",
     "topic": "homemaker", "source": "idp"},
    
    # Worker/Unemployed specific
    {"question": "I lost my job because of the flood and I feel worthless",
     "answer": "Your worth is not measured by your employment. The flood took your livelihood — it did not take your value as a person. Many strong people across Pakistan are carrying this same silent shame and none of them deserve it. What kind of work did you do before?",
     "topic": "daily_worker", "source": "idp"},
    
    {"question": "I was a government employee and now I have nothing",
     "answer": "Losing the identity and routine of government service after displacement is a particular kind of grief. The respect, structure, and purpose you carried — those are still inside you. What part of your work gave you the most satisfaction?",
     "topic": "government", "source": "idp"},
    
    {"question": "I cannot find any work in this new place",
     "answer": "Searching for work while also managing displacement, family, and your own mental health is exhausting. The labor market is broken, not you. What kind of work have you been looking for?",
     "topic": "unemployed", "source": "idp"},
    
    {"question": "I feel ashamed that I cannot support my family anymore",
     "answer": "The weight of being unable to provide is crushing, but it does not erase your value as a husband, father, son, or brother. This shame belongs to the disaster, not to you. How long have you been carrying this weight alone?",
     "topic": "unemployed", "source": "idp"},
    
    # Student specific
    {"question": "I cannot focus on my studies after everything that happened",
     "answer": "When your home, routine, and safety have all been disrupted, your mind is still trying to process the loss. Focus does not return on demand after trauma — it returns slowly with safety and time. How long have you been displaced?",
     "topic": "student", "source": "idp"},
    
    {"question": "I have exams but I cannot concentrate at all",
     "answer": "Studying for exams while displaced is like trying to read a book in a storm. Your mind is protecting you by staying alert to danger rather than focusing on text. Have you been able to talk to anyone at your school or university about what you are going through?",
     "topic": "student", "source": "idp"},
    
    {"question": "I feel like I will never finish my education",
     "answer": "Displacement disrupts education — this is a reality millions of students face, and it is not your fault. Education is not a race, and your path may look different now, but that does not mean it has ended. What subject did you love most before everything changed?",
     "topic": "student", "source": "idp"},
    
    # Grief and Trauma
    {"question": "My friend who died keeps appearing in my dreams calling me",
     "answer": "Dreams of someone we have lost after sudden tragedy like a flood are a very natural part of grief. Your mind is still processing the loss and dreams are one way that happens. How long ago did you lose your friend?",
     "topic": "grief", "source": "idp"},
    
    {"question": "I survived the flood but my family did not and I feel guilty",
     "answer": "Survivor guilt after a disaster is one of the most painful experiences — the weight of being alive when others were not. You did not choose to survive. Grief and guilt are walking together inside you right now and both deserve space. Are you carrying this completely alone?",
     "topic": "grief", "source": "idp"},
    
    {"question": "I keep seeing the flood water rising when I close my eyes",
     "answer": "When trauma replays like this, it is your mind's way of trying to make sense of something that should never have happened. These images are not weakness — they are your brain's attempt to protect you from future danger. Do these images come more at night or during the day?",
     "topic": "ptsd", "source": "idp"},
    
    {"question": "I get scared whenever it rains now",
     "answer": "After surviving a flood, rain becomes a trigger — not just water. Your body remembers what your mind wants to forget. This is a normal trauma response. What helps you feel even slightly safer when the rain starts?",
     "topic": "ptsd", "source": "idp"},
    
    # Depression and Hopelessness
    {"question": "I am so tired all the time and have no energy",
     "answer": "After displacement, exhaustion is not just physical — it is the weight of constant uncertainty, broken sleep, grief, and the effort of surviving each day. When did this exhaustion begin — was it immediately after the flood or has it grown slowly?",
     "topic": "depression", "source": "idp"},
    
    {"question": "I feel hopeless about the future",
     "answer": "Hopelessness after displacement is not a personal weakness — it is a natural response when everything familiar has been disrupted. But hopelessness is not permanent even when it feels that way. What is one thing you would want for yourself if things were better?",
     "topic": "depression", "source": "idp"},
    
    {"question": "I feel heavy and lost I do not know what to do",
     "answer": "That heaviness after losing everything familiar is real and it makes complete sense. When home, routine, and safety are all gone at once, feeling lost is not a failure — it is a natural response. Can you tell me what feels heaviest right now — your thoughts, your sleep, or something else?",
     "topic": "depression", "source": "idp"},
    
    {"question": "Nothing brings me joy anymore",
     "answer": "When joy disappears after trauma and loss, it is not because joy is gone forever — it is because your mind is in survival mode. Pleasure and enjoyment return slowly, often starting with very small moments. What used to bring you joy, even a little, before everything changed?",
     "topic": "depression", "source": "idp"},
    
    # Isolation and Connection
    {"question": "I feel alone even though there are people around me in the camp",
     "answer": "Feeling alone in a crowd is one of the most painful kinds of loneliness. In camps, everyone is managing their own grief which makes real connection feel impossible. Is there one person in the camp you feel slightly comfortable being near?",
     "topic": "isolation", "source": "idp"},
    
    {"question": "nobody understands what I am going through",
     "answer": "In camps and temporary shelters, everyone is managing their own pain which can make real understanding feel impossible to find. Your experience is unique even if others around you also suffered. What do you wish someone understood about what you are carrying?",
     "topic": "isolation", "source": "idp"},
    
    {"question": "I have no one to talk to",
     "answer": "Being displaced often means losing not just a home but also your community, neighbors, and the people who knew you before. This isolation is real and painful. I am glad you reached out here. How long have you been feeling this alone?",
     "topic": "isolation", "source": "idp"},
    
    # Anxiety and Worry
    {"question": "I worry all the time about what will happen next",
     "answer": "Constant worry after displacement is your mind trying to prepare for the next disaster — it is trying to protect you. But living in constant alert is exhausting. What specific worry comes up most often?",
     "topic": "anxiety", "source": "idp"},
    
    {"question": "My heart races and I cannot breathe sometimes",
     "answer": "What you are describing sounds like panic — your body's alarm system going off even when there is no immediate danger. After displacement, this alarm can get stuck in the 'on' position. When did you first notice this happening?",
     "topic": "anxiety", "source": "idp"},
    
    {"question": "I feel like something terrible is about to happen",
     "answer": "That sense of impending doom is your nervous system still in survival mode. After living through a disaster, your body expects another one. This is not intuition — it is trauma. Can you tell me if this feeling is constant or does it come and go?",
     "topic": "anxiety", "source": "idp"},
    
    # Shame and Identity
    {"question": "I am ashamed to ask for help from others",
     "answer": "The shame of asking for help after displacement is one of the most common silent struggles among IDP communities. It is not weakness to need help after a disaster stripped away everything. What type of help feels most needed right now?",
     "topic": "shame", "source": "idp"},
    
    {"question": "I feel like I have lost my identity",
     "answer": "When home, work, community, and routine are all gone, it is natural to feel like you have lost yourself. But identity is deeper than circumstance. What parts of who you were before feel most distant now?",
     "topic": "identity", "source": "idp"},
    
    {"question": "I do not know who I am anymore",
     "answer": "Displacement can strip away everything that told you who you were — your home, your work, your role in the community. But you are still here, and that means you are more than what was lost. What is one thing about yourself that feels true even now?",
     "topic": "identity", "source": "idp"},
    
    # Spiritual distress
    {"question": "I feel like Allah has abandoned me",
     "answer": "Feeling abandoned by Allah after disaster is a painful and honest spiritual struggle. Many people of deep faith have felt this way in times of immense suffering. Would you like to talk about what specifically makes you feel this distance?",
     "topic": "spiritual", "source": "idp"},
    
    {"question": "Why did Allah let this happen to me",
     "answer": "That question has echoed through every disaster, every displacement, every loss. Faith does not mean having answers — it means continuing to ask while still holding on. Has anyone in your community or family offered you spiritual support?",
     "topic": "spiritual", "source": "idp"},
]

for pair in IDP_QA:
    qa_pairs.append(pair)

print(f"Added {len(IDP_QA)} IDP-specific pairs")
print(f"Total QA pairs: {len(qa_pairs)}")

# ─────────────────────────────────────────────
# VECTORIZE
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

print("Vectorizing all questions...")
questions_clean  = [clean_text(p['question']) for p in qa_pairs]
question_vectors = tfidf.transform(questions_clean)
print(f"Done: {question_vectors.shape[0]} vectors, {question_vectors.shape[1]} dimensions")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
output_path = "data/retrieval_system.pkl"
with open(output_path, "wb") as f:
    pickle.dump({
        'qa_pairs':         qa_pairs,
        'question_vectors': question_vectors,
        'crisis_detector':  crisis_data,  # Include for completeness
    }, f)

print(f"\n✓ Saved retrieval system to {output_path}")
print(f"  - {len(qa_pairs)} total QA pairs")
print(f"  - {len([p for p in qa_pairs if p['source']=='idp'])} IDP-specific pairs")
print(f"  - {len([p for p in qa_pairs if p['source']=='counsel_chat'])} counsel_chat pairs")
print(f"  - {len([p for p in qa_pairs if p['source']=='mental_health'])} mental_health pairs")
print(f"  - {len([p for p in qa_pairs if p['source']=='mental_chat'])} mental_chat pairs")

# Quick test
print("\n" + "="*60)
print("QUICK TEST")
print("="*60)

test_queries = [
    ("I lost my home in the flood", "depression", "homemaker"),
    ("I cannot provide for my children", "depression", "homemaker"),
    ("I feel hopeless", "depression", "default"),
]

for query, emotion, role in test_queries:
    cleaned = clean_text(query)
    user_vec = tfidf.transform([cleaned])
    sims = cosine_similarity(user_vec, question_vectors)[0]
    
    # Boost IDP content
    for i, pair in enumerate(qa_pairs):
        if pair['source'] == 'idp':
            sims[i] *= 1.8
        if role and pair['topic'] == role:
            sims[i] *= 1.3
    
    top_idx = np.argmax(sims)
    top_pair = qa_pairs[top_idx]
    
    print(f"\nQuery: '{query}'")
    print(f"Matched: [{top_pair['source']}] {top_pair['question'][:60]}...")
    print(f"Answer: {top_pair['answer'][:100]}...")

print("\n✓ Retrieval system built successfully!")
print("Now you can run app.py")