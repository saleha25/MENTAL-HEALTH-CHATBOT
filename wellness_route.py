# ─────────────────────────────────────────────
# WELLNESS PAGE
# Paste this block into app.py above the quotes route / before # RUN APP
# ─────────────────────────────────────────────

def _score_to_10(value, maximum):
    if value is None:
        return None
    try:
        return round((float(value) / float(maximum)) * 10, 1)
    except Exception:
        return None

def _display_date(dt_value):
    try:
        dt = datetime.fromisoformat(str(dt_value))
        return dt.strftime("%d %b %Y")
    except Exception:
        return str(dt_value)

def _mood_to_num(mood):
    mood_map = {
        "very_low": 1,
        "low": 2,
        "okay": 3,
        "good": 4,
        "better": 4,
        "anxious": 2
    }
    return mood_map.get((mood or "").lower())

def _energy_to_num(energy):
    energy_map = {
        "very_low": 1,
        "low": 2,
        "medium": 3,
        "high": 4
    }
    return energy_map.get((energy or "").lower())

def _mood_dot_color(mood):
    mood = (mood or "").lower()
    if mood == "very_low":
        return "#d9534f"
    if mood in ["low", "anxious"]:
        return "#f0ad4e"
    if mood == "okay":
        return "#5bc0de"
    return "#5cb85c"

def _energy_dot_color(energy):
    energy = (energy or "").lower()
    if energy == "very_low":
        return "#d9534f"
    if energy == "low":
        return "#f0ad4e"
    if energy == "medium":
        return "#5bc0de"
    return "#5cb85c"

def _sleep_bar_color(hours):
    if hours is None:
        return "#d8cdbf"
    if hours >= 7:
        return "#4CAF50"
    if hours >= 5:
        return "#F4C542"
    return "#D9534F"

@app.route("/wellness")
@login_required
def wellness():
    db = get_db()
    account_id = session.get("account_id")
    user = get_user_by_account_id(account_id) if account_id else None

    if not user:
        return redirect(url_for("dashboard"))

    user_id = user["id"]

    wellbeing_rows = db.execute("""
        SELECT *
        FROM wellbeing_logs
        WHERE user_id = ?
        ORDER BY log_date ASC, id ASC
    """, (user_id,)).fetchall()

    assessment_rows_raw = db.execute("""
        SELECT *
        FROM assessments
        WHERE user_id = ?
        ORDER BY assessment_date ASC, id ASC
    """, (user_id,)).fetchall()

    avg_sleep_values = [float(r["sleep_hours"]) for r in wellbeing_rows if r["sleep_hours"] is not None]
    avg_sleep = round(sum(avg_sleep_values) / len(avg_sleep_values), 1) if avg_sleep_values else 0

    latest_mood = "—"
    latest_energy = "—"
    if wellbeing_rows:
        for r in reversed(wellbeing_rows):
            if latest_mood == "—" and r["mood"]:
                latest_mood = str(r["mood"]).replace("_", " ").title()
            if latest_energy == "—" and r["energy"]:
                latest_energy = str(r["energy"]).replace("_", " ").title()
            if latest_mood != "—" and latest_energy != "—":
                break

    assessment_rows = []
    for row in assessment_rows_raw:
        assessment_rows.append({
            "display_date": _display_date(row["assessment_date"]),
            "phq10": _score_to_10(row["phq9_total"], 27),
            "gad10": _score_to_10(row["gad7_total"], 21),
            "who10": _score_to_10(row["who5_total"], 25),
            "severity": row["severity"] or "—",
            "emotion_detected": row["emotion_detected"] or "—"
        })

    sleep_labels = []
    sleep_values = []
    sleep_colors = []

    mood_labels = []
    mood_values = []
    mood_colors = []

    energy_labels = []
    energy_values = []
    energy_colors = []

    for row in wellbeing_rows:
        label = _display_date(row["log_date"])

        if row["sleep_hours"] is not None:
            sleep_labels.append(label)
            sleep_values.append(float(row["sleep_hours"]))
            sleep_colors.append(_sleep_bar_color(float(row["sleep_hours"])))

        if row["mood"]:
            mood_labels.append(label)
            mood_values.append(_mood_to_num(row["mood"]))
            mood_colors.append(_mood_dot_color(row["mood"]))

        if row["energy"]:
            energy_labels.append(label)
            energy_values.append(_energy_to_num(row["energy"]))
            energy_colors.append(_energy_dot_color(row["energy"]))

    assessment_labels = [_display_date(r["assessment_date"]) for r in assessment_rows_raw]
    phq_values = [_score_to_10(r["phq9_total"], 27) for r in assessment_rows_raw]
    gad_values = [_score_to_10(r["gad7_total"], 21) for r in assessment_rows_raw]
    who_values = [_score_to_10(r["who5_total"], 25) for r in assessment_rows_raw]

    chart_data = {
        "sleep_labels": sleep_labels,
        "sleep_values": sleep_values,
        "sleep_colors": sleep_colors,
        "mood_labels": mood_labels,
        "mood_values": mood_values,
        "mood_colors": mood_colors,
        "energy_labels": energy_labels,
        "energy_values": energy_values,
        "energy_colors": energy_colors,
        "assessment_labels": assessment_labels,
        "phq_values": phq_values,
        "gad_values": gad_values,
        "who_values": who_values,
    }

    summary = {
        "avg_sleep": avg_sleep,
        "latest_mood": latest_mood,
        "latest_energy": latest_energy,
        "total_assessments": len(assessment_rows_raw)
    }

    has_data = bool(wellbeing_rows or assessment_rows_raw)

    return render_template(
        "wellness.html",
        summary=summary,
        chart_data=chart_data,
        assessment_rows=assessment_rows[::-1],
        has_data=has_data
    )