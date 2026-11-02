"""
Voice-Enabled Assistant for Campus Queries
------------------------------------------
Flask backend: intent detection + entity extraction + database-backed answers.

Pipeline:  (browser) speech -> text  ->  POST /api/query  ->  intent + entities
           -> database lookup -> answer text  ->  (browser) text -> speech

All data in the seed section is SAMPLE data. Replace it with official
timetable / fee / notice data before any real use.
"""
import difflib
import json
import os
import re
import sqlite3
import time
import urllib.request
from datetime import datetime, timedelta

from flask import Flask, g, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(BASE_DIR, "campus.db")

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}
COURSES = {
    "MCA": ["mca", "master of computer applications", "masters in computer applications"],
    "BCA": ["bca", "bachelor of computer applications"],
    "BTECH": ["btech", "b tech", "b.tech", "engineering", "be cse", "cse"],
    "MBA": ["mba", "master of business administration"],
}
DEFAULT_COURSE = "MCA"
DEFAULT_SEM = 1

# --------------------------------------------------------------------------
# Intent definitions: phrase -> weight.  Multi-word phrases are matched with
# word boundaries; single words are matched as tokens (exact or fuzzy).
# --------------------------------------------------------------------------
INTENTS = {
    "greeting": {
        "hello": 1.0, "hi": 1.0, "hey": 1.0, "namaste": 1.0, "good morning": 1.2,
        "good afternoon": 1.2, "good evening": 1.2, "sat sri akal": 1.2,
    },
    "timetable": {
        "timetable": 2.5, "time table": 2.5, "schedule": 1.5, "class": 2.0,
        "classes": 2.0, "lecture": 2.0, "lectures": 2.0, "period": 1.5,
        "periods": 1.5, "routine": 1.5, "today's schedule": 2.5, "next class": 3.0,
        "what do i have": 1.5, "subjects today": 2.0, "subject": 1.5, "subjects": 1.5,
    },
    "fees": {
        "fee": 2.5, "fees": 2.5, "tuition": 2.5, "payment": 1.5, "pay": 1.2,
        "cost": 1.5, "charges": 1.5, "fee structure": 3.0, "how much": 1.5,
        "hostel fee": 3.0, "hostel fees": 3.0, "fee due": 2.5,
    },
    "notice": {
        "notice": 2.5, "notices": 2.5, "announcement": 2.5, "announcements": 2.5,
        "circular": 2.0, "news": 1.8, "notification": 2.0, "notifications": 2.0,
        "update": 1.2, "updates": 1.2, "whats new": 2.0, "any new": 1.5,
        "latest": 1.2, "noticeboard": 2.5, "notice board": 2.5,
    },
    "exam": {
        "exam": 2.8, "exams": 2.8, "examination": 2.8, "datesheet": 3.0,
        "date sheet": 3.0, "exam schedule": 4.0, "exam date": 4.0, "exam dates": 4.0,
        "midterm": 2.5, "mid term": 2.5, "endterm": 2.5, "end term": 2.5,
        "test": 1.0, "viva": 1.5, "when is the exam": 4.0,
    },
    "help": {
        "help": 2.0, "what can you do": 3.0, "options": 1.5, "features": 1.5,
        "how to use": 2.5, "how do i use": 2.5,
    },
}

MIN_SCORE = 1.0


# --------------------------------------------------------------------------
# Text normalisation + entity extraction
# --------------------------------------------------------------------------
def collapse_letters(tokens):
    """Speech engines often return 'M C A' -> merge runs of single letters."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def normalize(text):
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9'\s\.]", " ", text)
    text = text.replace("b.tech", "btech").replace("b tech", "btech")
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"'s\b", "s", text)  # today's -> todays
    tokens = collapse_letters(text.split())
    return " ".join(tokens)


def extract_course(norm):
    for code, aliases in COURSES.items():
        for a in aliases:
            a_norm = normalize(a)
            if re.search(r"\b" + re.escape(a_norm) + r"\b", norm):
                return code
    return None


def extract_semester(norm):
    m = re.search(r"\bsem(?:ester)?\s*(\d)\b", norm) or re.search(r"\b(\d)(?:st|nd|rd|th)?\s*sem", norm)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 8:
            return n
    for word, n in ORDINALS.items():
        if re.search(r"\b" + word + r"\s+sem", norm):
            return n
    return None


def extract_day(norm, now=None):
    now = now or datetime.now()
    if re.search(r"\btomorrow\b", norm):
        return DAYS[(now + timedelta(days=1)).weekday()], "tomorrow"
    if re.search(r"\byesterday\b", norm):
        return DAYS[(now - timedelta(days=1)).weekday()], "yesterday"
    for d in DAYS:
        if re.search(r"\b" + d + r"\b", norm):
            return d, d.capitalize()
    # fuzzy: speech may give 'monde', 'wensday'
    for tok in norm.split():
        if len(tok) >= 5:
            match = difflib.get_close_matches(tok, DAYS, n=1, cutoff=0.8)
            if match:
                return match[0], match[0].capitalize()
    return DAYS[now.weekday()], "today"


def extract_entities(text, now=None):
    norm = normalize(text)
    day, day_label = extract_day(norm, now)
    return {
        "course": extract_course(norm),
        "semester": extract_semester(norm),
        "day": day,
        "day_label": day_label,
        "hostel": bool(re.search(r"\bhostel\b", norm)),
        "all": bool(re.search(r"\b(all|every|list)\b", norm)),
    }


# --------------------------------------------------------------------------
# Intent detection (rule-based, fuzzy) + optional LLM fallback
# --------------------------------------------------------------------------
def score_intents(norm):
    tokens = norm.split()
    token_set = set(tokens)
    scores = {}
    for intent, phrases in INTENTS.items():
        total = 0.0
        for phrase, weight in phrases.items():
            p = normalize(phrase)
            if " " in p:
                if re.search(r"\b" + re.escape(p) + r"\b", norm):
                    total += weight
            else:
                if p in token_set:
                    total += weight
                elif len(p) >= 5:  # fuzzy match for speech-recognition slips
                    if difflib.get_close_matches(p, [t for t in tokens if len(t) >= 4], n=1, cutoff=0.86):
                        total += weight * 0.8
        scores[intent] = round(total, 3)
    return scores


def rule_based_intent(text):
    norm = normalize(text)
    if not norm:
        return "empty", 0.0, {}
    scores = score_intents(norm)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_s = ranked[0]
    second_s = ranked[1][1]
    if best_s < MIN_SCORE:
        return "unknown", 0.0, scores
    confidence = best_s / (best_s + second_s + 0.5)
    return best, round(min(confidence, 0.99), 2), scores


def llm_intent(text):
    """Optional fallback. Active only if ANTHROPIC_API_KEY is set.
    Returns an intent name or None. Never raises."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    labels = list(INTENTS.keys()) + ["unknown"]
    prompt = (
        "Classify this campus-assistant query into exactly one label from: "
        + ", ".join(labels)
        + ". Reply with the label only.\nQuery: " + text
    )
    body = json.dumps({
        "model": os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"),
        "max_tokens": 10,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            out = json.loads(r.read())["content"][0]["text"].strip().lower()
        return out if out in labels else None
    except Exception:
        return None


def detect_intent(text):
    intent, conf, scores = rule_based_intent(text)
    source = "rules"
    if intent in ("unknown",) or (intent != "empty" and conf < 0.5):
        guess = llm_intent(text)
        if guess:
            intent, conf, source = guess, max(conf, 0.7), "llm"
    return {"intent": intent, "confidence": conf, "scores": scores, "source": source}


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS fees (
    course TEXT PRIMARY KEY, tuition_per_sem INTEGER, exam_fee_per_sem INTEGER,
    hostel_per_year INTEGER
);
CREATE TABLE IF NOT EXISTS timetable (
    id INTEGER PRIMARY KEY AUTOINCREMENT, course TEXT, semester INTEGER, day TEXT,
    start_time TEXT, end_time TEXT, subject TEXT, room TEXT, faculty TEXT
);
CREATE TABLE IF NOT EXISTS notices (
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, body TEXT, posted_on TEXT
);
CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT, course TEXT, semester INTEGER,
    subject TEXT, exam_date TEXT, start_time TEXT, kind TEXT
);
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, text TEXT, intent TEXT,
    confidence REAL, latency_ms REAL
);
"""


def seed(db):
    if db.execute("SELECT COUNT(*) FROM fees").fetchone()[0]:
        return
    today = datetime.now().date()
    db.executemany("INSERT INTO fees VALUES (?,?,?,?)", [
        ("MCA", 75000, 3000, 90000),
        ("BCA", 55000, 2500, 90000),
        ("BTECH", 95000, 3500, 95000),
        ("MBA", 110000, 3500, 95000),
    ])
    slots = [("09:00", "09:50"), ("10:00", "10:50"), ("11:30", "12:20"), ("12:30", "13:20")]
    plan = {
        "monday":    [("Data Structures", "A-101", "Dr. Sharma"), ("Python Programming", "Lab-3", "Prof. Kaur"),
                      ("Database Management Systems", "A-102", "Dr. Gill"), ("Discrete Mathematics", "A-101", "Prof. Verma")],
        "tuesday":   [("Computer Organization", "A-103", "Dr. Singh"), ("Data Structures", "A-101", "Dr. Sharma"),
                      ("Communication Skills", "B-201", "Prof. Mehta"), ("Python Programming", "Lab-3", "Prof. Kaur")],
        "wednesday": [("Database Management Systems", "A-102", "Dr. Gill"), ("Discrete Mathematics", "A-101", "Prof. Verma"),
                      ("Data Structures Lab", "Lab-1", "Dr. Sharma"), ("Data Structures Lab", "Lab-1", "Dr. Sharma")],
        "thursday":  [("Python Programming", "Lab-3", "Prof. Kaur"), ("Computer Organization", "A-103", "Dr. Singh"),
                      ("Database Management Systems", "A-102", "Dr. Gill"), ("Communication Skills", "B-201", "Prof. Mehta")],
        "friday":    [("Discrete Mathematics", "A-101", "Prof. Verma"), ("Data Structures", "A-101", "Dr. Sharma"),
                      ("DBMS Lab", "Lab-2", "Dr. Gill"), ("DBMS Lab", "Lab-2", "Dr. Gill")],
        "saturday":  [("Seminar / Mentoring", "A-104", "Prof. Mehta")],
    }
    rows = []
    for day, subjects in plan.items():
        for (st, en), (subj, room, fac) in zip(slots, subjects):
            rows.append(("MCA", 1, day, st, en, subj, room, fac))
    db.executemany(
        "INSERT INTO timetable (course,semester,day,start_time,end_time,subject,room,faculty) VALUES (?,?,?,?,?,?,?,?)",
        rows)
    notices = [
        ("Mid-semester registration open", "Register for mid-semester exams on the student portal before the deadline.", today - timedelta(days=1)),
        ("Library timing change", "The central library will remain open until 8 PM on weekdays.", today - timedelta(days=3)),
        ("Tech fest registrations", "Registrations for the annual tech fest are open for all students.", today - timedelta(days=5)),
        ("Fee payment reminder", "Students with pending tuition dues should clear them at the accounts office.", today - timedelta(days=8)),
    ]
    db.executemany("INSERT INTO notices (title, body, posted_on) VALUES (?,?,?)",
                   [(t, b, d.isoformat()) for t, b, d in notices])
    exam_rows = []
    for i, subj in enumerate(["Data Structures", "Python Programming", "Database Management Systems",
                              "Computer Organization", "Discrete Mathematics"]):
        exam_rows.append(("MCA", 1, subj, (today + timedelta(days=18 + 2 * i)).isoformat(), "10:00", "Mid-semester"))
    db.executemany("INSERT INTO exams (course,semester,subject,exam_date,start_time,kind) VALUES (?,?,?,?,?,?)",
                   exam_rows)
    db.commit()


def init_db(path):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    seed(db)
    db.close()


# --------------------------------------------------------------------------
# Handlers: each returns {"answer": str, "data": list|None, "kind": str}
# --------------------------------------------------------------------------
def fmt_time(t):
    return datetime.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")


def fmt_date(iso):
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b %Y")


def money(n):
    return f"₹{n:,}"


def h_greeting(db, ent):
    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    return {"kind": "text", "data": None,
            "answer": f"Good {part}! I can help with your timetable, fees, notices and exam dates. What would you like to know?"}


def h_help(db, ent):
    return {"kind": "text", "data": None,
            "answer": ("I can answer questions about: timetable (\"What is my schedule on Monday?\"), "
                       "fees (\"MCA fee structure\"), notices (\"Any new notices?\") and exams "
                       "(\"When are the exams?\"). Tap the mic and speak, or type your question.")}


def h_timetable(db, ent):
    course = ent["course"] or DEFAULT_COURSE
    sem = ent["semester"] or DEFAULT_SEM
    day, label = ent["day"], ent["day_label"]
    if day == "sunday":
        return {"kind": "text", "data": None, "answer": "There are no classes on Sunday."}
    rows = db.execute(
        "SELECT * FROM timetable WHERE course=? AND semester=? AND day=? ORDER BY start_time",
        (course, sem, day)).fetchall()
    if not rows:
        return {"kind": "text", "data": None,
                "answer": f"I could not find a timetable for {course} semester {sem} on {day.capitalize()}."}
    data = [{"time": f"{fmt_time(r['start_time'])} - {fmt_time(r['end_time'])}", "subject": r["subject"],
             "room": r["room"], "faculty": r["faculty"]} for r in rows]
    suffix = f" ({label})" if label in ("today", "tomorrow", "yesterday") else ""
    lines = "; ".join(f"{fmt_time(r['start_time'])} {r['subject']} in {r['room']}" for r in rows)
    return {"kind": "table", "data": data, "columns": ["time", "subject", "room", "faculty"],
            "answer": f"{course} semester {sem}, {day.capitalize()}{suffix}: {lines}."}


def h_fees(db, ent):
    course = ent["course"] or DEFAULT_COURSE
    r = db.execute("SELECT * FROM fees WHERE course=?", (course,)).fetchone()
    if not r:
        return {"kind": "text", "data": None, "answer": f"I do not have fee details for {course}."}
    if ent["hostel"]:
        return {"kind": "text", "data": None,
                "answer": f"The hostel fee is {money(r['hostel_per_year'])} per year."}
    total = r["tuition_per_sem"] + r["exam_fee_per_sem"]
    return {"kind": "text", "data": None,
            "answer": (f"{course} tuition fee is {money(r['tuition_per_sem'])} per semester, plus an exam fee of "
                       f"{money(r['exam_fee_per_sem'])}. Total per semester: {money(total)}.")}


def h_notice(db, ent):
    limit = 5 if ent["all"] else 3
    rows = db.execute("SELECT * FROM notices ORDER BY posted_on DESC, id DESC LIMIT ?", (limit,)).fetchall()
    if not rows:
        return {"kind": "text", "data": None, "answer": "There are no notices right now."}
    data = [{"date": fmt_date(r["posted_on"]), "title": r["title"], "details": r["body"]} for r in rows]
    spoken = " ".join(f"{i}. {r['title']}." for i, r in enumerate(rows, 1))
    return {"kind": "table", "data": data, "columns": ["date", "title", "details"],
            "answer": f"Here are the latest {len(rows)} notices. {spoken}"}


def h_exam(db, ent):
    course = ent["course"] or DEFAULT_COURSE
    sem = ent["semester"] or DEFAULT_SEM
    today = datetime.now().date().isoformat()
    rows = db.execute(
        "SELECT * FROM exams WHERE course=? AND semester=? AND exam_date>=? ORDER BY exam_date",
        (course, sem, today)).fetchall()
    if not rows:
        return {"kind": "text", "data": None, "answer": f"No upcoming exams found for {course} semester {sem}."}
    data = [{"date": fmt_date(r["exam_date"]), "subject": r["subject"],
             "time": fmt_time(r["start_time"]), "type": r["kind"]} for r in rows]
    first = rows[0]
    return {"kind": "table", "data": data, "columns": ["date", "subject", "time", "type"],
            "answer": (f"{course} semester {sem} has {len(rows)} upcoming exams. The first is "
                       f"{first['subject']} on {fmt_date(first['exam_date'])} at {fmt_time(first['start_time'])}.")}


def h_unknown(db, ent):
    return {"kind": "text", "data": None,
            "answer": ("Sorry, I can only help with campus queries like timetable, fees, notices and exams. "
                       "Try: \"What is my timetable for Monday?\"")}


def h_empty(db, ent):
    return {"kind": "text", "data": None, "answer": "Sorry, I didn't catch that. Please try again."}


HANDLERS = {
    "greeting": h_greeting, "help": h_help, "timetable": h_timetable, "fees": h_fees,
    "notice": h_notice, "exam": h_exam, "unknown": h_unknown, "empty": h_empty,
}


def answer_query(db, text):
    """Core logic, usable without Flask (tests / eval)."""
    text = (text or "").strip()[:300]
    det = detect_intent(text)
    ent = extract_entities(text)
    result = HANDLERS[det["intent"]](db, ent)
    return {"query": text, "intent": det["intent"], "confidence": det["confidence"],
            "source": det["source"], "entities": ent, **result}


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
def create_app(db_path=None):
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path or os.environ.get("CAMPUS_DB", DEFAULT_DB)
    app.config["ADMIN_KEY"] = os.environ.get("ADMIN_KEY", "admin123")
    init_db(app.config["DB_PATH"])

    def get_db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DB_PATH"])
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/admin")
    def admin():
        return render_template("admin.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", time=datetime.now().isoformat(timespec="seconds"))

    @app.post("/api/query")
    def api_query():
        t0 = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        db = get_db()
        res = answer_query(db, payload.get("text", ""))
        latency = round((time.perf_counter() - t0) * 1000, 1)
        res["latency_ms"] = latency
        db.execute("INSERT INTO query_log (ts,text,intent,confidence,latency_ms) VALUES (?,?,?,?,?)",
                   (datetime.now().isoformat(timespec="seconds"), res["query"], res["intent"],
                    res["confidence"], latency))
        db.commit()
        return jsonify(res)

    @app.get("/api/notices")
    def list_notices():
        rows = get_db().execute("SELECT * FROM notices ORDER BY posted_on DESC, id DESC").fetchall()
        return jsonify([dict(r) for r in rows])

    @app.post("/api/notices")
    def add_notice():
        p = request.get_json(silent=True) or {}
        if p.get("key") != app.config["ADMIN_KEY"]:
            return jsonify(error="Invalid admin key"), 403
        title, body = (p.get("title") or "").strip(), (p.get("body") or "").strip()
        if not title:
            return jsonify(error="Title is required"), 400
        db = get_db()
        db.execute("INSERT INTO notices (title, body, posted_on) VALUES (?,?,?)",
                   (title[:150], body[:500], datetime.now().date().isoformat()))
        db.commit()
        return jsonify(ok=True), 201

    @app.get("/api/stats")
    def stats():
        db = get_db()
        total = db.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        avg = db.execute("SELECT AVG(latency_ms) FROM query_log").fetchone()[0]
        dist = db.execute("SELECT intent, COUNT(*) c FROM query_log GROUP BY intent").fetchall()
        return jsonify(total_queries=total, avg_latency_ms=round(avg or 0, 2),
                       intents={r["intent"]: r["c"] for r in dist})

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
