# Voice-Enabled Assistant for Campus Queries

A web-based voice assistant that answers student questions about **timetable, fees, notices and exam dates**.
The student speaks (or types) a question, the system detects the intent, looks up the answer in a
database, and speaks the reply back.

> All timetable, fee, notice and exam data in this repo is **sample data** for demonstration.
> Replace it with official data before real use.

## Features
- Voice input (browser Web Speech API) and typed input
- Spoken replies (browser text-to-speech), with an on/off toggle
- 7 intents: `timetable`, `fees`, `notice`, `exam`, `greeting`, `help`, `unknown` (out-of-scope)
- Entity extraction: course (MCA/BCA/BTech/MBA), semester, day (`today`, `tomorrow`, weekday names)
- Handles speech-recognition slips: "M C A" is read as MCA, and misspelt words are fuzzy-matched
- Admin page to post new notices, which the assistant reads out immediately
- Query log with latency and intent statistics (`/admin`, `/api/stats`)
- Optional LLM fallback for hard-to-classify queries (off unless `ANTHROPIC_API_KEY` is set)
- 36 automated tests plus an intent evaluation script (accuracy, precision, recall, F1, confusion matrix)

## Tech stack
Python 3 · Flask · SQLite · HTML/CSS/JavaScript · Web Speech API (STT + TTS)

## How to run
```bash
cd voice_campus_assistant
pip install -r requirements.txt
python app.py
```
Open **http://localhost:5000** in **Chrome or Edge** and allow microphone access.
(Voice input needs Chrome/Edge and an internet connection, because the browser's speech service is
cloud-based. Typing works in any browser.)

The database file `campus.db` is created and seeded automatically on first run.
Delete it to reset the data.

### Optional: LLM fallback
```bash
export ANTHROPIC_API_KEY=your_key      # Windows PowerShell: $env:ANTHROPIC_API_KEY="your_key"
```
When the rule-based classifier is unsure, the query is classified by an LLM. Without a key, the system
works fully on rules alone.

### Admin key
Default is `admin123` (demo only). Change it with `ADMIN_KEY=yourkey python app.py`.

## Architecture
```
 Browser                                   Flask server
┌──────────────┐  text   ┌───────────────────────────────────────────┐
│ Mic -> Web    │ ──────> │ POST /api/query                           │
│ Speech API    │         │   1. normalize text                        │
│ (speech->text)│         │   2. detect intent (rules + fuzzy, LLM opt)│
│               │         │   3. extract entities (course/sem/day)     │
│ Text -> TTS   │ <────── │   4. handler queries SQLite -> answer      │
│ (speak reply) │  JSON   │   5. log query + latency                   │
└──────────────┘         └───────────────────────────────────────────┘
                                    │
                         SQLite: fees · timetable · notices · exams · query_log
```

## API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/query` | `{"text": "..."}` returns intent, confidence, entities, answer, data, latency_ms |
| GET | `/api/notices` | List all notices |
| POST | `/api/notices` | Add notice (`title`, `body`, `key`) |
| GET | `/api/stats` | Total queries, average latency, intent distribution |
| GET | `/api/health` | Health check |

## Testing
```bash
python -m pytest tests -v        # 36 unit and API tests
python eval_intents.py           # intent accuracy, precision/recall/F1, confusion matrix
```
Test coverage: normalisation, entity extraction, intent detection, database-backed answers, edge cases
(empty input, very long input, bad JSON, SQL-injection attempt), admin authorisation, latency.

Results on the bundled 57-query labelled set (rule-based, no LLM): see the output of
`eval_intents.py`. Extend `DATASET` in that file with your own recorded or typed queries and
re-run it to report your own numbers.

**Manual tests to document in your report**
1. Speech-to-text: speak 20-30 queries in quiet and noisy rooms and note which are mis-transcribed.
   Word Error Rate = (substitutions + deletions + insertions) / words in the reference.
2. Failure handling: deny mic permission, go offline, stay silent, then confirm that the UI shows a
   clear message each time.
3. User acceptance: 5-10 classmates use it, then rate ease of use, accuracy and speed from 1 to 5.

## 5-minute demo script
1. Open the app and tap a quick-reply chip to show the text flow.
2. Tap the mic and say **"What is my timetable for Monday?"** (table shown, answer spoken).
3. Say **"MCA fee structure"**, then **"What is the hostel fee?"**
4. Say **"Any new notices?"**
5. Open `/admin`, post a new notice, return and ask for notices again. The new one is read first.
6. Say **"When are the exams?"**
7. Say **"Who won the cricket match?"** to show the graceful out-of-scope reply.
8. Show `/admin` stats (query count, average latency) and run `python eval_intents.py` for accuracy.

## Project structure
```
voice_campus_assistant/
├── app.py               # Flask app, intent engine, handlers, DB
├── templates/
│   ├── index.html       # Voice + chat UI
│   └── admin.html       # Notice admin + stats
├── tests/test_app.py    # 36 automated tests
├── eval_intents.py      # intent evaluation (metrics for the report)
├── requirements.txt
└── README.md
```

## Limitations and future work
- Voice input works in English (en-IN); Hindi/Punjabi would need a language option and keyword lists.
- Web Speech API needs internet and is supported mainly in Chrome/Edge.
- Sample data covers MCA semester 1 for the timetable and exams; add rows to the tables for other courses.
- No student login, so answers are the same for everyone. Future: per-student timetable and fee dues
  after authentication.
- Intent detection is rule-based with fuzzy matching. A trained classifier or an LLM could
  handle more varied phrasing.
