"""
Intent detection evaluation.
Run:  python eval_intents.py
Prints accuracy, per-intent precision / recall / F1 and a confusion matrix.
Use these numbers directly in the project report.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as campus  # noqa: E402

# Force rule-based only so results are reproducible (no network / API key).
os.environ.pop("ANTHROPIC_API_KEY", None)

DATASET = [
    # timetable
    ("What is my timetable for Monday?", "timetable"),
    ("show me today's schedule", "timetable"),
    ("when is my next class", "timetable"),
    ("lecture timings for tuesday", "timetable"),
    ("which classes do I have tomorrow", "timetable"),
    ("time table for MCA first semester", "timetable"),
    ("what subjects are there on friday", "timetable"),
    ("class routine for wednesday", "timetable"),
    ("timetable pls", "timetable"),
    ("tell me my lectures today", "timetable"),
    # fees
    ("MCA fee structure", "fees"),
    ("how much is the tuition fee", "fees"),
    ("what is the hostel fee", "fees"),
    ("fees for BCA", "fees"),
    ("tell me the fee for first semester", "fees"),
    ("what are the charges for MBA", "fees"),
    ("fee due date and payment details", "fees"),
    ("how much do I have to pay", "fees"),
    ("btech fees", "fees"),
    ("semester fee amount", "fees"),
    # notices
    ("any new notices?", "notice"),
    ("latest announcements", "notice"),
    ("what's new on the notice board", "notice"),
    ("read out the circulars", "notice"),
    ("show notifications", "notice"),
    ("any news from college", "notice"),
    ("list all notices", "notice"),
    ("recent updates please", "notice"),
    ("notices today", "notice"),
    ("is there any announcement", "notice"),
    # exams
    ("when are the exams", "exam"),
    ("exam schedule", "exam"),
    ("mid term dates", "exam"),
    ("date sheet for MCA", "exam"),
    ("when is my first exam", "exam"),
    ("end term examination dates", "exam"),
    ("tell me the exam dates for semester 1", "exam"),
    ("upcoming exams", "exam"),
    ("when is the viva", "exam"),
    ("examination schedule please", "exam"),
    # greeting
    ("hello", "greeting"),
    ("hi there", "greeting"),
    ("good morning", "greeting"),
    ("hey", "greeting"),
    ("namaste", "greeting"),
    # help
    ("what can you do", "help"),
    ("help", "help"),
    ("how to use this", "help"),
    ("what are your features", "help"),
    # out of scope
    ("who won the cricket match", "unknown"),
    ("what's the weather today", "unknown"),
    ("tell me a joke", "unknown"),
    ("play some music", "unknown"),
    ("what is the capital of france", "unknown"),
    ("book a cab for me", "unknown"),
    ("how do I cook pasta", "unknown"),
    ("what is the price of gold", "unknown"),
]

LABELS = ["timetable", "fees", "notice", "exam", "greeting", "help", "unknown"]


def main():
    y_true, y_pred, wrong = [], [], []
    for text, label in DATASET:
        pred = campus.detect_intent(text)["intent"]
        y_true.append(label)
        y_pred.append(pred)
        if pred != label:
            wrong.append((text, label, pred))

    n = len(DATASET)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"Samples: {n}   Correct: {correct}   Accuracy: {correct / n:.1%}\n")

    print(f"{'intent':<11}{'precision':>10}{'recall':>9}{'f1':>7}{'support':>9}")
    f1s = []
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        print(f"{lab:<11}{prec:>10.2f}{rec:>9.2f}{f1:>7.2f}{tp + fn:>9}")
    print(f"{'macro avg':<11}{'':>10}{'':>9}{sum(f1s) / len(f1s):>7.2f}\n")

    print("Confusion matrix (rows = actual, columns = predicted)")
    cols = LABELS + ["empty"]
    print(f"{'':<11}" + "".join(f"{c[:6]:>8}" for c in cols))
    for a in LABELS:
        row = [sum(t == a and p == c for t, p in zip(y_true, y_pred)) for c in cols]
        print(f"{a:<11}" + "".join(f"{v:>8}" for v in row))

    if wrong:
        print("\nMisclassified:")
        for text, label, pred in wrong:
            print(f"  '{text}'  expected={label}  got={pred}")


if __name__ == "__main__":
    main()
