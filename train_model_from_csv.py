"""
train_model_from_csv.py
Run once (or whenever you want a fresh retrain) to produce:
  data/calendar_attendance_model.pkl
  data/label_encoder_user.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FOLDER  = "data"
CSV_FILE     = os.path.join(DATA_FOLDER, "dummy_calendar_events.csv")
MODEL_FILE   = os.path.join(DATA_FOLDER, "calendar_attendance_model.pkl")
ENCODER_FILE = os.path.join(DATA_FOLDER, "label_encoder_user.pkl")

# ── Features (must match model.py FEATURES list exactly) ─────────────────────
FEATURES = [
    "Hour",
    "DayOfWeek",
    "DurationMinutes",
    "IsRecurring",
    "Importance",
    "UserIdEnc",
    "RescheduleCount",
    "AvgDaysRescheduled",
    "EditCount",
    "ViewSignalValue",
    "HasLinkedTask",
    "LinkedTaskReopenCount",
    "LinkedTaskStatusChanges",
    "LinkedTaskCompletionRate",
]

# ── Load ──────────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Dataset not found at {CSV_FILE}. Run generate_dataset.py first."
    )

df = pd.read_csv(CSV_FILE, parse_dates=["StartDate", "EndDate"])
print(f"Loaded {len(df)} rows from {CSV_FILE}.")

# ── Preprocessing ─────────────────────────────────────────────────────────────
df["Hour"]      = df["StartDate"].dt.hour
df["DayOfWeek"] = df["StartDate"].dt.dayofweek

df["DurationMinutes"] = (
    (df["EndDate"] - df["StartDate"]).dt.total_seconds() / 60
).fillna(0).clip(lower=0)

# Boolean columns — guard against missing columns
df["IsRecurring"]   = df["IsRecurring"].fillna(False).astype(int)   if "IsRecurring"   in df.columns else 0
df["HasLinkedTask"] = df["HasLinkedTask"].fillna(False).astype(int)  if "HasLinkedTask" in df.columns else 0

# Importance
importance_map = {"Low": 0, "Medium": 1, "High": 2}
df["Importance"] = df["Importance"].map(importance_map).fillna(1).astype(int)

# Behavioral / task signals
for col, default in [
    ("RescheduleCount",    0),
    ("AvgDaysRescheduled", 0.0),
    ("EditCount",          0),
    ("ViewSignalValue",    0.0),
    ("LinkedTaskReopenCount",    0),
    ("LinkedTaskStatusChanges",  0),
    ("LinkedTaskCompletionRate", 0.0),
]:
    if col in df.columns:
        df[col] = df[col].fillna(default)
    else:
        df[col] = default

# User encoding
le_user = LabelEncoder()
df["UserIdEnc"] = le_user.fit_transform(df["UserId"])

# ── Split ─────────────────────────────────────────────────────────────────────
X = df[FEATURES]
y = df["Attended"]

print(f"Class distribution:\n{y.value_counts().to_string()}\n")

# Only stratify when we have enough samples per class
min_class_count = y.value_counts().min()
stratify = y if min_class_count >= 5 else None
if stratify is None:
    print("WARNING: Not enough samples per class for stratified split — splitting without stratify.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ── Train ─────────────────────────────────────────────────────────────────────
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ── Feature importances ───────────────────────────────────────────────────────
print("Feature importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<30} {imp:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(DATA_FOLDER, exist_ok=True)
joblib.dump(model,   MODEL_FILE)
joblib.dump(le_user, ENCODER_FILE)
print(f"\n✅ Model saved to {MODEL_FILE}")
print(f"✅ Encoder saved to {ENCODER_FILE}")