import pandas as pd
import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FOLDER = "data"
MODEL_FILE   = os.path.join(DATA_FOLDER, "calendar_attendance_model.pkl")
ENCODER_FILE = os.path.join(DATA_FOLDER, "label_encoder_user.pkl")

# ── Ordered feature list (must match train script exactly) ────────────────────
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

# ── Lazy-load model & encoder (don't crash on startup) ────────────────────────
_model   = None
_le_user = None


def is_model_loaded() -> bool:
    return _model is not None and _le_user is not None


def _load_if_needed():
    global _model, _le_user
    if _model is not None:
        return
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        print("Loading model...")
        _model   = joblib.load(MODEL_FILE)
        _le_user = joblib.load(ENCODER_FILE)
        print("Model and encoder loaded.")
    else:
        print(
            "WARNING: No trained model found. "
            "Run train_model_from_csv.py before calling /predict."
        )


# ── Safe column accessor ──────────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str, default) -> pd.Series:
    """Return the column if it exists, otherwise a Series filled with default."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


# ── Feature engineering ───────────────────────────────────────────────────────

def prepare_features(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()

    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df["EndDate"]   = pd.to_datetime(df["EndDate"])

    # Temporal
    df["Hour"]      = df["StartDate"].dt.hour
    df["DayOfWeek"] = df["StartDate"].dt.dayofweek

    # Duration (authoritative: always recalculate from dates)
    df["DurationMinutes"] = (
        (df["EndDate"] - df["StartDate"]).dt.total_seconds() / 60
    ).fillna(0).clip(lower=0)

    # Boolean → int
    df["IsRecurring"]  = _safe_col(df, "IsRecurring",  False).astype(int)
    df["HasLinkedTask"] = _safe_col(df, "HasLinkedTask", False).astype(int)

    df["Importance"] = df["Importance"].fillna(1).astype(int)

    # Behavioral signals
    df["RescheduleCount"]    = _safe_col(df, "RescheduleCount",    0)
    df["AvgDaysRescheduled"] = _safe_col(df, "AvgDaysRescheduled", 0.0)
    df["EditCount"]          = _safe_col(df, "EditCount",          0)
    df["ViewSignalValue"]    = _safe_col(df, "ViewSignalValue",    0.0)

    # Task signals
    df["LinkedTaskReopenCount"]    = _safe_col(df, "LinkedTaskReopenCount",    0)
    df["LinkedTaskStatusChanges"]  = _safe_col(df, "LinkedTaskStatusChanges",  0)
    df["LinkedTaskCompletionRate"] = _safe_col(df, "LinkedTaskCompletionRate", 0.0)

    # User encoding — handle unseen users without corrupting the encoder
    _load_if_needed()
    le = _le_user  # may still be None during an initial train call (handled in train_model)

    if le is not None:
        known_users = set(le.classes_)
        new_users   = sorted(set(df["UserId"]) - known_users)
        if new_users:
            # Keep classes sorted so transform() works correctly
            le.classes_ = np.array(sorted(np.append(le.classes_, new_users)))
        df["UserIdEnc"] = le.transform(df["UserId"])
    else:
        # Fallback during the very first training run (before any encoder exists)
        tmp = LabelEncoder()
        df["UserIdEnc"] = tmp.fit_transform(df["UserId"])

    return df[FEATURES]


# ── Public API ────────────────────────────────────────────────────────────────

def predict(events: pd.DataFrame) -> np.ndarray:
    """Return attendance probability (0–1) for each row."""
    _load_if_needed()
    if _model is None:
        raise RuntimeError("Model is not loaded. Train first.")
    X = prepare_features(events)
    return _model.predict_proba(X)[:, 1]


def train_model(events: pd.DataFrame, labels: pd.Series):
    """Re-fit (full re-train) with the supplied data and persist to disk."""
    global _model, _le_user

    _load_if_needed()  # load existing model/encoder if available

    # If there's no encoder yet, build one from scratch
    if _le_user is None:
        _le_user = LabelEncoder()
        _le_user.fit(events["UserId"])

    X = prepare_features(events)

    if _model is None:
        # First-ever training: build a fresh classifier
        from sklearn.ensemble import GradientBoostingClassifier
        _model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
        )

    _model.fit(X, labels)

    os.makedirs(DATA_FOLDER, exist_ok=True)
    joblib.dump(_model,   MODEL_FILE)
    joblib.dump(_le_user, ENCODER_FILE)
    print(f"Model retrained on {len(labels)} samples and saved.")