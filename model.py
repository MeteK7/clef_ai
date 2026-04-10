import pandas as pd
import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Paths ---
DATA_FOLDER = "data"
MODEL_FILE = os.path.join(DATA_FOLDER, "calendar_attendance_model.pkl")
ENCODER_FILE = os.path.join(DATA_FOLDER, "label_encoder_user.pkl")

# --- Load trained model ---
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    print("Loading model...")
    model = joblib.load(MODEL_FILE)
    print("Model loaded")
    le_user = joblib.load(ENCODER_FILE)
    print("Encoder loaded")
else:
    raise FileNotFoundError("Trained model or encoder not found. Run train_model_from_csv.py first.")

# --- Safe feature preparation ---
def prepare_features(events: pd.DataFrame):
    df = events.copy()

    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])

    # --- Time features ---
    df['Hour'] = df.get('HourOfDay', df['StartDate'].dt.hour)
    df['DayOfWeek'] = df.get('DayOfWeek', df['StartDate'].dt.dayofweek)

    # --- Duration ---
    df['DurationMinutes'] = (
        (df['EndDate'] - df['StartDate']).dt.total_seconds() / 60
    ).fillna(0)

    # --- Boolean ---
    df['IsRecurring'] = df.get('IsRecurring', False).astype(int)
    df['HasLinkedTask'] = df.get('HasLinkedTask', False).astype(int)

    # --- Importance ---
    importance_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Importance'] = df['Importance'].map(importance_map).fillna(0).astype(int)

    # --- Behavioral signals (safe defaults) ---
    df['RescheduleCount'] = df.get('RescheduleCount', 0).fillna(0)
    df['AvgDaysRescheduled'] = df.get('AvgDaysRescheduled', 0).fillna(0)
    df['EditCount'] = df.get('EditCount', 0).fillna(0)
    df['ViewSignalValue'] = df.get('ViewSignalValue', 0).fillna(0)

    # --- Task signals ---
    df['LinkedTaskReopenCount'] = df.get('LinkedTaskReopenCount', 0).fillna(0)
    df['LinkedTaskStatusChanges'] = df.get('LinkedTaskStatusChanges', 0).fillna(0)
    df['LinkedTaskCompletionRate'] = df.get('LinkedTaskCompletionRate', 0).fillna(0)

    # --- User encoding ---
    known_users = set(le_user.classes_)
    new_users = set(df['UserId']) - known_users

    if new_users:
        le_user.classes_ = np.append(le_user.classes_, list(new_users))

    df['UserIdEnc'] = le_user.transform(df['UserId'])

    return df[[
        'Hour',
        'DayOfWeek',
        'DurationMinutes',
        'IsRecurring',
        'Importance',
        'UserIdEnc',

        # behavioral
        'RescheduleCount',
        'AvgDaysRescheduled',
        'EditCount',
        'ViewSignalValue',

        # task
        'HasLinkedTask',
        'LinkedTaskReopenCount',
        'LinkedTaskStatusChanges',
        'LinkedTaskCompletionRate'
    ]]

# --- Predict attendance probability ---
def predict(events: pd.DataFrame):
    X = prepare_features(events)
    return model.predict_proba(X)[:,1]  # probability of attending

# --- Train model with new data ---
def train_model(events: pd.DataFrame, labels: pd.Series):
    global model, le_user
    X = prepare_features(events)
    model.fit(X, labels)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le_user, ENCODER_FILE)