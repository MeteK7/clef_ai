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
    model = joblib.load(MODEL_FILE)
    le_user = joblib.load(ENCODER_FILE)
else:
    raise FileNotFoundError("Trained model or encoder not found. Run train_model_from_csv.py first.")

# --- Safe feature preparation ---
def prepare_features(events: pd.DataFrame):
    df = events.copy()

    # Convert dates to datetime if not already
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])

    # Time features
    df['Hour'] = df['StartDate'].dt.hour
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek

    # Boolean to int
    df['IsRecurring'] = df['IsRecurring'].astype(int)

    # Map Importance safely
    importance_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Importance'] = df['Importance'].map(importance_map).fillna(0).astype(int)  # Unknown → Low

    # Encode UserId safely
    known_users = set(le_user.classes_)
    new_users = set(df['UserId']) - known_users

    if new_users:
        # Extend LabelEncoder classes with new users
        le_user.classes_ = np.append(le_user.classes_, list(new_users))

    df['UserIdEnc'] = le_user.transform(df['UserId'])

    return df[['Hour', 'DayOfWeek', 'IsRecurring', 'Importance', 'UserIdEnc']]

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