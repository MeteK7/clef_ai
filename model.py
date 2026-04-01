import pandas as pd
import os
import joblib
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

# --- Convert CalendarEvent to ML features ---
def prepare_features(events: pd.DataFrame):
    df = events.copy()
    df['Hour'] = df['StartDate'].dt.hour
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek
    df['IsRecurring'] = df['IsRecurring'].astype(int)
    df['Importance'] = df['Importance'].map({'Low':0, 'Medium':1, 'High':2})
    
    # Encode UserId
    df['UserIdEnc'] = le_user.transform(df['UserId'])
    
    return df[['Hour', 'DayOfWeek', 'IsRecurring', 'Importance', 'UserIdEnc']]

# --- Predict attendance probability ---
def predict(events: pd.DataFrame):
    X = prepare_features(events)
    return model.predict_proba(X)[:,1]  # probability of attending

# --- Optional: train model with new data ---
def train_model(events: pd.DataFrame, labels: pd.Series):
    global model
    X = prepare_features(events)
    model.fit(X, labels)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le_user, ENCODER_FILE)