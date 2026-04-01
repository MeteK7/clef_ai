import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_FILE = "calendar_attendance_model.pkl"

# Initialize or load model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    le_user = joblib.load("label_encoder_user.pkl")
else:
    model = GradientBoostingClassifier()
    le_user = LabelEncoder()

# Convert CalendarEvent to ML features
def prepare_features(events: pd.DataFrame):
    df = events.copy()
    df['Hour'] = df['StartDate'].dt.hour
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek
    df['IsRecurring'] = df['IsRecurring'].astype(int)
    df['Importance'] = df['Importance'].map({'Low':0, 'Medium':1, 'High':2})
    
    # Encode UserId
    if not hasattr(le_user, 'classes_') or len(le_user.classes_) == 0:
        df['UserIdEnc'] = le_user.fit_transform(df['UserId'])
    else:
        df['UserIdEnc'] = le_user.transform(df['UserId'])
    
    features = df[['Hour', 'DayOfWeek', 'IsRecurring', 'Importance', 'UserIdEnc']]
    return features

# Train model with new batch
def train_model(events: pd.DataFrame, labels: pd.Series):
    global model
    X = prepare_features(events)
    model.fit(X, labels)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le_user, "label_encoder_user.pkl")

# Predict attendance probability
def predict(events: pd.DataFrame):
    X = prepare_features(events)
    return model.predict_proba(X)[:,1]  # probability of attending