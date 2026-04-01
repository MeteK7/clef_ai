import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

# --- File paths ---
DATA_FOLDER = "data"
CSV_FILE = os.path.join(DATA_FOLDER, "dummy_calendar_events.csv")
MODEL_FILE = os.path.join(DATA_FOLDER, "calendar_attendance_model.pkl")
ENCODER_FILE = os.path.join(DATA_FOLDER, "label_encoder_user.pkl")

# --- Load dataset ---
df = pd.read_csv(CSV_FILE, parse_dates=['StartDate', 'EndDate'])

# --- Basic preprocessing ---
df['Hour'] = df['StartDate'].dt.hour
df['DayOfWeek'] = df['StartDate'].dt.dayofweek
df['IsRecurring'] = df['IsRecurring'].astype(int)
df['Importance'] = df['Importance'].map({'Low':0, 'Medium':1, 'High':2})

# --- Encode UserId ---
le_user = LabelEncoder()
df['UserIdEnc'] = le_user.fit_transform(df['UserId'])

# --- Features & labels ---
FEATURES = ['Hour', 'DayOfWeek', 'IsRecurring', 'Importance', 'UserIdEnc']
X = df[FEATURES]
y = df['Attended']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Train model ---
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Save model & encoder ---
joblib.dump(model, MODEL_FILE)
joblib.dump(le_user, ENCODER_FILE)

# --- Optional: Feature importance ---
importances = model.feature_importances_
for f, imp in zip(FEATURES, importances):
    print(f"Feature: {f}, Importance: {imp:.3f}")

print("\n✅ Model training complete. Model and encoder saved in 'data/' folder.")