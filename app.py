import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
from model import predict, train_model, is_model_loaded, _load_if_needed
from datetime import datetime
from pydantic import BaseModel, Field

app = FastAPI(title="Smart Calendar AI")

TRAIN_API_KEY = os.environ.get("AI_TRAIN_API_KEY")
PREDICT_API_KEY = os.environ.get("AI_PREDICT_API_KEY")

MAX_PREDICT_BATCH_SIZE = 500


def verify_train_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    # Fail closed: an unset secret rejects every request rather than allowing all of them.
    if not TRAIN_API_KEY or x_api_key != TRAIN_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


def verify_predict_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    # Fail closed, same as verify_train_api_key.
    if not PREDICT_API_KEY or x_api_key != PREDICT_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")

@app.on_event("startup")
def load_model():
    print("🔄 Loading model at startup...")
    try:
        _load_if_needed()
        print("✅ Model loaded:", is_model_loaded())
    except Exception as e:
        print("⚠️ Model load failed:", e)

# CORS: this service is only ever called server-to-server, from ClefCraft-Backend's
# own HttpClient - never directly from a browser - so no origin needs cross-origin
# access here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic models
# -----------------------------

class CalendarEventInput(BaseModel):
    UserId: str = Field(alias="userId")
    EventId: int = Field(alias="eventId")

    StartDate: datetime = Field(alias="startDate")
    EndDate: datetime = Field(alias="endDate")

    DurationMinutes: float = Field(alias="durationMinutes")
    HourOfDay: int = Field(alias="hourOfDay")
    DayOfWeek: int = Field(alias="dayOfWeek")

    Importance: int = Field(alias="importance")
    IsRecurring: bool = Field(alias="isRecurring")

    RescheduleCount: int = Field(default=0, alias="rescheduleCount")
    AvgDaysRescheduled: float = Field(default=0.0, alias="avgDaysRescheduled")
    EditCount: int = Field(default=0, alias="editCount")
    ViewSignalValue: float = Field(default=0.0, alias="viewSignalValue")

    HasLinkedTask: bool = Field(default=False, alias="hasLinkedTask")
    LinkedTaskReopenCount: int = Field(default=0, alias="linkedTaskReopenCount")
    LinkedTaskStatusChanges: int = Field(default=0, alias="linkedTaskStatusChanges")
    LinkedTaskCompletionRate: float = Field(default=0.0, alias="linkedTaskCompletionRate")

    model_config = {"populate_by_name": True}


class TrainDataInput(BaseModel):
    events: List[CalendarEventInput]
    labels: List[int]


# -----------------------------
# Helpers
# -----------------------------

def events_to_df(events: List[CalendarEventInput]) -> pd.DataFrame:
    df = pd.DataFrame([e.model_dump() for e in events])

    df["StartDate"] = pd.to_datetime(df["StartDate"], utc=True).dt.tz_convert(None)
    df["EndDate"]   = pd.to_datetime(df["EndDate"],   utc=True).dt.tz_convert(None)

    return df


# -----------------------------
# Endpoints
# -----------------------------

import sklearn

@app.get("/version")
def version():
    return {
        "sklearn": sklearn.__version__
    }
    
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": is_model_loaded()}


@app.post("/predict", dependencies=[Depends(verify_predict_api_key)])
def predict_endpoint(events: List[CalendarEventInput]):
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model_from_csv.py first."
        )

    if not events:
        raise HTTPException(status_code=400, detail="Event list is empty.")

    if len(events) > MAX_PREDICT_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large: {len(events)} events (max {MAX_PREDICT_BATCH_SIZE})."
        )

    df = events_to_df(events)
    probs = predict(df)

    return {"predictions": probs.tolist()}


@app.post("/train", dependencies=[Depends(verify_train_api_key)])
def train_endpoint(data: TrainDataInput):
    if len(data.events) != len(data.labels):
        raise HTTPException(
            status_code=400,
            detail="events and labels lists must be the same length."
        )

    if not data.events:
        raise HTTPException(status_code=400, detail="Event list is empty.")

    df = events_to_df(data.events)
    train_model(df, pd.Series(data.labels))

    return {"status": "trained", "samples": len(data.labels)}