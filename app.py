from fastapi import FastAPI
from typing import List
import pandas as pd
from model import predict, train_model
from datetime import datetime
from pydantic import BaseModel, Field

app = FastAPI(title="Smart Calendar AI")

class CalendarEventInput(BaseModel):
    UserId: str = Field(alias="userId")
    EventId: int = Field(alias="eventId")

    StartDate: datetime = Field(alias="startDate")
    EndDate: datetime = Field(alias="endDate")

    DurationMinutes: float = Field(alias="durationMinutes")
    HourOfDay: int = Field(alias="hourOfDay")
    DayOfWeek: int = Field(alias="dayOfWeek")

    Importance: str = Field(alias="importance")
    IsRecurring: bool = Field(alias="isRecurring")

    RescheduleCount: int = Field(default=0, alias="rescheduleCount")
    AvgDaysRescheduled: float = Field(default=0, alias="avgDaysRescheduled")
    EditCount: int = Field(default=0, alias="editCount")
    ViewSignalValue: float = Field(default=0, alias="viewSignalValue")

    HasLinkedTask: bool = Field(default=False, alias="hasLinkedTask")
    LinkedTaskReopenCount: int = Field(default=0, alias="linkedTaskReopenCount")
    LinkedTaskStatusChanges: int = Field(default=0, alias="linkedTaskStatusChanges")
    LinkedTaskCompletionRate: float = Field(default=0, alias="linkedTaskCompletionRate")

    class Config:
        populate_by_name = True

class TrainDataInput(BaseModel):
    events: List[CalendarEventInput]
    labels: List[int]  # 1 = attended, 0 = skipped

@app.post("/predict")
def predict_endpoint(events: List[CalendarEventInput]):
    df = pd.DataFrame([e.dict(by_alias=True) for e in events])
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])
    probs = predict(df)
    return {"predictions": probs.tolist()}

@app.post("/train")
def train_endpoint(data: TrainDataInput):
    df = pd.DataFrame([e.dict(by_alias=True) for e in data.events])
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])
    train_model(df, pd.Series(data.labels))
    return {"status": "trained"}


    