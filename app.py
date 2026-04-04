from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from model import predict, train_model
from datetime import datetime
from pydantic import BaseModel, Field

app = FastAPI(title="Smart Calendar AI")

class CalendarEventInput(BaseModel):
    UserId: str = Field(alias="userId")
    StartDate: datetime = Field(alias="startDate")
    EndDate: datetime = Field(alias="endDate")
    Importance: str = Field(alias="importance")
    IsRecurring: bool = Field(alias="isRecurring")

    class Config:
        populate_by_name = True

class TrainDataInput(BaseModel):
    events: List[CalendarEventInput]
    labels: List[int]  # 1 = attended, 0 = skipped

@app.post("/predict")
def predict_endpoint(events: List[CalendarEventInput]):
    df = pd.DataFrame([e.dict() for e in events])
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])
    probs = predict(df)
    return {"predictions": probs.tolist()}

@app.post("/train")
def train_endpoint(data: TrainDataInput):
    df = pd.DataFrame([e.dict() for e in data.events])
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])
    train_model(df, pd.Series(data.labels))
    return {"status": "trained"}


    