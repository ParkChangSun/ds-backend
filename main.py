from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_probability = "model_prob.pkl"
model_wounded = "model_wounded.pkl"
model_damage = "model_damage.pkl"


class PredictReqBody(BaseModel):
    process_type: int
    process_type2: int
    orderer: int
    season: int
    day: int
    bid_percent: int
    process: int
    place: int
    total_price: int
    process_price: int
    process_percent: int
    man_count: int
    safety_plan: int
    part: int
    timezone: int
    incident_object2: int

    process_day: int
    total_day: int
    temperature: float
    humidity: float

    protection: bool
    personal_safety: bool


class PredictResBody(BaseModel):
    prob: int
    wounded: int
    damage: int


@app.post("/predict")
async def predict(input: PredictReqBody):
    prob = predictProb(
        [
            input.process_type,
            input.bid_percent,
            input.orderer,
            input.temperature,
            input.humidity,
            input.season,
            input.day,
        ]
    )

    wounded = predictWounded(
        [
            input.personal_safety,
            input.process,
            input.place,
            input.part,
            input.total_price,
            input.process_price,
            input.bid_percent,
            input.process_percent,
            input.man_count,
            input.safety_plan,
            input.total_day,
            input.process_day,
            happyPoint(input.temperature, input.humidity),
            input.day,
            input.season,
            input.timezone,
            input.incident_object2,
        ]
    )

    damage = predictDamage(
        [
            input.humidity,
            input.protection,
            input.process,
            input.place,
            input.total_price,
            input.process_price,
            input.bid_percent,
            input.process_percent,
            input.man_count,
            input.safety_plan,
            input.total_day,
            input.process_day,
            happyPoint(input.temperature, input.humidity),
            input.process_type2,
            input.incident_object2,
        ]
    )

    body = PredictResBody(prob=prob, wounded=wounded, damage=damage)
    json_compatible_item_data = jsonable_encoder(body)
    return JSONResponse(content=json_compatible_item_data)


def predictProb(input):
    model = joblib.load(model_probability)
    X = np.array(input).reshape(1, -1)
    y_pred = model.predict(X)[0]
    return y_pred


def predictWounded(input):
    model = joblib.load(model_wounded)
    X = np.array(input).reshape(1, -1)
    y_pred = model.predict(X)[0][0]
    return y_pred


def predictDamage(input):
    model = joblib.load(model_damage)
    X = np.array(input).reshape(1, -1)
    y_pred = model.predict(X)[0]
    return y_pred


def happyPoint(t, h):
    return 9 / 5 * t - 0.55 * (1 - (h / 100)) * (9 / 5 * t - 26) + 32
