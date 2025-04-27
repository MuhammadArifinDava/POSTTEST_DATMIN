from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle

app = FastAPI(title="Total Spend Prediction API (XGBoost with Scaler)")


with open("XGBoost_Model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

class CustomerData(BaseModel):
    Gender: int
    Age: int
    City: int
    Membership_Type: int = Field(..., alias="Membership Type")
    Items_Purchased: int = Field(..., alias="Items Purchased")
    Average_Rating: float = Field(..., alias="Average Rating")
    Discount_Applied: int = Field(..., alias="Discount Applied")
    Days_Since_Last_Purchase: int = Field(..., alias="Days Since Last Purchase")
    Satisfaction_Level: int = Field(..., alias="Satisfaction Level")
    Age_Group: int = Field(..., alias="Age Group")
    Spend_per_Item: float = Field(..., alias="Spend per Item")

    class Config:
        allow_population_by_field_name = True

class PredictionResponse(BaseModel):
    prediction: float

@app.get("/")
def read_root():
    return {"message": "✅ API is running and ready for predictions!"}

def preprocess_input(data: CustomerData):
    df = pd.DataFrame([data.dict(by_alias=True)])
    df_scaled = scaler.transform(df)  
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)  
    return df_scaled

@app.post("/predict", response_model=PredictionResponse)
def predict_spend(data: CustomerData):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    return {"prediction": round(float(prediction), 2)}



