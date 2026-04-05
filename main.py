import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    raise RuntimeError("model.pkl not found. Run train.py first.")

app = FastAPI(
    title="Glintt Interview Prep",
    description="Predicts whether social media has a Positive, Neutral, or Negative impact on a student.",
    version="1.0.0",
)

class StudentInput(BaseModel):
    Age: int                         = Field(..., example=21)
    Gender: str                      = Field(..., example="Male")
    Academic_Level: str              = Field(..., example="Undergraduate")
    Country: str                     = Field(..., example="USA")
    Avg_Daily_Usage_Hours: float     = Field(..., example=4.5)
    Most_Used_Platform: str          = Field(..., example="Instagram")
    Affects_Academic_Performance: str = Field(..., example="Yes")
    Sleep_Hours_Per_Night: float     = Field(..., example=6.5)
    Mental_Health_Score: float       = Field(..., example=5.0)

class PredictionOutput(BaseModel):
    prediction: str
    probabilities: dict[str, float]

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Social Media Health Impact API is running."}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: StudentInput):
    """
    Predict the overall impact of social media on a student's health.
 
    Returns:
    - **prediction**: "Positive", "Neutral", or "Negative"
    - **probabilities**: confidence scores for each class
    """
    #the pipeline expects a DataFrame so conversion is necessary
    input_df = pd.DataFrame([data.model_dump()])
 
    try:
        prediction   = model.predict(input_df)[0]
        proba_values = model.predict_proba(input_df)[0]
        classes      = model.classes_
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
 
    probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba_values)}
 
    return PredictionOutput(
        prediction=prediction,
        probabilities=probabilities,
    )