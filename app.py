import pickle
import numpy as np
import pandas as pd   # Added for DataFrame input (fixes warning)
import os

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI(title="Titanic Survival Predictor")

# Templates
templates = Jinja2Templates(directory="templates")

# =====================
# LOAD MODEL
# =====================
MODEL_FILENAME = './model/titanic_survival_model.pkl'
model = None

if os.path.exists(MODEL_FILENAME):
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
else:
    print(f"Error: '{MODEL_FILENAME}' not found. Please train and save the model first.")

# =====================
# ROUTES
# =====================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction_text": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  Pclass: int = Form(...),
                  Sex: int = Form(...),      # 0 for male, 1 for female
                  Age: float = Form(...),
                  SibSp: int = Form(...),
                  Parch: int = Form(...),
                  Fare: float = Form(...)):
    
    if model is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": "Error: Model not loaded."}
        )
    
    try:
        # Create DataFrame with exact feature names (eliminates warning)
        input_df = pd.DataFrame([{
            'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare
        }])
        
        # Predict class and probability
        prediction = model.predict(input_df)[0]
        survival_prob = model.predict_proba(input_df)[0][1] * 100  # % chance of Survived (class 1)
        
        output = 'Survived' if prediction == 1 else 'Did Not Survive'
        prediction_text = f"Prediction: The passenger likely {output} ({survival_prob:.1f}% confidence)"
        
    except Exception as e:
        prediction_text = f"Error during prediction: {str(e)}"
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction_text": prediction_text}
    )

# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)