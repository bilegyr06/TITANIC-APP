import pickle
import numpy as np
import os

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse 

app = FastAPI(title="Titanic Survival Predictor")

# Templates (place your index.html in a 'templates' folder)
templates = Jinja2Templates(directory="templates")

# =====================
# LOAD MODEL
# =====================
MODEL_FILENAME = 'titanicmodel.pkl'
model = None

if os.path.exists(MODEL_FILENAME):
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
else:
    print(f"Error: '{MODEL_FILENAME}' not found. Please train and save the model first.")

# =====================
# ROUTES
# =====================
@app.get("") 
async def redirect_root(): 
    return RedirectResponse(url="/")

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
        # Create feature array
        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare]])
        
        # Predict
        prediction = model.predict(features)[0]
        output = 'Survived' if prediction == 1 else 'Did Not Survive'
        
        prediction_text = f"Prediction: The passenger likely {output}"
        
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