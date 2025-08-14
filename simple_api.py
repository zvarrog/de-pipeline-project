from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import List
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kindle Rating Prediction API",
    description="Predict Amazon Kindle review ratings based on text",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
vectorizer = None


def load_model():
    global model, vectorizer

    model_path = "models/real_amazon_model.pkl"
    vectorizer_path = "models/real_amazon_vectorizer.pkl"

    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            logger.info("Model and vectorizer loaded successfully")
        else:
            logger.error(
                f"Model files not found: {model_path}, {vectorizer_path}")
            raise FileNotFoundError("Model files not found")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e


load_model()


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    predicted_rating: float
    confidence: float
    probabilities: dict


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


@app.get("/")
async def root():
    return {"message": "Kindle Rating Prediction API", "status": "active"}


@app.get("/health")
async def health_check():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_rating(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        text_vectorized = vectorizer.transform([request.text])
        probabilities = model.predict_proba(text_vectorized)[0]
        predicted_class = model.predict(text_vectorized)[0]
        confidence = max(probabilities)

        prob_dict = {f"rating_{i+1}": float(prob)
                     for i, prob in enumerate(probabilities)}

        return PredictionResponse(
            predicted_rating=float(predicted_class),
            confidence=float(confidence),
            probabilities=prob_dict
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        texts_vectorized = vectorizer.transform(request.texts)
        probabilities_batch = model.predict_proba(texts_vectorized)
        predictions_batch = model.predict(texts_vectorized)

        results = []
        for i, (text, probs, pred) in enumerate(zip(request.texts, probabilities_batch, predictions_batch)):
            confidence = max(probs)
            prob_dict = {f"rating_{j+1}": float(prob)
                         for j, prob in enumerate(probs)}

            results.append(PredictionResponse(
                predicted_rating=float(pred),
                confidence=float(confidence),
                probabilities=prob_dict
            ))

        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "features": vectorizer.get_feature_names_out()[:10].tolist(),
        "total_features": len(vectorizer.get_feature_names_out()),
        "classes": model.classes_.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
