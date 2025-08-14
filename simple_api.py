from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging
import os
from typing import List, Dict, Any
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kindle Reviews Rating Prediction API",
    description="ML API for predicting ratings from Amazon Kindle review text using multiple ML approaches",
    version="2.0.0"
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

class ReviewRequest(BaseModel):
    text: str = Field(..., description="Review text to analyze", min_length=1)

class ReviewBatchRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of review texts", max_items=100)

class PredictionResponse(BaseModel):
    rating: float = Field(..., description="Predicted rating (1-5 stars)")
    confidence: float = Field(..., description="Model confidence (0-1)")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_features: int

def load_model():
    global model, vectorizer
    
    try:
        # Try to load production models
        model_files = [
            ('models/production_model.pkl', 'models/vectorizer.pkl'),
            ('models/real_amazon_model.pkl', 'models/real_amazon_vectorizer.pkl'),
            ('models/sklearn_rating_model.pkl', 'models/sklearn_rating_vectorizer.pkl')
        ]
        
        model_loaded = False
        for model_path, vec_path in model_files:
            if os.path.exists(model_path) and os.path.exists(vec_path):
                model = joblib.load(model_path)
                vectorizer = joblib.load(vec_path)
                logger.info(f"Model loaded successfully from {model_path}")
                model_loaded = True
                break
        
        if not model_loaded:
            logger.warning("No pre-trained model found. Training new model...")
            train_new_model()
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        train_new_model()

def train_new_model():
    """Train a new model if no existing model is found"""
    global model, vectorizer
    
    try:
        logger.info("Training new model with sample data...")
        
        # Create sample training data
        sample_data = [
            ("This book is absolutely amazing! I loved every single page.", 5),
            ("Great story, well written and engaging throughout.", 5),
            ("Really enjoyed this book, highly recommend it.", 4),
            ("Good book, worth reading but not exceptional.", 4),
            ("It was okay, nothing special but readable.", 3),
            ("Average story, some parts were interesting.", 3),
            ("Not great, had some issues with the plot.", 2),
            ("Disappointing book, expected much better.", 2),
            ("Terrible book, complete waste of time.", 1),
            ("Awful writing, couldn't finish it.", 1)
        ]
        
        # Expand sample data
        expanded_data = sample_data * 10
        texts, ratings = zip(*expanded_data)
        
        # Train model
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        model = LogisticRegression(random_state=42)
        model.fit(X, ratings)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/sample_model.pkl')
        joblib.dump(vectorizer, 'models/sample_vectorizer.pkl')
        
        logger.info("Sample model trained and saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to train sample model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Kindle Reviews Rating Prediction API",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_features = getattr(vectorizer, 'max_features', 0)
    if hasattr(vectorizer, 'vocabulary_'):
        total_features = len(vectorizer.vocabulary_)
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        total_features=total_features
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_rating(request: ReviewRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Vectorize text
        text_features = vectorizer.transform([request.text])
        
        # Make prediction
        prediction = model.predict(text_features)[0]
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_features)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 0.8  # Default confidence
        
        return PredictionResponse(
            rating=float(prediction),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_ratings_batch(request: ReviewBatchRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Too many reviews (max 100)")
    
    try:
        # Vectorize all texts
        text_features = vectorizer.transform(request.reviews)
        
        # Make predictions
        predictions = model.predict(text_features)
        
        # Get confidences
        confidences = []
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_features)
            confidences = [float(np.max(probs)) for probs in probabilities]
        else:
            confidences = [0.8] * len(predictions)  # Default confidence
        
        # Format results
        results = []
        for pred, conf in zip(predictions, confidences):
            results.append(PredictionResponse(
                rating=float(pred),
                confidence=conf
            ))
        
        return BatchPredictionResponse(predictions=results)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = {
        "model_type": str(type(model).__name__),
        "model_params": getattr(model, 'get_params', lambda: {})(),
    }
    
    if hasattr(vectorizer, 'max_features'):
        model_info["vectorizer_max_features"] = vectorizer.max_features
    if hasattr(vectorizer, 'vocabulary_'):
        model_info["vocabulary_size"] = len(vectorizer.vocabulary_)
    if hasattr(model, 'classes_'):
        model_info["model_classes"] = list(model.classes_)
    
    return model_info

@app.get("/test")
async def test_predictions():
    """Test endpoint with sample predictions"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    test_reviews = [
        "This book is absolutely amazing! I loved every page.",
        "Terrible book, waste of time and money.",
        "It was okay, nothing special but readable.",
        "Great story but the ending was disappointing.",
        "Perfect! Exactly what I was looking for."
    ]
    
    results = []
    for review in test_reviews:
        response = await predict_rating(ReviewRequest(text=review))
        results.append({
            "review": review,
            "predicted_rating": response.rating,
            "confidence": response.confidence
        })
    
    return {"test_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
