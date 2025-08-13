"""
ML Pipeline для анализа sentiment и рекомендаций
"""
import os
import mlflow
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


def setup_mlflow():
    """Настройка MLflow tracking"""
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment("kindle_reviews_sentiment")


def prepare_features(df):
    """Feature engineering для ML"""
    # Простой sentiment analysis на основе рейтинга
    df['sentiment'] = (df['overall'] >= 4).astype(
        int)  # 4-5 = positive, 1-3 = negative

    # Text features
    df['review_length'] = df['reviewText'].fillna('').str.len()
    df['summary_length'] = df['summary'].fillna('').str.len()
    df['has_summary'] = df['summary'].notna().astype(int)

    # Rating features
    df['is_extreme_rating'] = (
        (df['overall'] == 1) | (df['overall'] == 5)).astype(int)

    return df


def train_sentiment_model(df):
    """Обучение простой модели sentiment analysis"""

    with mlflow.start_run(run_name="sentiment_baseline"):
        # Подготовка данных
        df = prepare_features(df)

        # Убираем строки без текста
        df_clean = df.dropna(subset=['reviewText'])

        # Features
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_text = vectorizer.fit_transform(df_clean['reviewText'])

        # Target
        y = df_clean['sentiment']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )

        # Model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # MLflow logging
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))

        # Save model artifacts
        model_dir = Path("/app/models")
        model_dir.mkdir(exist_ok=True)

        joblib.dump(model, model_dir / "sentiment_model.pkl")
        joblib.dump(vectorizer, model_dir / "vectorizer.pkl")

        mlflow.log_artifact(str(model_dir / "sentiment_model.pkl"))
        mlflow.log_artifact(str(model_dir / "vectorizer.pkl"))

        print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
        print(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        return model, vectorizer, accuracy


def recommendation_features(df):
    """Feature engineering для рекомендательной системы"""

    # User features
    user_stats = df.groupby('reviewerID').agg({
        'overall': ['mean', 'count', 'std'],
        'helpful_ratio': 'mean'
    }).round(3)
    user_stats.columns = [
        'user_avg_rating', 'user_review_count', 'user_rating_std', 'user_helpful_ratio']

    # Item features
    item_stats = df.groupby('asin').agg({
        'overall': ['mean', 'count', 'std'],
        'helpful_ratio': 'mean'
    }).round(3)
    item_stats.columns = [
        'item_avg_rating', 'item_review_count', 'item_rating_std', 'item_helpful_ratio']

    # Merge back
    df = df.merge(user_stats, on='reviewerID', how='left')
    df = df.merge(item_stats, on='asin', how='left')

    return df


if __name__ == "__main__":
    # Setup
    setup_mlflow()

    # Load processed data
    df = pd.read_parquet("/app/output/reviews_processed.parquet")
    print(f"📊 Loaded {len(df)} reviews")

    # Train sentiment model
    model, vectorizer, accuracy = train_sentiment_model(df)

    # Prepare recommendation features
    df_features = recommendation_features(df)
    df_features.to_parquet("/app/output/reviews_with_ml_features.parquet")

    print("🎯 ML Pipeline completed!")
    print(f"   - Sentiment model accuracy: {accuracy:.3f}")
    print(f"   - Features saved to: /app/output/reviews_with_ml_features.parquet")
    print(f"   - MLflow tracking: /app/mlruns")
