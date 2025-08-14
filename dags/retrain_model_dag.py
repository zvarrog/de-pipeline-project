from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys


default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 14),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_rating_model',
    default_args=default_args,
    description='Model retraining pipeline for rating prediction',
    schedule_interval=timedelta(days=7),
    catchup=False,
    tags=['ml', 'rating', 'retraining'],
)


def run_spark_processing(**context):
    """Execute Spark data processing"""
    print("Starting Spark data processing...")

    sys.path.append('/app')

    try:
        from spark_processing import SparkDataProcessor
        
        processor = SparkDataProcessor()
        df = processor.load_data("/app/data/original/kindle_reviews.csv")
        df_clean = processor.clean_data(df)
        df_features = processor.feature_engineering(df_clean)
        processor.save_processed_data(df_features, "/app/data/processed")
        processor.stop()
        
        print("Spark processing completed successfully")
        return True
        
    except Exception as e:
        print(f"Spark processing failed: {str(e)}")
        raise


def run_pytorch_training(**context):
    """Execute PyTorch model training"""
    print("Starting PyTorch model training...")
    
    sys.path.append('/app')
    
    try:
        from torch_model import TorchModelTrainer
        
        trainer = TorchModelTrainer()
        texts, ratings = trainer.load_data('/app/data/original/kindle_reviews.csv', sample_size=30000)
        train_loader, test_loader = trainer.prepare_data_loaders(texts, ratings, batch_size=8)
        trainer.train_model(train_loader, test_loader, epochs=2)
        
        os.makedirs('/app/models', exist_ok=True)
        trainer.save_model('/app/models/torch_rating_model.pth')
        
        print("PyTorch training completed successfully")
        return True
        
    except Exception as e:
        print(f"PyTorch training failed: {str(e)}")
        raise


def run_mlflow_experiments(**context):
    """Execute MLflow experiment tracking"""
    print("Starting MLflow experiments...")
    
    sys.path.append('/app')
    
    try:
        from mlflow_integration import ExperimentRunner
        
        runner = ExperimentRunner()
        runner.load_data('/app/data/original/kindle_reviews.csv', sample_size=30000)
        results = runner.run_sklearn_experiments()
        runner.save_production_model()
        
        print("MLflow experiments completed successfully")
        for model_name, accuracy in results.items():
            print(f"  {model_name}: {accuracy:.4f}")
        
        return results
        
    except Exception as e:
        print(f"MLflow experiments failed: {str(e)}")
        raise


def load_and_prepare_data(**context):
    """Load and prepare training data"""
    print("Loading training data...")
    
    try:
        df = pd.read_csv('/app/data/original/kindle_reviews.csv')
        df = df.dropna(subset=['reviewText', 'overall'])
        df = df[df['reviewText'].str.len() > 10]
        
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)
        
        df.to_csv('/app/data/prepared_data.csv', index=False)
        print(f"Prepared {len(df)} samples for training")
        
        return True
        
    except Exception as e:
        print(f"Data preparation failed: {str(e)}")
        raise


def train_baseline_model(**context):
    """Train baseline scikit-learn model"""
    print("Training baseline model...")
    
    try:
        df = pd.read_csv('/app/data/prepared_data.csv')
        
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['reviewText'])
        y = df['overall'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        os.makedirs('/app/models', exist_ok=True)
        joblib.dump(model, '/app/models/baseline_model.pkl')
        joblib.dump(vectorizer, '/app/models/baseline_vectorizer.pkl')
        
        print(f"Baseline model trained successfully. Accuracy: {accuracy:.4f}")
        
        with open('/app/models/model_metrics.txt', 'w') as f:
            f.write(f"Baseline Model Accuracy: {accuracy:.4f}\n")
            f.write(f"Training samples: {len(X_train)}\n")
            f.write(f"Test samples: {len(X_test)}\n")
        
        return accuracy
        
    except Exception as e:
        print(f"Baseline model training failed: {str(e)}")
        raise


def validate_models(**context):
    """Validate all trained models"""
    print("Validating trained models...")
    
    required_files = [
        '/app/models/baseline_model.pkl',
        '/app/models/baseline_vectorizer.pkl',
        '/app/models/model_metrics.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing model files: {missing_files}")
    
    print("All models validated successfully")
    
    if os.path.exists('/app/models/model_metrics.txt'):
        with open('/app/models/model_metrics.txt', 'r') as f:
            print("Model metrics:")
            print(f.read())
    
    return True


def cleanup_temp_files(**context):
    """Clean up temporary files"""
    print("Cleaning up temporary files...")
    
    temp_files = [
        '/app/data/prepared_data.csv'
    ]
    
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
    
    print("Cleanup completed")
    return True


prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=load_and_prepare_data,
    dag=dag,
)

spark_task = PythonOperator(
    task_id='spark_data_processing',
    python_callable=run_spark_processing,
    dag=dag,
)

pytorch_task = PythonOperator(
    task_id='pytorch_model_training',
    python_callable=run_pytorch_training,
    dag=dag,
)

mlflow_task = PythonOperator(
    task_id='mlflow_experiment_tracking',
    python_callable=run_mlflow_experiments,
    dag=dag,
)

baseline_task = PythonOperator(
    task_id='train_baseline_model',
    python_callable=train_baseline_model,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    dag=dag,
)

prepare_data_task >> [spark_task, pytorch_task, mlflow_task, baseline_task]
[spark_task, pytorch_task, mlflow_task, baseline_task] >> validate_task >> cleanup_task
