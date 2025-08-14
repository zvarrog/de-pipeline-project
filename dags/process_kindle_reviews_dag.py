from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

sys.path.append('/workspaces/de-pipeline-project')

from spark_processing import SparkDataProcessor
from torch_model import TorchModelTrainer
from mlflow_integration import ExperimentRunner


default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}


def run_data_validation():
    """Validate input data exists and has correct format"""
    data_path = '/workspaces/de-pipeline-project/data/original/kindle_reviews.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Data validation passed: {data_path}")
    return True


def run_spark_processing():
    """Execute Spark data processing pipeline"""
    processor = SparkDataProcessor()
    
    try:
        df = processor.load_data("data/original/kindle_reviews.csv")
        df_clean = processor.clean_data(df)
        df_features = processor.feature_engineering(df_clean)
        df_analyzed = processor.analyze_data(df_features)
        df_ml, pipeline_model = processor.prepare_ml_features(df_analyzed)
        spark_model, accuracy = processor.train_spark_model(df_ml)
        processor.save_processed_data(df_features, "data/processed")
        
        print(f"Spark processing completed. Accuracy: {accuracy:.3f}")
        return accuracy
        
    finally:
        processor.stop()


def run_torch_training():
    """Execute PyTorch model training"""
    trainer = TorchModelTrainer()
    
    texts, ratings = trainer.load_data('data/original/kindle_reviews.csv', sample_size=30000)
    train_loader, test_loader = trainer.prepare_data_loaders(texts, ratings, batch_size=8)
    trainer.train_model(train_loader, test_loader, epochs=2)
    
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/torch_rating_model.pth')
    
    print("PyTorch training completed")


def run_mlflow_experiments():
    """Execute MLflow experiment tracking"""
    runner = ExperimentRunner()
    
    runner.load_data('data/original/kindle_reviews.csv', sample_size=30000)
    results = runner.run_sklearn_experiments()
    
    print("MLflow experiments completed")
    for model_name, accuracy in results.items():
        print(f"  {model_name}: {accuracy:.4f}")
    
    runner.mlflow_manager.compare_models()
    runner.save_production_model()
    
    return results


def validate_outputs():
    """Validate all pipeline outputs"""
    required_files = [
        'models/production_model.pkl',
        'models/vectorizer.pkl',
        'models/torch_rating_model.pth'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing output files: {missing_files}")
    
    print("All pipeline outputs validated successfully")
    return True


dag = DAG(
    'kindle_reviews_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for Kindle reviews rating prediction',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'spark', 'pytorch', 'mlflow']
)

validate_data = PythonOperator(
    task_id='validate_input_data',
    python_callable=run_data_validation,
    dag=dag
)

spark_processing = PythonOperator(
    task_id='spark_data_processing',
    python_callable=run_spark_processing,
    dag=dag
)

torch_training = PythonOperator(
    task_id='pytorch_model_training',
    python_callable=run_torch_training,
    dag=dag
)

mlflow_experiments = PythonOperator(
    task_id='mlflow_experiment_tracking',
    python_callable=run_mlflow_experiments,
    dag=dag
)

validate_outputs_task = PythonOperator(
    task_id='validate_pipeline_outputs',
    python_callable=validate_outputs,
    dag=dag
)

cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='find /tmp -name "*.tmp" -delete || true',
    dag=dag
)

validate_data >> spark_processing
validate_data >> torch_training
validate_data >> mlflow_experiments

[spark_processing, torch_training, mlflow_experiments] >> validate_outputs_task >> cleanup_task
