import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from datetime import datetime


class MLflowManager:
    def __init__(self, experiment_name="kindle_reviews_rating_prediction"):
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"MLflow experiment: {experiment_name}")
    
    def log_sklearn_experiment(self, model, X_test, y_test, params, model_name):
        with mlflow.start_run(run_name=f"sklearn_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(params)
            
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("test_samples", len(y_test))
            
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"sklearn_{model_name}_rating_predictor"
            )
            
            report = classification_report(y_test, predictions, output_dict=True)
            for rating, metrics in report.items():
                if rating.isdigit():
                    mlflow.log_metric(f"precision_rating_{rating}", metrics['precision'])
                    mlflow.log_metric(f"recall_rating_{rating}", metrics['recall'])
                    mlflow.log_metric(f"f1_rating_{rating}", metrics['f1-score'])
            
            print(f"Logged sklearn experiment. Accuracy: {accuracy:.4f}")
            return accuracy
    
    def log_pytorch_experiment(self, model, accuracy, params, model_path):
        with mlflow.start_run(run_name=f"pytorch_distilbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="pytorch_distilbert_rating_predictor"
            )
            
            print(f"Logged PyTorch experiment. Accuracy: {accuracy:.4f}")
    
    def compare_models(self):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print("No runs found")
            return
        
        print("Model comparison:")
        print(runs[['run_id', 'metrics.accuracy', 'tags.mlflow.runName']].sort_values('metrics.accuracy', ascending=False))
        
        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        print(f"\nBest model: {best_run['tags.mlflow.runName']} (Accuracy: {best_run['metrics.accuracy']:.4f})")
        
        return best_run
    
    def load_best_model(self, model_type="sklearn"):
        best_run = self.compare_models()
        
        if model_type in best_run['tags.mlflow.runName']:
            model_uri = f"runs:/{best_run['run_id']}/model"
            
            if model_type == "sklearn":
                model = mlflow.sklearn.load_model(model_uri)
            elif model_type == "pytorch":
                model = mlflow.pytorch.load_model(model_uri)
            
            print(f"Loaded best {model_type} model")
            return model
        
        print(f"No {model_type} model found")
        return None


class ExperimentRunner:
    def __init__(self):
        self.mlflow_manager = MLflowManager()
        self.data = None
        self.vectorizer = None
    
    def load_data(self, file_path, sample_size=50000):
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        df = df.dropna(subset=['reviewText', 'overall'])
        df = df[df['reviewText'].str.len() > 10]
        
        self.data = df[['reviewText', 'overall']]
        print(f"Loaded {len(self.data)} samples")
    
    def prepare_features(self, max_features=5000):
        print("Preparing features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        X = self.vectorizer.fit_transform(self.data['reviewText'])
        y = self.data['overall'].astype(int)
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def run_sklearn_experiments(self):
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        models = {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {'max_iter': 1000, 'solver': 'lbfgs', 'random_state': 42}
            },
            'logistic_regression_balanced': {
                'model': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
                'params': {'max_iter': 1000, 'class_weight': 'balanced', 'solver': 'lbfgs', 'random_state': 42}
            },
            'logistic_regression_l1': {
                'model': LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', random_state=42),
                'params': {'max_iter': 1000, 'penalty': 'l1', 'solver': 'liblinear', 'random_state': 42}
            }
        }
        
        results = {}
        
        for name, config in models.items():
            print(f"Training {name}...")
            model = config['model']
            model.fit(X_train, y_train)
            
            accuracy = self.mlflow_manager.log_sklearn_experiment(
                model, X_test, y_test, config['params'], name
            )
            
            results[name] = accuracy
        
        return results
    
    def save_production_model(self):
        best_model = self.mlflow_manager.load_best_model("sklearn")
        
        if best_model and self.vectorizer:
            os.makedirs('models', exist_ok=True)
            
            with open('models/production_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            
            with open('models/vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            print("Production model saved")
    
    def test_production_model(self):
        try:
            with open('models/production_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            with open('models/vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            
            test_reviews = [
                "This book is absolutely amazing! Best read of the year.",
                "Terrible book, complete waste of time and money.",
                "It was okay, nothing special but not bad either."
            ]
            
            X_test = vectorizer.transform(test_reviews)
            predictions = model.predict(X_test)
            
            print("Production model test:")
            for review, rating in zip(test_reviews, predictions):
                print(f"  '{review[:50]}...' -> {rating} stars")
            
        except FileNotFoundError:
            print("Production model not found")


def main():
    runner = ExperimentRunner()
    
    try:
        runner.load_data('data/original/kindle_reviews.csv')
        results = runner.run_sklearn_experiments()
        
        print("\nExperiment results:")
        for model_name, accuracy in results.items():
            print(f"  {model_name}: {accuracy:.4f}")
        
        runner.mlflow_manager.compare_models()
        runner.save_production_model()
        runner.test_production_model()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
