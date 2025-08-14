import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, tokenizer, max_length=512):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rating = self.ratings[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(rating - 1, dtype=torch.long)
        }


class RatingPredictor(nn.Module):
    def __init__(self, n_classes=5, dropout=0.3):
        super(RatingPredictor, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.dim, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.classifier(output)


class TorchModelTrainer:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        print(f"Using device: {self.device}")
    
    def load_data(self, file_path, sample_size=50000):
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        df = df.dropna(subset=['reviewText', 'overall'])
        df = df[df['reviewText'].str.len() > 10]
        
        texts = df['reviewText'].values
        ratings = df['overall'].astype(int).values
        
        print(f"Loaded {len(texts)} samples")
        return texts, ratings
    
    def prepare_data_loaders(self, texts, ratings, batch_size=16, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, ratings, test_size=test_size, random_state=42, stratify=ratings
        )
        
        train_dataset = ReviewDataset(X_train, y_train, self.tokenizer)
        test_dataset = ReviewDataset(X_test, y_test, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train_model(self, train_loader, test_loader, epochs=3, learning_rate=2e-5):
        self.model = RatingPredictor().to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print("Starting training...")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, ratings)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = self.evaluate_model(test_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
    
    def evaluate_model(self, test_loader):
        self.model.eval()
        predictions = []
        actual_ratings = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                actual_ratings.extend(ratings.cpu().numpy())
        
        accuracy = accuracy_score(actual_ratings, predictions)
        return accuracy
    
    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        predictions = []
        
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.item() + 1)
        
        return predictions
    
    def save_model(self, model_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = RatingPredictor().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        print(f"Model loaded from {model_path}")


def main():
    trainer = TorchModelTrainer()
    
    try:
        texts, ratings = trainer.load_data('data/original/kindle_reviews.csv')
        train_loader, test_loader = trainer.prepare_data_loaders(texts, ratings)
        trainer.train_model(train_loader, test_loader)
        
        os.makedirs('models', exist_ok=True)
        trainer.save_model('models/torch_rating_model.pth')
        
        test_texts = [
            "This book is amazing! I loved every page.",
            "Terrible book, waste of time.",
            "It was okay, nothing special."
        ]
        
        predictions = trainer.predict(test_texts)
        print("Sample predictions:", predictions)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
