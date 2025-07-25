#!/usr/bin/env python3
"""
GPU-optimized training script for CS2 skin volume prediction
Designed for remote GPU training with experiment tracking
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import wandb
from tqdm import tqdm

from data_processor import SkinDataProcessor

class DeepSkinPredictor(nn.Module):
    """Deep neural network for skin volume prediction"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout=0.3):
        super(DeepSkinPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class GPUSkinTrainer:
    def __init__(self, use_wandb=True, project_name="cs2-skin-prediction"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name)
        
        self.models = {}
        self.best_model = None
        self.best_score = float('-inf')
        
    def train_xgboost_gpu(self, X_train, X_test, y_train, y_test, params=None):
        """Train XGBoost with GPU acceleration"""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',  # GPU acceleration
                'gpu_id': 0,
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        print("Training XGBoost with GPU...")
        model = xgb.XGBRegressor(**params)
        
        # Early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        self.models['xgboost_gpu'] = {
            'model': model,
            'score': score,
            'predictions': y_pred
        }
        
        print(f"XGBoost GPU R² Score: {score:.4f}")
        return model, score
    
    def train_lightgbm_gpu(self, X_train, X_test, y_train, y_test, params=None):
        """Train LightGBM with GPU acceleration"""
        if params is None:
            params = {
                'objective': 'regression',
                'device': 'gpu',  # GPU acceleration
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }
        
        print("Training LightGBM with GPU...")
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        self.models['lightgbm_gpu'] = {
            'model': model,
            'score': score,
            'predictions': y_pred
        }
        
        print(f"LightGBM GPU R² Score: {score:.4f}")
        return model, score
    
    def train_catboost_gpu(self, X_train, X_test, y_train, y_test, params=None):
        """Train CatBoost with GPU acceleration"""
        if params is None:
            params = {
                'task_type': 'GPU',  # GPU acceleration
                'devices': '0',
                'depth': 8,
                'learning_rate': 0.1,
                'iterations': 1000,
                'random_seed': 42,
                'verbose': False
            }
        
        print("Training CatBoost with GPU...")
        model = CatBoostRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        self.models['catboost_gpu'] = {
            'model': model,
            'score': score,
            'predictions': y_pred
        }
        
        print(f"CatBoost GPU R² Score: {score:.4f}")
        return model, score
    
    def train_deep_network(self, X_train, X_test, y_train, y_test, epochs=200, batch_size=256):
        """Train deep neural network"""
        print("Training Deep Neural Network...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = DeepSkinPredictor(X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="Training"):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor).squeeze()
                val_loss = criterion(val_outputs, y_test_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_deep_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if self.use_wandb and epoch % 10 == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss
                })
        
        # Load best model and evaluate
        model.load_state_dict(torch.load('best_deep_model.pth'))
        model.eval()
        
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor).squeeze()
            y_pred = y_pred_tensor.cpu().numpy()
        
        score = r2_score(y_test, y_pred)
        
        self.models['deep_network'] = {
            'model': model,
            'score': score,
            'predictions': y_pred
        }
        
        print(f"Deep Network R² Score: {score:.4f}")
        return model, score
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all available models"""
        print("=" * 60)
        print("Training all GPU-optimized models...")
        print("=" * 60)
        
        # Train gradient boosting models
        try:
            self.train_xgboost_gpu(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"XGBoost GPU training failed: {e}")
        
        try:
            self.train_lightgbm_gpu(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"LightGBM GPU training failed: {e}")
        
        try:
            self.train_catboost_gpu(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"CatBoost GPU training failed: {e}")
        
        # Train deep network
        try:
            self.train_deep_network(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"Deep Network training failed: {e}")
        
        # Find best model
        if self.models:
            best_name = max(self.models.keys(), key=lambda x: self.models[x]['score'])
            self.best_model = self.models[best_name]
            self.best_score = self.best_model['score']
            
            print(f"\nBest model: {best_name} (R² Score: {self.best_score:.4f})")
            
            if self.use_wandb:
                wandb.log({
                    'best_model': best_name,
                    'best_score': self.best_score
                })
        
        return self.models
    
    def save_models(self, save_dir='models'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model_info in self.models.items():
            if name == 'deep_network':
                # Save PyTorch model
                torch.save(model_info['model'].state_dict(), f'{save_dir}/{name}.pth')
            else:
                # Save sklearn-compatible models
                joblib.dump(model_info['model'], f'{save_dir}/{name}.pkl')
        
        # Save metadata
        metadata = {
            'models': {name: info['score'] for name, info in self.models.items()},
            'best_model': max(self.models.keys(), key=lambda x: self.models[x]['score']) if self.models else None,
            'training_date': datetime.now().isoformat()
        }
        
        with open(f'{save_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {save_dir}/")

def main():
    """Main training function"""
    print("CS2 Skin Volume Prediction - GPU Training")
    print("=" * 50)
    
    # Initialize components
    processor = SkinDataProcessor()
    trainer = GPUSkinTrainer()
    
    # Load data
    print("Loading JSON data...")
    df = processor.load_single_json('22313.json')
    
    if df is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(df)} records")
    
    # Prepare data
    try:
        X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(df)
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return
    
    # Train models
    models = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save results
    trainer.save_models()
    
    # Save scaler for future predictions
    joblib.dump(processor.scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()