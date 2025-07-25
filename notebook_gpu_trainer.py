# Cell 3: GPU-Optimized Model Training
# Copy this entire cell into your Jupyter notebook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")
        
        self.models = {}
        self.best_model = None
        self.best_score = float('-inf')
        self.best_model_name = None
        
    def train_xgboost_gpu(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with GPU acceleration"""
        print("🚀 Training XGBoost with GPU...")
        
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
        
        model = xgb.XGBRegressor(**params)
        
        # Train with early stopping
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
        
        print(f"✓ XGBoost GPU R² Score: {score:.4f}")
        return model, score
    
    def train_lightgbm_gpu(self, X_train, X_test, y_train, y_test):
        """Train LightGBM with GPU acceleration"""
        print("🚀 Training LightGBM with GPU...")
        
        params = {
            'objective': 'regression',
            'device': 'gpu',
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
        
        print(f"✓ LightGBM GPU R² Score: {score:.4f}")
        return model, score
    
    def train_catboost_gpu(self, X_train, X_test, y_train, y_test):
        """Train CatBoost with GPU acceleration"""
        print("🚀 Training CatBoost with GPU...")
        
        params = {
            'task_type': 'GPU',
            'devices': '0',
            'depth': 8,
            'learning_rate': 0.1,
            'iterations': 1000,
            'random_seed': 42,
            'verbose': False
        }
        
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
        
        print(f"✓ CatBoost GPU R² Score: {score:.4f}")
        return model, score
    
    def train_deep_network(self, X_train, X_test, y_train, y_test, epochs=200, batch_size=256):
        """Train deep neural network"""
        print("🚀 Training Deep Neural Network...")
        
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
        train_losses = []
        val_losses = []
        
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
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            
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
            'predictions': y_pred,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"✓ Deep Network R² Score: {score:.4f}")
        return model, score
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest (CPU baseline)"""
        print("🌲 Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        self.models['random_forest'] = {
            'model': model,
            'score': score,
            'predictions': y_pred
        }
        
        print(f"✓ Random Forest R² Score: {score:.4f}")
        return model, score
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all available models"""
        print("=" * 60)
        print("🎯 Training all GPU-optimized models...")
        print("=" * 60)
        
        # Train all models
        try:
            self.train_xgboost_gpu(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ XGBoost GPU failed: {e}")
        
        try:
            self.train_lightgbm_gpu(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ LightGBM GPU failed: {e}")
        
        try:
            self.train_catboost_gpu(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ CatBoost GPU failed: {e}")
        
        try:
            self.train_deep_network(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Deep Network failed: {e}")
        
        try:
            self.train_random_forest(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Random Forest failed: {e}")
        
        # Find best model
        if self.models:
            best_name = max(self.models.keys(), key=lambda x: self.models[x]['score'])
            self.best_model = self.models[best_name]
            self.best_score = self.best_model['score']
            self.best_model_name = best_name
            
            print(f"\n🏆 Best model: {best_name} (R² Score: {self.best_score:.4f})")
        
        return self.models
    
    def plot_results(self, y_test):
        """Plot training results and comparisons"""
        if not self.models:
            print("No models trained yet!")
            return
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, (name, model_info) in enumerate(self.models.items()):
            if i >= len(axes):
                break
                
            y_pred = model_info['predictions']
            score = model_info['score']
            
            axes[i].scatter(y_test, y_pred, alpha=0.6)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f'{name}\nR² = {score:.4f}')
        
        # Hide unused subplots
        for i in range(len(self.models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Plot model comparison
        if len(self.models) > 1:
            model_names = list(self.models.keys())
            scores = [self.models[name]['score'] for name in model_names]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, scores)
            plt.title('Model Performance Comparison')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45)
            
            # Highlight best model
            best_idx = scores.index(max(scores))
            bars[best_idx].set_color('gold')
            
            plt.tight_layout()
            plt.show()
    
    def save_best_model(self, filename='best_cs2_model.pkl'):
        """Save the best performing model"""
        if self.best_model is None:
            print("No trained model to save!")
            return
        
        model_data = {
            'model': self.best_model['model'],
            'model_name': self.best_model_name,
            'score': self.best_score
        }
        
        joblib.dump(model_data, filename)
        print(f"✓ Best model ({self.best_model_name}) saved to {filename}")

# Initialize the trainer
trainer = GPUSkinTrainer()
print("✓ GPU trainer initialized")