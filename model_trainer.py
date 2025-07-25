import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

class SkinVolumePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def initialize_models(self):
        """Initialize different ML models for comparison"""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                verbose=-1
            ),
            'svr': SVR(kernel='rbf', C=1.0)
        }
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def train_and_compare_models(self, X_train, X_test, y_train, y_test, feature_names):
        """Train multiple models and compare performance"""
        self.feature_names = feature_names
        self.initialize_models()
        
        results = {}
        trained_models = {}
        
        print("Training and evaluating models...")
        print("-" * 60)
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test)
                results[name] = metrics
                
                print(f"{name:20} | R2: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        # Find best model based on R2 score
        best_name = max(results.keys(), key=lambda x: results[x]['R2'])
        self.best_model = trained_models[best_name]
        self.best_model_name = best_name
        
        print(f"\nBest model: {best_name} (R2: {results[best_name]['R2']:.4f})")
        
        return results, trained_models
    
    def plot_predictions(self, X_test, y_test):
        """Plot actual vs predicted values"""
        if self.best_model is None:
            print("No trained model available")
            return
        
        y_pred = self.best_model.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Success Sales Volume')
        plt.ylabel('Predicted Success Sales Volume')
        plt.title(f'Actual vs Predicted - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig('predictions_plot.png')
        plt.show()
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model is None or self.feature_names is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()
            
            return feature_importance
        
        return None
    
    def predict(self, on_sale_qty, seek_qty, price=3.0, seek_price=2.8):
        """Make prediction for new data"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Create feature array matching training features
        demand_ratio = seek_qty / (on_sale_qty + 1)
        supply_demand_diff = on_sale_qty - seek_qty
        price_spread = price - seek_price
        price_spread_pct = price_spread / (seek_price + 0.01)
        
        # Basic features (adjust based on available features during training)
        features = np.array([[
            on_sale_qty,                    # onSaleQuantity
            seek_qty,                       # seekQuantity  
            price,                          # price
            seek_price,                     # seekPrice
            demand_ratio,                   # demand_ratio
            supply_demand_diff,             # supply_demand_diff
            price_spread,                   # price_spread
            price_spread_pct,               # price_spread_pct
            12,                             # hour (default noon)
            1,                              # day_of_week (default Tuesday)
            6,                              # month (default June)
            price,                          # price_ma_7 (use current price)
            on_sale_qty,                    # volume_ma_7 (use current volume)
            seek_qty,                       # demand_ma_7 (use current demand)
            0.1,                            # price_volatility_7 (default low volatility)
            price,                          # price_lag_1
            on_sale_qty,                    # volume_lag_1
            seek_qty,                       # demand_lag_1
            np.log1p(on_sale_qty),         # onSaleQuantity_log
            np.log1p(seek_qty),            # seekQuantity_log
            np.log1p(price)                # price_log
        ]])
        
        prediction = self.best_model.predict(features)[0]
        return max(0, prediction)  # Ensure non-negative prediction
    
    def save_model(self, filepath='skin_volume_model.pkl'):
        """Save the trained model"""
        if self.best_model is None:
            print("No trained model to save")
            return
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='skin_volume_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")