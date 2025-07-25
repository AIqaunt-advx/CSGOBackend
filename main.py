#!/usr/bin/env python3
"""
CS2 Skin Trading Volume Predictor
Main script to train and use the AI model for predicting skin trading success volumes
"""

import pandas as pd
import numpy as np
from data_processor import SkinDataProcessor
from model_trainer import SkinVolumePredictor
import requests
import json

def create_sample_data():
    """Create sample data for demonstration (replace with your actual data loading)"""
    np.random.seed(42)
    n_samples = 2000
    
    # Generate synthetic data that mimics real trading patterns
    data = []
    for i in range(n_samples):
        current_vol = np.random.exponential(50) + 10
        wanted_vol = current_vol * np.random.uniform(0.3, 1.5)
        
        # Success volume depends on supply/demand dynamics with some noise
        demand_ratio = wanted_vol / current_vol
        base_success = wanted_vol * min(1.0, current_vol / wanted_vol) * 0.8
        noise = np.random.normal(0, base_success * 0.1)
        success_vol = max(0, base_success + noise)
        
        data.append({
            'current_sales_volume': current_vol,
            'wanted_sales_volume': wanted_vol,
            'success_sales_volume': success_vol
        })
    
    return pd.DataFrame(data)

def train_model(data_file=None):
    """Train the skin volume prediction model"""
    print("=== CS2 Skin Trading Volume Predictor ===")
    
    # Initialize processors
    processor = SkinDataProcessor()
    predictor = SkinVolumePredictor()
    
    # Load data
    if data_file:
        if data_file.endswith('.json'):
            df = processor.load_single_json(data_file)
        else:
            # Try to load multiple JSON files
            df = processor.load_multiple_json_files(data_file)
    else:
        print("Using sample data for demonstration...")
        df = create_sample_data()
    
    if df is None:
        print("Failed to load data")
        return None, None
    
    # Explore data
    processor.explore_data(df)
    
    # Prepare data for training (use transactionAmount as target)
    try:
        X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(df, target_col='transactionAmount')
    except ValueError as e:
        print(f"Error: {e}")
        print("Trying with transcationNum as target...")
        try:
            X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(df, target_col='transcationNum')
        except ValueError:
            print("No valid transaction data found. Using synthetic target...")
            # Create synthetic success volume based on demand ratio
            df['synthetic_success'] = df['seekQuantity'] * np.minimum(1.0, df['onSaleQuantity'] / (df['seekQuantity'] + 1)) * 0.8
            X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(df, target_col='synthetic_success')
    
    # Train and compare models
    results, models = predictor.train_and_compare_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Visualize results
    predictor.plot_predictions(X_test, y_test)
    feature_importance = predictor.get_feature_importance()
    
    if feature_importance is not None:
        print("\nFeature Importance:")
        print(feature_importance)
    
    # Save the best model
    predictor.save_model()
    
    return predictor, processor

def predict_from_api(api_url=None, current_volume=None, wanted_volume=None):
    """Make prediction using market API data or manual input"""
    # Load trained model
    predictor = SkinVolumePredictor()
    predictor.load_model()
    
    if api_url:
        try:
            # Fetch data from API
            response = requests.get(api_url)
            data = response.json()
            current_volume = data.get('current_sales_volume')
            wanted_volume = data.get('wanted_sales_volume')
        except Exception as e:
            print(f"Error fetching API data: {e}")
            return None
    
    if current_volume is None or wanted_volume is None:
        print("Please provide current_volume and wanted_volume")
        return None
    
    # Make prediction
    predicted_success = predictor.predict(current_volume, wanted_volume)
    
    print(f"\n=== Prediction Results ===")
    print(f"Current Sales Volume: {current_volume:.2f}")
    print(f"Wanted Sales Volume: {wanted_volume:.2f}")
    print(f"Predicted Success Volume: {predicted_success:.2f}")
    
    # Calculate trading insights
    success_rate = predicted_success / wanted_volume if wanted_volume > 0 else 0
    demand_ratio = wanted_volume / current_volume if current_volume > 0 else 0
    
    print(f"\n=== Trading Insights ===")
    print(f"Predicted Success Rate: {success_rate:.2%}")
    print(f"Demand Ratio: {demand_ratio:.2f}")
    
    if success_rate > 0.8:
        print("🟢 HIGH SUCCESS PROBABILITY - Good buying opportunity")
    elif success_rate > 0.5:
        print("🟡 MODERATE SUCCESS PROBABILITY - Consider market conditions")
    else:
        print("🔴 LOW SUCCESS PROBABILITY - High risk")
    
    return predicted_success

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            data_file = sys.argv[2] if len(sys.argv) > 2 else None
            train_model(data_file)
            
        elif command == "predict":
            if len(sys.argv) >= 4:
                current_vol = float(sys.argv[2])
                wanted_vol = float(sys.argv[3])
                predict_from_api(current_volume=current_vol, wanted_volume=wanted_vol)
            else:
                print("Usage: python main.py predict <current_volume> <wanted_volume>")
                
        elif command == "api_predict":
            api_url = sys.argv[2] if len(sys.argv) > 2 else None
            predict_from_api(api_url=api_url)
    else:
        print("Usage:")
        print("  python main.py train [data_file.csv]")
        print("  python main.py predict <current_volume> <wanted_volume>")
        print("  python main.py api_predict <api_url>")
        print("\nRunning training with sample data...")
        train_model()