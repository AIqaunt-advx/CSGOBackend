import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class SkinDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['onSaleQuantity', 'seekQuantity', 'price', 'seekPrice']
        self.target_column = 'transactionAmount'
        
    def load_single_json(self, file_path):
        """Load single JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('success') and 'data' in data:
                df = pd.DataFrame(data['data'])
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                
                # Fill null values with 0 for transaction metrics
                df['transactionAmount'] = df['transactionAmount'].fillna(0)
                df['transcationNum'] = df['transcationNum'].fillna(0)
                df['surviveNum'] = df['surviveNum'].fillna(0)
                
                return df
            else:
                print(f"Invalid JSON structure in {file_path}")
                return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_multiple_json_files(self, file_pattern="*.json"):
        """Load multiple JSON files and combine them"""
        json_files = glob.glob(file_pattern)
        print(f"Found {len(json_files)} JSON files")
        
        all_data = []
        for file_path in json_files:
            df = self.load_single_json(file_path)
            if df is not None:
                df['source_file'] = file_path
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined data: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            return combined_df
        else:
            print("No valid data found")
            return None
    
    def explore_data(self, df):
        """Basic data exploration and visualization"""
        print("\n=== Data Overview ===")
        print(df.info())
        print("\n=== Statistical Summary ===")
        print(df.describe())
        
        # Check for missing values
        print("\n=== Missing Values ===")
        print(df.isnull().sum())
        
        # Correlation analysis
        if all(col in df.columns for col in self.feature_columns + [self.target_column]):
            print("\n=== Correlation Matrix ===")
            corr_matrix = df[self.feature_columns + [self.target_column]].corr()
            print(corr_matrix)
            
            # Visualize correlations
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png')
            plt.show()
    
    def create_features(self, df):
        """Engineer additional features from the base data"""
        df_features = df.copy()
        
        # Sort by timestamp for time-series features
        df_features = df_features.sort_values('timestamp').reset_index(drop=True)
        
        # Basic ratio features
        df_features['demand_ratio'] = df_features['seekQuantity'] / (df_features['onSaleQuantity'] + 1)
        df_features['supply_demand_diff'] = df_features['onSaleQuantity'] - df_features['seekQuantity']
        df_features['price_spread'] = df_features['price'] - df_features['seekPrice']
        df_features['price_spread_pct'] = df_features['price_spread'] / (df_features['seekPrice'] + 0.01)
        
        # Time-based features
        df_features['hour'] = df_features['datetime'].dt.hour
        df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
        df_features['month'] = df_features['datetime'].dt.month
        
        # Rolling window features (7-day windows)
        df_features['price_ma_7'] = df_features['price'].rolling(window=7, min_periods=1).mean()
        df_features['volume_ma_7'] = df_features['onSaleQuantity'].rolling(window=7, min_periods=1).mean()
        df_features['demand_ma_7'] = df_features['seekQuantity'].rolling(window=7, min_periods=1).mean()
        
        # Price volatility
        df_features['price_volatility_7'] = df_features['price'].rolling(window=7, min_periods=1).std()
        
        # Lag features (previous day values)
        df_features['price_lag_1'] = df_features['price'].shift(1)
        df_features['volume_lag_1'] = df_features['onSaleQuantity'].shift(1)
        df_features['demand_lag_1'] = df_features['seekQuantity'].shift(1)
        
        # Success rate (if we have historical data)
        if 'transactionAmount' in df_features.columns:
            df_features['success_rate'] = df_features['transactionAmount'] / (df_features['seekQuantity'] + 1)
        
        # Log transformations for skewed data
        for col in ['onSaleQuantity', 'seekQuantity', 'price', 'transactionAmount']:
            if col in df_features.columns:
                df_features[f'{col}_log'] = np.log1p(df_features[col])
        
        return df_features
    
    def prepare_data(self, df, test_size=0.2, random_state=42, target_col='transactionAmount'):
        """Prepare data for training"""
        # Create features
        df_processed = self.create_features(df)
        
        # Remove rows where target is null (no transaction data)
        df_processed = df_processed.dropna(subset=[target_col])
        
        if len(df_processed) == 0:
            raise ValueError(f"No valid data found for target column {target_col}")
        
        # Select features for training
        feature_cols = [
            'onSaleQuantity', 'seekQuantity', 'price', 'seekPrice',
            'demand_ratio', 'supply_demand_diff', 'price_spread', 'price_spread_pct',
            'hour', 'day_of_week', 'month',
            'price_ma_7', 'volume_ma_7', 'demand_ma_7', 'price_volatility_7',
            'price_lag_1', 'volume_lag_1', 'demand_lag_1',
            'onSaleQuantity_log', 'seekQuantity_log', 'price_log'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df_processed.columns]
        
        X = df_processed[available_features]
        y = df_processed[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        print(f"Features selected: {len(available_features)}")
        print(f"Training samples: {len(X)}")
        
        # Split data chronologically (time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, available_features