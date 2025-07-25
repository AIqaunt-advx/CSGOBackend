# Cell 3: Data Processing Class (Clean Version)
# Copy this entire cell into your Jupyter notebook

import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CS2SkinDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['onSaleQuantity', 'seekQuantity', 'price', 'seekPrice']
        self.target_column = 'transactionAmount'
        
    def explore_data(self, df):
        """Basic data exploration and visualization"""
        print("\n=== Data Overview ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for datetime column
        if 'datetime' in df.columns:
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        elif 'timestamp' in df.columns:
            print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print("No datetime/timestamp column found")
        
        print("\n=== Column Info ===")
        print(df.info())
        
        print("\n=== Statistical Summary ===")
        print(df.describe())
        
        print("\n=== Missing Values ===")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Time series (if datetime available)
        if 'datetime' in df.columns and 'price' in df.columns:
            axes[0,0].plot(df['datetime'], df['price'])
            axes[0,0].set_title('Price Over Time')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Price')
        elif 'timestamp' in df.columns and 'price' in df.columns:
            axes[0,0].plot(df['timestamp'], df['price'])
            axes[0,0].set_title('Price Over Time')
            axes[0,0].set_xlabel('Timestamp')
            axes[0,0].set_ylabel('Price')
        else:
            # Show first numeric column distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].hist(bins=50, ax=axes[0,0])
                axes[0,0].set_title(f'Distribution of {numeric_cols[0]}')
            else:
                axes[0,0].text(0.5, 0.5, 'No numeric data for plotting', ha='center', va='center')
                axes[0,0].set_title('No Data Available')
        
        # Plot 2: Volume distributions (if available)
        volume_cols = [col for col in ['onSaleQuantity', 'seekQuantity'] if col in df.columns]
        if len(volume_cols) >= 2:
            axes[0,1].hist(df[volume_cols[0]], bins=50, alpha=0.7, label=volume_cols[0])
            axes[0,1].hist(df[volume_cols[1]], bins=50, alpha=0.7, label=volume_cols[1])
            axes[0,1].set_title('Volume Distributions')
            axes[0,1].set_xlabel('Quantity')
            axes[0,1].legend()
        elif len(volume_cols) == 1:
            df[volume_cols[0]].hist(bins=50, ax=axes[0,1])
            axes[0,1].set_title(f'Distribution of {volume_cols[0]}')
        else:
            # Show distribution of second numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                df[numeric_cols[1]].hist(bins=50, ax=axes[0,1])
                axes[0,1].set_title(f'Distribution of {numeric_cols[1]}')
            else:
                axes[0,1].text(0.5, 0.5, 'No volume data', ha='center', va='center')
                axes[0,1].set_title('Volume Data (Not Available)')
        
        # Plot 3: Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
            axes[1,0].set_title('Correlation Matrix')
        else:
            axes[1,0].text(0.5, 0.5, 'Need 2+ numeric columns', ha='center', va='center')
            axes[1,0].set_title('Correlation (Not Available)')
        
        # Plot 4: Transaction amount over time (if available)
        if 'transactionAmount' in df.columns and df['transactionAmount'].sum() > 0:
            if 'datetime' in df.columns:
                axes[1,1].plot(df['datetime'], df['transactionAmount'])
                axes[1,1].set_xlabel('Date')
            elif 'timestamp' in df.columns:
                axes[1,1].plot(df['timestamp'], df['transactionAmount'])
                axes[1,1].set_xlabel('Timestamp')
            else:
                axes[1,1].plot(df['transactionAmount'])
                axes[1,1].set_xlabel('Index')
            axes[1,1].set_title('Transaction Amount Over Time')
            axes[1,1].set_ylabel('Transaction Amount')
        else:
            # Show any remaining numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            remaining_cols = [col for col in numeric_cols if col not in ['price', volume_cols[0] if volume_cols else '']]
            if remaining_cols:
                df[remaining_cols[0]].plot(ax=axes[1,1])
                axes[1,1].set_title(f'{remaining_cols[0]} Over Index')
            else:
                axes[1,1].text(0.5, 0.5, 'No Transaction Data', ha='center', va='center')
                axes[1,1].set_title('Transaction Amount (No Data)')
        
        plt.tight_layout()
        plt.show()
        
        # Print sample data
        print("\n=== Sample Data ===")
        print(df.head(10))
    
    def create_features(self, df):
        """Engineer features from the base data"""
        df_features = df.copy()
        
        # Check if we have timestamp column for sorting
        if 'timestamp' in df_features.columns:
            df_features = df_features.sort_values('timestamp').reset_index(drop=True)
        elif 'datetime' in df_features.columns:
            df_features = df_features.sort_values('datetime').reset_index(drop=True)
        else:
            print("⚠️ No timestamp column found, using original order")
        
        # Basic ratio features (only if columns exist)
        if 'seekQuantity' in df_features.columns and 'onSaleQuantity' in df_features.columns:
            df_features['demand_ratio'] = df_features['seekQuantity'] / (df_features['onSaleQuantity'] + 1)
            df_features['supply_demand_diff'] = df_features['onSaleQuantity'] - df_features['seekQuantity']
        
        if 'price' in df_features.columns and 'seekPrice' in df_features.columns:
            df_features['price_spread'] = df_features['price'] - df_features['seekPrice']
            df_features['price_spread_pct'] = df_features['price_spread'] / (df_features['seekPrice'] + 0.01)
        
        # Time-based features (only if datetime exists)
        if 'datetime' in df_features.columns:
            df_features['hour'] = df_features['datetime'].dt.hour
            df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
            df_features['month'] = df_features['datetime'].dt.month
            df_features['day_of_year'] = df_features['datetime'].dt.dayofyear
        elif 'timestamp' in df_features.columns:
            # Try to create datetime from timestamp
            try:
                df_features['datetime'] = pd.to_datetime(df_features['timestamp'], unit='s', errors='coerce')
                df_features['hour'] = df_features['datetime'].dt.hour
                df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
                df_features['month'] = df_features['datetime'].dt.month
                df_features['day_of_year'] = df_features['datetime'].dt.dayofyear
            except:
                print("⚠️ Could not create time features from timestamp")
        
        # Rolling window features (only if relevant columns exist)
        if 'price' in df_features.columns:
            df_features['price_ma_7'] = df_features['price'].rolling(window=7, min_periods=1).mean()
            df_features['price_volatility_7'] = df_features['price'].rolling(window=7, min_periods=1).std()
            df_features['price_volatility_7'] = df_features['price_volatility_7'].fillna(0)
            df_features['price_lag_1'] = df_features['price'].shift(1)
        
        if 'onSaleQuantity' in df_features.columns:
            df_features['volume_ma_7'] = df_features['onSaleQuantity'].rolling(window=7, min_periods=1).mean()
            df_features['volume_lag_1'] = df_features['onSaleQuantity'].shift(1)
        
        if 'seekQuantity' in df_features.columns:
            df_features['demand_ma_7'] = df_features['seekQuantity'].rolling(window=7, min_periods=1).mean()
            df_features['demand_lag_1'] = df_features['seekQuantity'].shift(1)
        
        # Log transformations for skewed data (only if columns exist)
        for col in ['onSaleQuantity', 'seekQuantity', 'price']:
            if col in df_features.columns:
                df_features[f'{col}_log'] = np.log1p(df_features[col])
        
        # Fill NaN values from lag features
        df_features = df_features.fillna(method='bfill').fillna(method='ffill')
        
        return df_features
    
    def clean_null_values(self, df):
        """Clean NULL values from the dataset with intelligent strategy"""
        print(f"🧹 Cleaning NULL values...")
        print(f"   Original shape: {df.shape}")
        
        # Check for different types of NULL values
        null_patterns = [
            df.isnull(),  # Standard pandas NULL
            df.isna(),    # Standard pandas NA
            df == 'NULL', # String 'NULL'
            df == 'null', # String 'null'
            df == 'None', # String 'None'
            df == '',     # Empty strings
            df == ' ',    # Space strings
        ]
        
        # Combine all null patterns
        is_null = pd.DataFrame(False, index=df.index, columns=df.columns)
        for pattern in null_patterns:
            try:
                is_null = is_null | pattern
            except:
                continue
        
        # Report NULL statistics
        null_counts = is_null.sum()
        if null_counts.sum() > 0:
            print(f"   NULL values found:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"     {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # INTELLIGENT CLEANING STRATEGY
        
        # Step 1: Drop columns that are mostly NULL (>95%)
        high_null_cols = []
        for col, count in null_counts.items():
            if count > len(df) * 0.95:
                high_null_cols.append(col)
        
        if high_null_cols:
            print(f"   🗑️ Dropping columns with >95% NULL: {high_null_cols}")
            df = df.drop(columns=high_null_cols)
            is_null = is_null.drop(columns=high_null_cols)
            null_counts = is_null.sum()
        
        # Step 2: Fill transaction columns with 0 (they're often legitimately 0)
        transaction_cols = ['transactionAmount', 'transcationNum', 'surviveNum']
        for col in transaction_cols:
            if col in df.columns:
                null_count_before = is_null[col].sum()
                df[col] = df[col].fillna(0)
                is_null[col] = False  # Mark as no longer null
                if null_count_before > 0:
                    print(f"   🔧 Filled {col} NULLs with 0: {null_count_before} values")
        
        # Step 3: Only remove rows with NULLs in ESSENTIAL columns
        essential_cols = ['timestamp', 'price', 'onSaleQuantity', 'seekQuantity', 'seekPrice']
        available_essential = [col for col in essential_cols if col in df.columns]
        
        if available_essential:
            print(f"   🎯 Checking essential columns for NULLs: {available_essential}")
            essential_nulls = is_null[available_essential].any(axis=1)
            clean_df = df[~essential_nulls].copy()
            
            removed_rows = len(df) - len(clean_df)
            print(f"   Removed {removed_rows} rows with NULLs in essential columns ({removed_rows/len(df)*100:.1f}%)")
        else:
            print("   ⚠️ No essential columns found, keeping all rows")
            clean_df = df.copy()
        
        print(f"   Clean shape: {clean_df.shape}")
        
        if len(clean_df) == 0:
            print("❌ All rows have NULLs in essential columns!")
            print("💡 Trying more lenient strategy...")
            
            # Fallback: Only require non-null values in at least 2 numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Keep rows that have at least 2 non-null numeric values
                numeric_null_counts = is_null[numeric_cols].sum(axis=1)
                max_allowed_nulls = len(numeric_cols) - 2
                clean_df = df[numeric_null_counts <= max_allowed_nulls].copy()
                
                removed_rows = len(df) - len(clean_df)
                print(f"   Fallback: Removed {removed_rows} rows, kept rows with ≥2 numeric values")
                print(f"   Fallback clean shape: {clean_df.shape}")
        
        if len(clean_df) == 0:
            raise ValueError("❌ All rows contain too many NULL values! Cannot proceed with training.")
        
        if len(clean_df) < len(df) * 0.1:
            print("⚠️ Warning: Removed >90% of data due to NULL values. Check data quality!")
        
        return clean_df
    
    def prepare_data(self, df, target_col='transactionAmount', test_size=0.2):
        """Prepare data for training"""
        print(f"📊 Input data shape: {df.shape}")
        print(f"📊 Input columns: {list(df.columns)}")
        
        # Step 1: Clean NULL values FIRST
        df_clean = self.clean_null_values(df)
        
        # Step 2: Create features
        df_processed = self.create_features(df_clean)
        print(f"📊 After feature engineering: {df_processed.shape}")
        
        # Step 3: Handle target column
        if target_col not in df_processed.columns or df_processed[target_col].sum() == 0:
            print(f"⚠️ Warning: {target_col} not available or all zeros. Creating synthetic target.")
            
            # Check if we have the necessary columns for synthetic target
            required_cols = ['seekQuantity', 'onSaleQuantity']
            missing_cols = [col for col in required_cols if col not in df_processed.columns]
            
            if missing_cols:
                print(f"⚠️ Missing columns for synthetic target: {missing_cols}")
                # Use available numeric columns to create a simple target
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    col1, col2 = numeric_cols[0], numeric_cols[1]
                    df_processed['synthetic_success'] = (
                        df_processed[col1] * 0.5 + df_processed[col2] * 0.3 + 
                        np.random.normal(0, 0.1, len(df_processed))
                    )
                    print(f"✓ Created synthetic target using {col1} and {col2}")
                else:
                    # Last resort: create random target
                    df_processed['synthetic_success'] = np.random.exponential(10, len(df_processed))
                    print("✓ Created random synthetic target")
            else:
                # Create synthetic success volume based on demand ratio
                df_processed['synthetic_success'] = (
                    df_processed['seekQuantity'] * 
                    np.minimum(1.0, df_processed['onSaleQuantity'] / (df_processed['seekQuantity'] + 1)) * 
                    0.8 + np.random.normal(0, 0.1, len(df_processed))
                )
                print("✓ Created synthetic target based on supply/demand")
            
            df_processed['synthetic_success'] = np.maximum(0, df_processed['synthetic_success'])
            target_col = 'synthetic_success'
        
        # Step 4: Select features for training - be flexible about what's available
        potential_features = [
            'onSaleQuantity', 'seekQuantity', 'price', 'seekPrice',
            'demand_ratio', 'supply_demand_diff', 'price_spread', 'price_spread_pct',
            'hour', 'day_of_week', 'month', 'day_of_year',
            'price_ma_7', 'volume_ma_7', 'demand_ma_7', 'price_volatility_7',
            'price_lag_1', 'volume_lag_1', 'demand_lag_1',
            'onSaleQuantity_log', 'seekQuantity_log', 'price_log'
        ]
        
        # Filter available features
        available_features = [col for col in potential_features if col in df_processed.columns]
        
        # If we don't have many features, use all numeric columns except target
        if len(available_features) < 3:
            print("⚠️ Few engineered features available, using all numeric columns")
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            available_features = [col for col in numeric_cols if col != target_col]
        
        print(f"✓ Available features: {available_features}")
        
        if len(available_features) == 0:
            raise ValueError("No suitable features found for training!")
        
        X = df_processed[available_features]
        y = df_processed[target_col]
        
        # Step 5: Final NULL check and removal (should be minimal after cleaning)
        print(f"🔍 Final NULL check...")
        
        # Check for any remaining NaN/inf values
        X_nulls = X.isnull().sum().sum()
        y_nulls = y.isnull().sum()
        X_infs = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        y_infs = np.isinf(y).sum() if np.issubdtype(y.dtype, np.number) else 0
        
        print(f"   X nulls: {X_nulls}, X infs: {X_infs}")
        print(f"   y nulls: {y_nulls}, y infs: {y_infs}")
        
        if X_nulls > 0 or y_nulls > 0 or X_infs > 0 or y_infs > 0:
            print("⚠️ Found remaining NULL/inf values, removing affected rows...")
            
            # Create mask for valid rows
            valid_mask = (
                ~X.isnull().any(axis=1) & 
                ~y.isnull() & 
                ~np.isinf(X.select_dtypes(include=[np.number])).any(axis=1)
            )
            
            if np.issubdtype(y.dtype, np.number):
                valid_mask = valid_mask & ~np.isinf(y)
            
            X = X[valid_mask]
            y = y[valid_mask]
            
            print(f"   Final clean shape: X={X.shape}, y={len(y)}")
        
        # Step 6: Ensure we have enough data
        if len(X) < 50:
            raise ValueError(f"❌ Too few samples after cleaning: {len(X)}. Need at least 50 samples.")
        
        print(f"✓ Features selected: {len(available_features)}")
        print(f"✓ Training samples: {len(X)}")
        print(f"✓ Target column: {target_col}")
        print(f"✓ Target stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
        
        # Step 7: Split data chronologically (time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        # Step 8: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Final validation - check for any NaN in scaled data
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            print("❌ NaN values found in scaled data! This shouldn't happen.")
            raise ValueError("Scaling produced NaN values")
        
        print("✅ Data preparation completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, available_features

# Initialize the processor
processor = CS2SkinDataProcessor()
print("✓ Data processor initialized")